import os
import csv

import matplotlib

matplotlib.use("Agg")

import hardflow.utils
import tqdm
import tyro

import numpy as np
import torch
from torch import nn

from hardflow.config.flow_matching import FlowMatchingEvaluationConfig
from hardflow.datasets.sequence import GoalDataset
from hardflow.models_flow.flow_policy import FlowPolicy
from hardflow.models_flow.unet import TemporalUnet, WrappedFlowUnet
from run.utils import deterministic, save_config, set_cuda_visible_device

try:
    import casadi as cs
    import l4casadi as l4c
except ImportError:
    cs = None
    l4c = None


class ProxyValueModel(nn.Module):

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        state_dim: int,
        objective: str = "",
        constraint: str = "",
        ellipse_margin: float = 0.0,
        value_objective_scale: float = 1.0,
        value_constraint_scale: float = 1.0,
        normalizer=None,
        dynamics_model=None,
    ):

        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.transition_dim = action_dim + state_dim

        self.objective = objective
        self.constraint = constraint
        self.ellipse_margin = ellipse_margin
        self.value_objective_scale = value_objective_scale
        self.value_constraint_scale = value_constraint_scale
        self.normalizer = normalizer
        self.dynamics_model = dynamics_model

        if normalizer is not None:
            self.obs_norm_mins = normalizer.normalizers["observations"].mins
            self.obs_norm_maxs = normalizer.normalizers["observations"].maxs

    def forward(
        self,
        x,
    ):
        """
        x: (batch_size, planning_horizon, transition_dim), should be normalized
        value: (batch_size, 1)
        """

        if self.objective == "":
            objective_value = torch.zeros(
                x.shape[0], 1, device=x.device, requires_grad=True
            )
        elif self.objective == "distance":
            objective_value = self._compute_distance_objective(x)
        else:
            raise NotImplementedError(f"Objective {self.objective} is not implemented.")

        if self.constraint == "":
            constraint_penalty = torch.zeros(
                x.shape[0], 1, device=x.device, requires_grad=True
            )
        elif self.constraint == "ellipses":
            constraint_values = self._compute_ellipse_constraints(x)
            constraint_violations = torch.clamp(-constraint_values, min=0.0)
            constraint_penalty = torch.sum(
                constraint_violations**2, dim=1, keepdim=True
            )
        elif self.constraint == "ellipses_and_dynamics":
            constraint_values = self._compute_ellipse_constraints(x)
            constraint_violations = torch.clamp(-constraint_values, min=0.0)
            constraint_penalty = torch.sum(
                constraint_violations**2, dim=1, keepdim=True
            )

            dynamics_constraint_values = self._compute_dynamics_constraints(x)
            dynamics_penalty = torch.sum(
                dynamics_constraint_values**2, dim=1, keepdim=True
            )
            constraint_penalty += dynamics_penalty
        else:
            raise NotImplementedError(
                f"Constraint {self.constraint} is not implemented."
            )

        value = (
            objective_value * self.value_objective_scale
            - constraint_penalty * self.value_constraint_scale
        )

        return value

    def _compute_distance_objective(self, x):
        """
        x: (batch_size, planning_horizon, transition_dim)
        objective_value: (batch_size, 1)
        """
        batch_size = x.shape[0]
        observations = x[
            :, :, self.action_dim :
        ]  # (batch_size, planning_horizon, state_dim)

        distance_objective = torch.zeros(
            batch_size, 1, device=x.device, requires_grad=True
        )

        # compute distance between consecutive positions
        for i in range(self.horizon - 1):
            current_pos = observations[:, i, :2]  # (batch_size, 2)
            next_pos = observations[:, i + 1, :2]  # (batch_size, 2)
            step_distance = torch.sum(
                (current_pos - next_pos) ** 2, dim=1, keepdim=True
            )  # (batch_size, 1)
            distance_objective = distance_objective + step_distance

        distance_objective = -1.0 * distance_objective

        return distance_objective

    def _compute_ellipse_constraints(self, x):
        """
        x: (batch_size, planning_horizon, transition_dim)
        constraint_values: (batch_size, num_constraints), >= 0 means feasible (outside ellipse)
        """
        batch_size = x.shape[0]
        observations = x[:, :, self.action_dim :]

        if self.normalizer is not None:
            obs_norm_min_0 = float(self.obs_norm_mins[0])
            obs_norm_max_0 = float(self.obs_norm_maxs[0])
            obs_norm_min_1 = float(self.obs_norm_mins[1])
            obs_norm_max_1 = float(self.obs_norm_maxs[1])
        else:
            raise ValueError("Normalizer is required for ellipse constraints.")

        xa = 2.0 * 0.8 / (obs_norm_max_1 - obs_norm_min_1)
        ya = 2.0 * 0.8 / (obs_norm_max_0 - obs_norm_min_0)
        off_xa = 2.0 * (5.3 - obs_norm_min_1) / (obs_norm_max_1 - obs_norm_min_1) - 1.0
        off_ya = 2.0 * (4.5 - obs_norm_min_0) / (obs_norm_max_0 - obs_norm_min_0) - 1.0

        xb = 2.0 * 0.8 / (obs_norm_max_1 - obs_norm_min_1)
        yb = 2.0 * 0.8 / (obs_norm_max_0 - obs_norm_min_0)
        off_xb = 2.0 * (4.8 - obs_norm_min_1) / (obs_norm_max_1 - obs_norm_min_1) - 1.0
        off_yb = 2.0 * (1.5 - obs_norm_min_0) / (obs_norm_max_0 - obs_norm_min_0) - 1.0

        constraint_values = []

        for i in range(1, self.horizon - 1):

            x_pos = observations[:, i, 1]  # (batch_size,)
            y_pos = observations[:, i, 0]  # (batch_size,)

            ellipse1_constraint = (
                (
                    (1.0 / xa * (x_pos - off_xa)) ** 2
                    + (1.0 / ya * (y_pos - off_ya)) ** 2
                )
                - 1.0
                - self.ellipse_margin
            )  # (batch_size,)

            ellipse2_constraint = (
                (
                    (1.0 / xb * (x_pos - off_xb)) ** 4
                    + (1.0 / yb * (y_pos - off_yb)) ** 4
                )
                - 1.0
                - self.ellipse_margin
            )  # (batch_size,)

            constraint_values.append(
                ellipse1_constraint.unsqueeze(1)
            )  # (batch_size, 1)
            constraint_values.append(
                ellipse2_constraint.unsqueeze(1)
            )  # (batch_size, 1)

        if len(constraint_values) == 0:
            return torch.zeros(batch_size, 0, device=x.device, requires_grad=True)

        constraint_values = torch.cat(constraint_values, dim=1)
        return constraint_values

    def _compute_dynamics_constraints(self, x):
        """
        x: (batch_size, planning_horizon, transition_dim)
        constraint_values: (batch_size, num_constraints), = 0 means feasible
        """
        if self.dynamics_model is None:
            return torch.zeros(x.shape[0], 0, device=x.device, requires_grad=True)

        batch_size = x.shape[0]
        actions = x[:, :, : self.action_dim]
        observations = x[:, :, self.action_dim :]

        A_torch = torch.tensor(self.dynamics_model["A"], device=x.device, dtype=x.dtype)
        B_torch = torch.tensor(self.dynamics_model["B"], device=x.device, dtype=x.dtype)
        c_torch = torch.tensor(self.dynamics_model["c"], device=x.device, dtype=x.dtype)

        constraint_values = []

        for i in range(self.horizon - 3):
            current_state = observations[:, i, :]
            next_state = observations[:, i + 1, :]
            current_action = actions[:, i, :]

            predicted_next_state = (
                torch.matmul(current_state, A_torch.T)
                + torch.matmul(current_action, B_torch.T)
                + c_torch.unsqueeze(0)
            )

            dynamics_constraint = predicted_next_state - next_state
            constraint_values.append(dynamics_constraint)

        if len(constraint_values) == 0:
            return torch.zeros(batch_size, 0, device=x.device, requires_grad=True)

        constraint_values = torch.cat(constraint_values, dim=1)
        return constraint_values


def create_constrained_casadi_functions(
    casadi_flow_fn,
    action_dim: int,
    state_dim: int,
    horizon: int,
):

    print(
        f"create_constrained_casadi_function: horizon={horizon}, action_dim={action_dim}, state_dim={state_dim}"
    )

    transition_dim = action_dim + state_dim
    N_full = horizon * transition_dim
    dof = N_full - state_dim * 2
    print(
        f"Calculated dimensions: transition_dim={transition_dim}, N_full={N_full}, dof={dof}"
    )

    dof_sym = cs.MX.sym("dof_sym", 1, dof)
    s0_sym = cs.MX.sym("s0_sym", 1, state_dim)
    sH_sym = cs.MX.sym("sH_sym", 1, state_dim)

    full_traj_sym = cs.horzcat(
        dof_sym[:, :action_dim], s0_sym, dof_sym[:, action_dim:], sH_sym
    )
    print(f"full_traj_sym shape: {full_traj_sym.shape}")

    time_sym = cs.MX.sym("t")
    flow_input_sym = cs.horzcat(full_traj_sym, time_sym)

    full_flow_output_sym = casadi_flow_fn(flow_input_sym)

    dof_flow_output_sym = cs.horzcat(
        full_flow_output_sym[:, :action_dim],
        full_flow_output_sym[:, action_dim + state_dim : N_full - state_dim],
    )

    constrained_flow_fn = cs.Function(
        "constrained_flow_fn",
        [dof_sym, time_sym, s0_sym, sH_sym],
        [dof_flow_output_sym],
    )

    dummy_dof = cs.DM.zeros(1, dof)
    dummy_time = cs.DM(0.0)
    dummy_s0 = cs.DM.zeros(1, state_dim)
    dummy_sH = cs.DM.zeros(1, state_dim)

    _ = constrained_flow_fn(dummy_dof, dummy_time, dummy_s0, dummy_sH)

    return constrained_flow_fn


def run_env(
    env,
    policy,
    cfg: FlowMatchingEvaluationConfig,
    run_id=0,
):
    assert (
        env.name == "maze2d-large-v1"
    ), "This script is designed to run maze2d-large-v1."

    renderer = hardflow.utils.Maze2dRenderer("maze2d-large-v1", env)

    observation = env.reset()

    if cfg.fixed_start:
        observation = np.array([0.94875744, 2.93648809, -0.01347715, 0.06358764])
        env.set_state(observation[0:2], observation[2:4])

    target = env._target
    conditions = {
        cfg.horizon - 1: np.array([*target, 0.0, 0]),
    }

    rollout = [observation.copy()]
    computation_times = []

    save_path = os.path.join(
        cfg.log_folder,
        cfg.env,
        "eval",
        cfg.exp_name,
    )

    total_rewards = 0
    score = 0
    total_violations = 0

    state_sequence = None
    action_sequence = None

    pbar = tqdm.tqdm(range(env.max_episode_steps), desc="Episode")
    for t in pbar:

        # only plan once at t=0
        if t == 0:
            conditions[0] = observation
            (
                action,
                samples,
                policy_x_chain,
                policy_x1_estimation,
                info,
            ) = policy(conditions, batch_size=cfg.batch_size)

            if "computation_time" in info:
                computation_times.append(info["computation_time"])

            state_sequence = samples.observations[0]
            action_sequence = samples.actions[0]

            # image of final sample
            renderer.composite(
                os.path.join(save_path, f"{run_id}.png"),
                samples.observations[:1],
                ncol=1,
            )

            #save all process waypoints
            if(1):
                # images of all diffusion timesteps from policy_x_chain
                diff_save_dir = os.path.join(save_path, f"{run_id}_chain")
                os.makedirs(diff_save_dir, exist_ok=True)
                num_diffusion_steps = policy_x_chain.shape[1]
                for step_idx in range(num_diffusion_steps):
                    step_observations = policy_x_chain[
                        0:1, step_idx, :, policy.action_dim :
                    ]
                    renderer.composite(
                        os.path.join(diff_save_dir, f"step_{step_idx:03d}.png"),
                        step_observations,
                        ncol=1,
                    )
                
            #save all process waypoints (x1 estimation)
            if(1):
                # images of all diffusion timesteps from policy_x1_estimation
                diff_save_dir = os.path.join(save_path, f"{run_id}_x1_estimation")
                os.makedirs(diff_save_dir, exist_ok=True)
                num_diffusion_steps = policy_x1_estimation.shape[1]
                for step_idx in range(num_diffusion_steps):
                    step_observations = policy_x1_estimation[
                        0:1, step_idx, :, policy.action_dim :
                    ]
                    renderer.composite(
                        os.path.join(diff_save_dir, f"step_{step_idx:03d}.png"),
                        step_observations,
                        ncol=1,
                    )
            print(f"Mean distance of a point in step {step_idx} and in final step: {np.linalg.norm(step_observations - samples.observations[:1], axis=-1).mean()}")
    

        if cfg.controller == "pd":  # proportional-derivative
            px, py, vx, vy = observation
            idx_t = min(t, len(state_sequence) - 1)
            pdes_t = state_sequence[idx_t][:2]
            vdes_t = state_sequence[idx_t][2:]
            Kp = np.array([1.0, 1.0]) * 5.0
            Kd = np.array([1.0, 1.0]) * 1.0
            pos_err = pdes_t - np.array([px, py])
            vel_err = vdes_t - np.array([vx, vy])
            action = Kp * pos_err + Kd * vel_err
        else:
            raise NotImplementedError

        next_observation, reward, _, _ = env.step(action)

        if cfg.render:
            env.render()

        total_rewards += reward
        score = env.get_normalized_score(total_rewards)
        rollout.append(next_observation.copy())
        observation = next_observation

        # update progress bar
        violation = check_violation(observation, cfg.constraint)
        total_violations += violation
        pbar.set_postfix(
            {
                "reward": f"{reward:.3f}",
                "total rewards": f"{total_rewards:.3f}",
                "score": f"{score:.3f}",
                "violation": f"{violation:.3f}",
                "total violations": f"{total_violations:.3f}",
            }
        )

        # if np.linalg.norm(observation[0:2] - env._target) <= 0.5:
        #     break

    real_save_path = os.path.join(save_path, f"{run_id}_real.png")
    import matplotlib.pyplot as plt

    background = env.maze_arr == 10
    extent = (0, 1, 1, 0)
    obs = np.array(rollout)
    obs = obs + 0.7
    iscale, jscale = 9, 12
    obs[:, 0] /= iscale
    obs[:, 1] /= jscale

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    fig.set_dpi(300)

    plt.imshow(background * 0.5, extent=extent, cmap=plt.cm.binary, vmin=0, vmax=1)
    ax = plt.gca()
    ax.set_position([0, 0, 1, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    theta = np.linspace(0, 2 * np.pi, 200)
    x1 = 0.8 / 12.0 * np.cos(theta) + 6.0 / 12.0
    y1 = 0.8 / 9.0 * np.sin(theta) + 5.2 / 9.0
    plt.fill(
        x1,
        y1,
        facecolor="red",
        edgecolor="none",
        alpha=0.35,
        zorder=5,
        label="Obstacles",
    )
    x2 = (
        0.8 / 12.0 * np.sqrt(np.abs(np.cos(theta))) * np.sign(np.cos(theta))
        + 5.5 / 12.0
    )
    y2 = 0.8 / 9.0 * np.sqrt(np.abs(np.sin(theta))) * np.sign(np.sin(theta)) + 2.2 / 9.0
    plt.fill(
        x2,
        y2,
        facecolor="red",
        edgecolor="none",
        alpha=0.35,
        zorder=5,
        label="_nolegend_",
    )

    plt.plot(
        obs[:, 1], obs[:, 0], c="blue", linewidth=0.9, zorder=12, label="Trajectory"
    )

    plt.scatter(
        obs[0, 1],
        obs[0, 0],
        c="limegreen",
        edgecolors="black",
        linewidths=0.5,
        s=50,
        marker="o",
        zorder=30,
        label="Start",
    )
    plt.scatter(
        obs[-1, 1],
        obs[-1, 0],
        c="orange",
        edgecolors="black",
        linewidths=0.6,
        s=90,
        marker="*",
        zorder=30,
        label="End",
    )

    plt.axis("off")
    plt.legend(loc="best", fontsize=11, framealpha=0.9)
    fig.savefig(real_save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    avg_computation_time = np.mean(computation_times) if computation_times else None

    return score, total_violations, np.array(rollout), avg_computation_time


def check_violation(observation, constraint):

    if constraint == "ellipses" or constraint == "ellipses_and_dynamics":
        y, x = observation[0], observation[1]

        center1_x = 5.3
        center1_y = 4.5
        radius1_x = 0.8
        radius1_y = 0.8

        ellipse1_check = ((x - center1_x) / radius1_x) ** 2 + (
            (y - center1_y) / radius1_y
        ) ** 2
        if ellipse1_check < 1.0:
            return 1.0

        center2_x = 4.8
        center2_y = 1.5
        radius2_x = 0.8
        radius2_y = 0.8

        ellipse2_check = ((x - center2_x) / radius2_x) ** 4 + (
            (y - center2_y) / radius2_y
        ) ** 4
        if ellipse2_check < 1.0:
            return 1.0

        return 0.0

    else:
        raise NotImplementedError(f"Constraint {constraint} is not implemented.")


def evaluate(cfg: FlowMatchingEvaluationConfig):

    # get dataset (only for its normalizer)
    dataset = GoalDataset(
        env=cfg.env,
        horizon=cfg.horizon,
        normalizer=cfg.normalizer,
        preprocess_fns=cfg.preprocess_fns,
        max_path_length=cfg.max_path_length,
        max_n_episodes=cfg.max_n_episodes,
        termination_penalty=0,
        seed=cfg.seed,
    )
    normalizer = dataset.normalizer

    # load dynamics if available
    dynamics_model = None
    dynamics_path = os.path.join("logs", cfg.env, "dynamics", "linear_model.npz")
    if os.path.exists(dynamics_path) and "dynamics" in cfg.constraint:
        print(f"Loading fitted dynamics from {dynamics_path}")
        dynamics_data = np.load(dynamics_path, allow_pickle=True)
        dynamics_model = {
            "A": dynamics_data["A"],
            "B": dynamics_data["B"],
            "c": dynamics_data["c"],
            "normalizer": dynamics_data["normalizer"].item(),
        }
        print("Fitted dynamics loaded successfully")
    else:
        print("No fitted dynamics found, proceeding without dynamics model")

    flow_model = TemporalUnet(
        horizon=cfg.horizon,
        transition_dim=cfg.state_dim + cfg.action_dim,
        cond_dim=cfg.state_dim,
        dim=32,
        dim_mults=(1, 4, 8),
        attention=False,
    ).to(cfg.device)
    flow_model.load_state_dict(
        torch.load(
            os.path.join(
                cfg.log_folder,
                cfg.env,
                "flow",
                cfg.flow_exp_name,
                f"model_ema_{cfg.flow_cp}.pth",
            ),
            map_location='cpu'
        )
    )

    value_model = ProxyValueModel(
        cfg.horizon,
        cfg.action_dim,
        cfg.state_dim,
        objective=cfg.value_objective,
        constraint=cfg.constraint,
        value_objective_scale=cfg.value_objective_scale,
        value_constraint_scale=cfg.value_constraint_scale,
        normalizer=normalizer,
        dynamics_model=dynamics_model,
        ellipse_margin=cfg.ellipse_margin,
    ).to(cfg.device)

    flow_policy = FlowPolicy(
        flow_model=flow_model,
        value_model=value_model,
        normalizer=normalizer,
        action_dim=cfg.action_dim,
        state_dim=cfg.state_dim,
        horizon=cfg.horizon,
        cfg=cfg,
        dynamics_model=dynamics_model,
    )

    # l4casadi
    if (
        cfg.guidance_method == "projection"
        or cfg.guidance_method == "projection_relaxed"
        or cfg.guidance_method == "hardflow"
    ):
        wrapped_flow_model = WrappedFlowUnet(
            horizon=cfg.horizon,
            transition_dim=cfg.state_dim + cfg.action_dim,
            cond_dim=cfg.state_dim,
            dim=32,
            dim_mults=(1, 4, 8),
            attention=False,
        ).to(cfg.device)
        wrapped_flow_model.load_state_dict(
            torch.load(
                os.path.join(
                    cfg.log_folder,
                    cfg.env,
                    "flow",
                    cfg.flow_exp_name,
                    f"model_ema_{cfg.flow_cp}.pth",
                ),
                map_location='cpu'
            )
        )
        wrapped_flow_model.add_info(
            horizon=cfg.horizon, transition_dim=cfg.state_dim + cfg.action_dim
        )

        l4c_flow_fn = l4c.L4CasADi(wrapped_flow_model, device="cpu", name="flow_model")

        constrained_flow_fn = create_constrained_casadi_functions(
            l4c_flow_fn,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
        )

        flow_policy.constrained_flow_fn = constrained_flow_fn

        if cfg.guidance_method == "projection":
            flow_policy.projection_formulate(
                print_level=cfg.solver_print_level,
                constraint=cfg.constraint,
            )
        elif cfg.guidance_method == "projection_relaxed":
            flow_policy.projection_relaxed_formulate(
                print_level=cfg.solver_print_level,
                constraint=cfg.constraint,
            )
        elif cfg.guidance_method == "hardflow":
            flow_policy.hardflow_formulate(
                print_level=cfg.solver_print_level,
                constraint=cfg.constraint,
                objective=cfg.cost,
            )

    if cfg.guidance_method == "hardflow_new":
        flow_policy.hardflow_formulate(
            print_level=cfg.solver_print_level,
            constraint=cfg.constraint,
            objective=cfg.cost,
        )

    # run env
    trajectory_data = []
    real_trajectories_ls = []
    for run_id in range(cfg.random_repeat):
        score, total_violations, real_trajectory, avg_computation_time = run_env(
            dataset.env, flow_policy, cfg, run_id=run_id
        )

        steps = len(real_trajectory) - 1
        safety = total_violations == 0

        trajectory_data.append(
            {
                "run_id": run_id,
                "steps": steps,
                "total_violations": float(total_violations),
                "score": float(score),
                "safety": safety,
                "average_computation_time": (
                    float(avg_computation_time)
                    if avg_computation_time is not None
                    else None
                ),
            }
        )

        real_trajectories_ls.append(real_trajectory)

    if real_trajectories_ls:
        ...

    csv_path = os.path.join(
        cfg.log_folder, cfg.env, "eval", cfg.exp_name, "trajectories.csv"
    )
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "run_id",
            "steps",
            "total_violations",
            "score",
            "safety",
            "average_computation_time",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in trajectory_data:
            writer.writerow(data)


if __name__ == "__main__":
    cfg = tyro.cli(FlowMatchingEvaluationConfig)
    #set_cuda_visible_device(cfg)
    deterministic(cfg.seed)  # seed everything

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, "eval", cfg.exp_name)
    save_config(cfg, log_subfolder)

    evaluate(cfg)
