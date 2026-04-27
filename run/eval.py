import os
import json
import tyro
import torch
import numpy as np
from torch.utils.data import DataLoader
import math
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp
from tqdm import tqdm

from hardflow.config.flow_matching import FlowMatchingEvaluationConfig
from hardflow.datasets.burgers import BurgersDataset, BurgersDataNormalizer
from hardflow.models_flow.flow_policy import FlowPolicy
from hardflow.models_flow.unet import TemporalUnet, WrappedFlowUnet
from hardflow.utils.rendering import (
    visualize_minimal_pde,
    visualize_state_slices,
)
from hardflow.utils.safety import build_safety_upper_profile, build_safety_lower_profile
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
        objective: str = "",
        constraint: str = "",
        value_objective_scale: float = 1.0,
        value_constraint_scale: float = 1.0,
        normalizer: BurgersDataNormalizer = None,
        safety_threshold: float = 0.8,
        safety_threshold_margin: float = 0.0,
        constraint_profile: str = "quadratic",
    ):

        super().__init__()

        self.objective = objective
        self.constraint = constraint
        self.value_objective_scale = value_objective_scale
        self.value_constraint_scale = value_constraint_scale
        self.normalizer = normalizer or BurgersDataNormalizer()
        self.constraint_profile = constraint_profile
        self.spatial_dim = 128
        temporal_steps = 11

        full_upper_profile = build_safety_upper_profile(
            spatial_dim=self.spatial_dim,
            safety_threshold=safety_threshold,
            profile_type=self.constraint_profile,
            safety_threshold_margin=safety_threshold_margin,
            temporal_steps=temporal_steps,
            device=torch.device("cpu"),
        )
        full_lower_profile = build_safety_lower_profile(
            spatial_dim=self.spatial_dim,
            safety_threshold=safety_threshold,
            profile_type=self.constraint_profile,
            safety_threshold_margin=safety_threshold_margin,
            temporal_steps=temporal_steps,
            device=torch.device("cpu"),
        )

        inner_upper = full_upper_profile[1:-1]
        inner_lower = full_lower_profile[1:-1]

        upper_normalized = self.normalizer.normalize_u(inner_upper)
        lower_normalized = self.normalizer.normalize_u(inner_lower)

        self.register_buffer(
            "safety_threshold_normalized_ub",
            upper_normalized.unsqueeze(0),
        )
        self.register_buffer(
            "safety_threshold_normalized_lb",
            lower_normalized.unsqueeze(0),
        )

    def forward(
        self,
        x,
    ):
        """
        x: (batch_size, 2, 16, 128), normalized
        value: (batch_size, 1)
        """

        if self.objective == "":
            objective_value = torch.zeros(
                x.shape[0], 1, device=x.device, requires_grad=True
            )
        elif self.objective == "control_energy":
            f = x[:, 1, 0:10, :]  # (batch_size, 10, 128)
            objective_value = -torch.sum(f**2, dim=(-1, -2), keepdim=True)
        else:
            raise NotImplementedError(f"Objective {self.objective} is not implemented.")

        if self.constraint == "":
            constraint_penalty = torch.zeros(
                x.shape[0], 1, device=x.device, requires_grad=True
            )
        elif self.constraint == "safety_score":
            u = x[:, 0, 1:10, :]  # (batch_size, 9, 128)

            violation_upper = torch.clamp(
                u - self.safety_threshold_normalized_ub, min=0.0
            )
            violation_lower = torch.clamp(
                self.safety_threshold_normalized_lb - u, min=0.0
            )

            constraint_penalty = (violation_upper.pow(2) + violation_lower.pow(2)).sum(
                dim=(-1, -2), keepdim=True
            )

        else:
            raise NotImplementedError(
                f"Constraint {self.constraint} is not implemented."
            )

        value = (
            objective_value * self.value_objective_scale
            - constraint_penalty * self.value_constraint_scale
        )

        return value


def create_constrained_casadi_functions(
    casadi_flow_fn,
    temporal_dim=11,
    spatial_dim=128,
):

    print(f"create_constrained_casadi_function")

    N_full = temporal_dim * spatial_dim + (temporal_dim - 1) * spatial_dim
    dof = N_full - 2 * spatial_dim
    print(f"Calculated dimensions: N_full={N_full}, dof={dof}")

    dof_sym = cs.MX.sym("dof_sym", 1, dof)
    u0_sym = cs.MX.sym("s0_sym", 1, spatial_dim)
    uT_sym = cs.MX.sym("sH_sym", 1, spatial_dim)

    full_traj_sym = cs.horzcat(
        u0_sym,
        dof_sym[:, : (temporal_dim - 2) * spatial_dim],
        uT_sym,
        dof_sym[:, (temporal_dim - 2) * spatial_dim :],
    )
    print(f"full_traj_sym shape: {full_traj_sym.shape}")

    time_sym = cs.MX.sym("t")
    flow_input_sym = cs.horzcat(full_traj_sym, time_sym)

    full_flow_output_sym = casadi_flow_fn(flow_input_sym)

    dof_flow_output_sym = cs.horzcat(
        full_flow_output_sym[:, spatial_dim : (temporal_dim - 1) * spatial_dim],
        full_flow_output_sym[:, (temporal_dim) * spatial_dim :],
    )

    constrained_flow_fn = cs.Function(
        "constrained_flow_fn",
        [dof_sym, time_sym, u0_sym, uT_sym],
        [dof_flow_output_sym],
    )

    dummy_dof = cs.DM.zeros(1, dof)
    dummy_time = cs.DM(0.0)
    dummy_u0 = cs.DM.zeros(1, spatial_dim)
    dummy_uT = cs.DM.zeros(1, spatial_dim)

    _ = constrained_flow_fn(dummy_dof, dummy_time, dummy_u0, dummy_uT)

    return constrained_flow_fn


def Diff_mat_1D(Nx, device="cpu"):
    """
    Create difference matrices for first and second derivatives.
    """

    D_1d = sp.diags([-1, 1], [-1, 1], shape=(Nx, Nx))
    D_1d = sp.lil_matrix(D_1d)
    D_1d[0, [0, 1, 2]] = [-3, 4, -1]
    D_1d[Nx - 1, [Nx - 3, Nx - 2, Nx - 1]] = [1, -4, 3]

    D2_1d = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx))
    D2_1d = sp.lil_matrix(D2_1d)
    D2_1d[0, [0, 1, 2, 3]] = [2, -5, 4, -1]
    D2_1d[Nx - 1, [Nx - 4, Nx - 3, Nx - 2, Nx - 1]] = [-1, 4, -5, 2]

    return D_1d, D2_1d


def burgers_numeric_solve(
    u0,
    f,
    visc=0.01,
    T=1.0,
    dt=1e-4,
):
    """
    Solve Burgers equation using finite difference method.

    Args:
        u0: Initial condition [batch_size, 128]
        f: Control [batch_size, 10, 128] (10 time intervals)
        visc: Viscosity (default 0.01)
        T: Total simulation time (default 1.0)
        dt: Time step for numerical integration (default 1e-4)

    Returns:
        trajectory: Solution [batch_size, 11, 128]
    """

    s = u0.size(-1)
    Nt = f.size(1)

    batch_size = u0.size(0)
    assert batch_size == f.size(0), "Batch sizes of u0 and f must match"

    xmin = 0.0
    xmax = 1.0
    delta_x = (xmax - xmin) / (s + 1)

    steps = math.ceil(T / dt)

    u = u0.reshape(batch_size, s)
    u = F.pad(u, (1, 1))
    f = f.reshape(batch_size, Nt, s)
    f = F.pad(f, (1, 1))

    record_time = math.floor(steps / Nt)

    D_1d, D2_1d = Diff_mat_1D(s + 2, device=u0.device)
    D_1d.rows[0] = D_1d.rows[0][:2]
    D_1d.rows[-1] = D_1d.rows[-1][-2:]
    D_1d.data[0] = D_1d.data[0][:2]
    D_1d.data[-1] = D_1d.data[-1][-2:]

    D2_1d.rows[0] = D2_1d.rows[0][:3]
    D2_1d.rows[-1] = D2_1d.rows[-1][-3:]
    D2_1d.data[0] = D2_1d.data[0][:3]
    D2_1d.data[-1] = D2_1d.data[-1][-3:]

    t_sys_ind = list(D_1d.rows)
    t_sys = torch.tensor(
        np.stack(D_1d.data) / (2 * delta_x), dtype=u0.dtype, device=u0.device
    )
    d_sys_ind = list(D2_1d.rows)
    d_sys = torch.tensor(
        visc * np.stack(D2_1d.data) / delta_x**2, dtype=u0.dtype, device=u0.device
    )

    sol = torch.zeros(batch_size, s, Nt, dtype=u0.dtype, device=u0.device)

    c = 0
    t = 0.0
    f_idx = -1
    for j in range(steps):
        u = u[..., 1:-1]
        u = F.pad(u, (1, 1))

        u_s = u**2
        transport = torch.einsum("nsi,si->ns", u_s[..., t_sys_ind], t_sys)
        diffusion = torch.einsum("nsi,si->ns", u[..., d_sys_ind], d_sys)
        if j % record_time == 0:
            f_idx += 1
        u = u + dt * (-(1 / 2) * transport + diffusion + f[:, f_idx, :])

        t += dt

        if (j + 1) % record_time == 0:
            sol[..., c] = u[..., 1:-1]
            c += 1

    sol = sol.permute(0, 2, 1)  # (batch_size, Nt, s)
    trajectory = torch.cat((u0.reshape(batch_size, 1, s), sol), dim=1)
    return trajectory


def calculate_safety_metrics(u, threshold_profile):
    """
    Calculate safety metrics

    Args:
        u: State [B, 11, 128], unnormalized
        threshold_profile: Safety threshold per spatial point

    Returns:
        Dictionary with safety metrics
    """

    u = u[:, 1:10, :]

    metrics = {}

    if isinstance(threshold_profile, (float, int)):
        threshold_tensor = torch.full(
            (1, u.shape[1], u.shape[-1]),
            float(threshold_profile),
            device=u.device,
            dtype=u.dtype,
        )
    else:
        threshold_tensor = torch.as_tensor(
            threshold_profile,
            device=u.device,
            dtype=u.dtype,
        )
        if threshold_tensor.dim() == 1:
            threshold_tensor = threshold_tensor.view(1, 1, -1).expand(1, u.shape[1], -1)
        elif threshold_tensor.dim() == 2:
            if threshold_tensor.shape[0] == u.shape[1]:
                threshold_tensor = threshold_tensor.unsqueeze(0)
            elif threshold_tensor.shape[0] == u.shape[1] + 2:
                threshold_tensor = threshold_tensor[1:-1].unsqueeze(0)
            else:
                raise ValueError(
                    "Threshold profile temporal dimension mismatch with state tensor."
                )
        elif threshold_tensor.dim() == 3:
            pass
        else:
            raise ValueError("Unsupported threshold profile shape.")

    exceed_mask = u.abs() > threshold_tensor

    # Point-wise exceed ratio (R_p)
    point_exceed = exceed_mask.any(dim=-2).float().mean()
    metrics["point_exceed_ratio"] = point_exceed.item()

    # Time-wise exceed ratio (R_t)
    time_exceed = exceed_mask.any(dim=-1).float().mean()
    metrics["time_exceed_ratio"] = time_exceed.item()

    # Sample-wise exceed ratio (R_s)
    sample_exceed = exceed_mask.any(dim=(-1, -2)).float()
    metrics["sample_exceed_ratio"] = sample_exceed.mean().item()

    # Safety score (max absolute value)
    safety_scores = u.abs().amax(dim=(-1, -2))
    metrics["safety_score_mean"] = safety_scores.mean().item()
    metrics["safety_score_std"] = safety_scores.std().item()
    metrics["safety_score_max"] = safety_scores.max().item()

    return metrics, sample_exceed


def evaluate_burgers(cfg: FlowMatchingEvaluationConfig):

    if "cuda" in cfg.device:
        assert torch.cuda.is_available(), f"CUDA error"
    device = torch.device(cfg.device)
    deterministic(cfg.seed)

    eval_dir = os.path.join(cfg.log_folder, "burgers", "eval", cfg.eval_exp_name)
    os.makedirs(eval_dir, exist_ok=True)

    normalizer = BurgersDataNormalizer()
    test_dataset = BurgersDataset(
        split="test", root_path=cfg.data_path, normalizer=normalizer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.n_test_samples,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    spatial_dim = getattr(test_dataset, "nx", 128)
    safety_upper_profile_eval = build_safety_upper_profile(
        spatial_dim=spatial_dim,
        safety_threshold=cfg.safety_threshold,
        profile_type=cfg.safety_constraint_profile,
        safety_threshold_margin=0.0,
        temporal_steps=11,
    )
    safety_lower_profile_eval = -safety_upper_profile_eval

    flow_model = TemporalUnet(
        output_channels=2,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        n_groups=1,
    ).to(device)

    checkpoint_path = os.path.join(
        cfg.log_folder,
        "burgers",
        "flow",
        cfg.flow_exp_name,
        f"model_ema_{cfg.flow_cp}.pth",
    )
    print(f"Loading checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(
            cfg.log_folder,
            "burgers",
            "flow",
            cfg.flow_exp_name,
            f"model_{cfg.flow_cp}.pth",
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    flow_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    value_model = ProxyValueModel(
        objective=cfg.value_objective,
        constraint=cfg.constraint,
        value_objective_scale=cfg.value_objective_scale,
        value_constraint_scale=cfg.value_constraint_scale,
        normalizer=normalizer,
        safety_threshold=cfg.safety_threshold,
        safety_threshold_margin=cfg.safety_threshold_margin,
        constraint_profile=cfg.safety_constraint_profile,
    ).to(device)

    flow_policy = FlowPolicy(
        flow_model=flow_model,
        value_model=value_model,
        cfg=cfg,
        normalizer=normalizer,
    )

    # l4casadi
    if cfg.guidance_method in [
        "projection",
        "hardflow",
        "projection_relaxed",
        "oc_flow",
    ]:
        wrapped_flow_model = WrappedFlowUnet(
            output_channels=2,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            n_groups=1,
        ).to(cfg.device)
        wrapped_flow_model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        wrapped_flow_model.add_info(
            temporal_dim=11, spatial_dim=128, padded_temporal_dim=16
        )

        l4c_flow_fn = l4c.L4CasADi(
            wrapped_flow_model,
            device="cuda",
            name="flow_model",
            generate_jac=False,
            generate_adj1=False,
            generate_jac_adj1=False,
            generate_jac_jac=False,
        )

        constrained_flow_fn = create_constrained_casadi_functions(
            l4c_flow_fn,
            temporal_dim=11,
            spatial_dim=128,
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
        elif cfg.guidance_method == "oc_flow":
            flow_policy.oc_flow_formulate(
                print_level=cfg.solver_print_level,
                constraint=cfg.constraint,
            )

    if cfg.guidance_method == "hardflow_new":
        flow_policy.hardflow_formulate(
            print_level=cfg.solver_print_level,
            constraint=cfg.constraint,
            objective=cfg.cost,
        )

    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    safety_threshold = cfg.safety_threshold
    center_idx = safety_upper_profile_eval.shape[0] // 2
    safety_center_threshold = safety_upper_profile_eval[center_idx, 0].item()
    print(
        f"Safety profile ({cfg.safety_constraint_profile}): "
        f"boundary={safety_threshold:.2f}, center={safety_center_threshold:.2f}"
    )
    if cfg.safety_threshold_margin != 0.0:
        print(f"Constraint margin: {cfg.safety_threshold_margin:.3f}")

    batch = next(iter(test_loader))
    batch = batch[: cfg.n_test_samples].to(device)  # [B, 2, 16, 128]

    # ground truth
    u_target = normalizer.denormalize_u(batch[:, 0, :11, :])  # [B, 11, 128]
    f_target = normalizer.denormalize_f(batch[:, 1, :10, :])  # [B, 10, 128]

    all_u_predicted = []
    all_u_controlled = []
    all_control_energy = []
    computation_times = []

    # generate samples
    for i in tqdm(range(batch.shape[0]), desc="Generating samples"):

        # boundary conditions
        u0_raw = u_target[i : i + 1, 0, :]  # [1, 128], unnormalized
        uT_raw = u_target[i : i + 1, -1, :]  # [1, 128], unnormalized
        upper_eval = safety_upper_profile_eval.to(u0_raw.device)
        lower_eval = safety_lower_profile_eval.to(u0_raw.device)

        assert torch.all(u0_raw <= upper_eval[0] + 1e-6)
        assert torch.all(u0_raw >= lower_eval[0] - 1e-6)
        assert torch.all(uT_raw <= upper_eval[-1] + 1e-6)
        assert torch.all(uT_raw >= lower_eval[-1] - 1e-6)

        u0_normalized = normalizer.normalize_u(u0_raw)  # [1, 128], normalized
        uT_normalized = normalizer.normalize_u(uT_raw)  # [1, 128], normalized

        # single sample
        conditions = (u0_normalized, uT_normalized)
        control, trajectories, _, _, info = flow_policy(
            conditions=conditions,
            batch_size=1,
        )
        if isinstance(info, dict) and "computation_time" in info:
            computation_times.append(float(info["computation_time"]))

        u_pred = torch.tensor(
            trajectories.states[:1], dtype=u0_raw.dtype, device=device
        )  # [1, 10, 128]
        f_pred = torch.tensor(
            trajectories.controls[:1], dtype=u0_raw.dtype, device=device
        )  # [1, 10, 128]

        u_controlled = burgers_numeric_solve(u0_raw, f_pred)

        visualize_minimal_pde(
            u_controlled[0],  # [11, 128]
            f_pred[0],  # [10, 128]
            eval_dir,
            i,
        )

        visualize_state_slices(
            u_controlled[0],  # [11, 128]
            safety_upper_profile_eval,  # [11, 128]
            safety_lower_profile_eval,  # [11, 128]
            eval_dir,
            i,
            times=(0.3, 0.5, 0.8),
        )

        all_u_predicted.append(u_pred)
        all_u_controlled.append(u_controlled)

        control_energy_single = (f_pred[0] ** 2).mean()  # per grid point
        all_control_energy.append(control_energy_single.cpu().numpy())

    all_u_predicted = torch.cat(all_u_predicted, dim=0)  # [B, 11, 128]
    all_u_controlled = torch.cat(all_u_controlled, dim=0)  # [B, 11, 128]
    all_control_energy = np.array(all_control_energy)  # [B]

    safety_metrics_controlled, _ = calculate_safety_metrics(
        all_u_controlled, safety_upper_profile_eval
    )

    safety_metrics_predicted, _ = calculate_safety_metrics(
        all_u_predicted, safety_upper_profile_eval
    )

    safety_metrics_target, _ = calculate_safety_metrics(
        u_target, safety_upper_profile_eval
    )

    results = {
        "control_energy_mean (J_energy)": float(all_control_energy.mean()),
        "control_energy_std": float(all_control_energy.std()),
        "point_exceed_ratio (R_p), target": safety_metrics_target["point_exceed_ratio"],
        "time_exceed_ratio (R_t), target": safety_metrics_target["time_exceed_ratio"],
        "sample_exceed_ratio (R_s), target": safety_metrics_target[
            "sample_exceed_ratio"
        ],
        "point_exceed_ratio (R_p), controlled": safety_metrics_controlled[
            "point_exceed_ratio"
        ],
        "time_exceed_ratio (R_t), controlled": safety_metrics_controlled[
            "time_exceed_ratio"
        ],
        "sample_exceed_ratio (R_s), controlled": safety_metrics_controlled[
            "sample_exceed_ratio"
        ],
        "safety_score_mean, controlled": safety_metrics_controlled["safety_score_mean"],
        "safety_score_std, controlled": safety_metrics_controlled["safety_score_std"],
        "safety_score_max, controlled": safety_metrics_controlled["safety_score_max"],
        "point_exceed_ratio (R_p), predicted": safety_metrics_predicted[
            "point_exceed_ratio"
        ],
        "time_exceed_ratio (R_t), predicted": safety_metrics_predicted[
            "time_exceed_ratio"
        ],
        "sample_exceed_ratio (R_s), predicted": safety_metrics_predicted[
            "sample_exceed_ratio"
        ],
        "safety_score_mean, predicted": safety_metrics_predicted["safety_score_mean"],
        "safety_score_std, predicted": safety_metrics_predicted["safety_score_std"],
        "safety_score_max, predicted": safety_metrics_predicted["safety_score_max"],
        "num_samples": cfg.n_test_samples,
        "safety_threshold_boundary": float(safety_threshold),
        "safety_threshold_center": float(safety_center_threshold),
        "safety_constraint_profile": cfg.safety_constraint_profile,
        "safety_threshold_margin": float(cfg.safety_threshold_margin),
    }
    if computation_times:
        comp_times = np.array(computation_times, dtype=np.float64)
        results["computation_time_mean"] = float(comp_times.mean())
        results["computation_time_std"] = float(comp_times.std())

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)

    print(
        f"Control Energy (J_energy): {results['control_energy_mean (J_energy)']:.6f} ± {results['control_energy_std']:.6f}"
    )

    print(
        f"Point Exceed Ratio (R_p), Target: {results['point_exceed_ratio (R_p), target'] * 100:.2f}%"
    )
    print(
        f"Time Exceed Ratio (R_t), Target: {results['time_exceed_ratio (R_t), target'] * 100:.2f}%"
    )
    print(
        f"Sample Exceed Ratio (R_s), Target: {results['sample_exceed_ratio (R_s), target'] * 100:.2f}%"
    )

    print(
        f"Point Exceed Ratio (R_p), Controlled: {results['point_exceed_ratio (R_p), controlled']*100:.2f}%"
    )
    print(
        f"Time Exceed Ratio (R_t), Controlled: {results['time_exceed_ratio (R_t), controlled']*100:.2f}%"
    )
    print(
        f"Sample Exceed Ratio (R_s), Controlled: {results['sample_exceed_ratio (R_s), controlled']*100:.2f}%"
    )

    print(
        f"Point Exceed Ratio (R_p), Predicted: {results['point_exceed_ratio (R_p), predicted'] * 100:.2f}%"
    )
    print(
        f"Time Exceed Ratio (R_t), Predicted: {results['time_exceed_ratio (R_t), predicted'] * 100:.2f}%"
    )
    print(
        f"Sample Exceed Ratio (R_s), Predicted: {results['sample_exceed_ratio (R_s), predicted'] * 100:.2f}%"
    )

    results_path = os.path.join(eval_dir, f"results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    cfg = tyro.cli(FlowMatchingEvaluationConfig)
    set_cuda_visible_device(cfg)
    deterministic(cfg.seed)  # seed everything

    log_subfolder = os.path.join(cfg.log_folder, "burgers", "eval", cfg.eval_exp_name)
    save_config(cfg, log_subfolder)

    evaluate_burgers(cfg)
