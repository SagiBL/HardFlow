from collections import namedtuple
import time
import torch
from torch import nn
from hardflow.config.flow_matching import FlowMatchingEvaluationConfig
from hardflow import utils
from hardflow.models_flow.flow_matcher import (
    apply_conditioning,
    apply_conditioning_from_conditioned_x,
)
from hardflow.utils.arrays import to_torch
from hardflow.utils.safety import build_safety_upper_profile, build_safety_lower_profile

import numpy as np

try:
    import casadi as cs
except ImportError:
    cs = None

Trajectories = namedtuple("Trajectories", "controls states values")


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class ConditionedODESolver(nn.Module):
    def __init__(
        self,
        model,
        value=None,
        ode_method="euler",
        guidance_lr=1.0,
        guide_stpes=0,
    ):
        super().__init__()
        self.model = model
        self.value = value
        assert ode_method in ["euler"], "Only Euler is supported for now"

        self.guidance_lr = guidance_lr
        self.guidance_steps = guide_stpes

    def forward(self, x, t_span, return_chain=False, *args, **kwargs):
        # x: (batch_size, 2, 16, 128)
        # t_span: at least 2 elements, should be evenly spaced
        assert len(t_span) > 1, "t_span must have at least 2 elements"

        x0 = x.clone()
        dt = t_span[1] - t_span[0]
        x_chain = [x0]
        for t in t_span:

            if self.value is not None and self.guidance_steps > 0:
                for _ in range(self.guidance_steps):
                    inputs = x.detach().clone().requires_grad_(True)
                    dx_dt = self.model(inputs, t)

                    dx_dt[:, 0, 0, :] = 0.0
                    dx_dt[:, 0, 10, :] = 0.0
                    dx_dt[:, 0, 11:, :] = 0.0
                    dx_dt[:, 1, 10:, :] = 0.0

                    predicted_x1 = inputs + dx_dt * (1.0 - t)
                    loss = -self.value(predicted_x1)
                    if loss.ndim > 0:
                        loss = loss.mean()
                    (g,) = torch.autograd.grad(loss, inputs, create_graph=False)

                    g[:, 0, 0, :] = 0.0
                    g[:, 0, 10, :] = 0.0
                    g[:, 0, 11:, :] = 0.0
                    g[:, 1, 10:, :] = 0.0

                    x = (inputs - self.guidance_lr * g).detach()

            dx_dt = self.model(x, t)
            # zero out velocity at conditioned points
            dx_dt[:, 0, 0, :] = 0.0
            dx_dt[:, 0, 10, :] = 0.0

            # zero out velocity at padded points
            dx_dt[:, 0, 11:, :] = 0.0
            dx_dt[:, 1, 10:, :] = 0.0

            x = x + dx_dt * dt
            x = x.detach()
            x_chain.append(x)

        if return_chain:
            return x, torch.stack(x_chain, dim=1).detach()
        else:
            return x


class FlowPolicy(nn.Module):
    def __init__(
        self,
        flow_model,
        value_model,
        normalizer,
        cfg: FlowMatchingEvaluationConfig,
    ):
        super().__init__()
        self.flow_model = flow_model
        self.value_model = value_model
        self.normalizer = normalizer
        self.cfg = cfg
        self._safety_state_lower = None
        self._safety_state_upper = None

    def __call__(
        self,
        conditions,
        batch_size=1,
    ):

        if self.cfg.guidance_method in ["original"]:
            pass
        elif self.cfg.guidance_method in ["gradient_guidance"]:
            return self.gradient_guidance_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ["projection"]:
            return self.projection_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ["projection_relaxed"]:
            return self.projection_relaxed_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ["hardflow"]:
            return self.hardflow_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ["hardflow_new"]:
            return self.hardflow_new_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ["oc_flow"]:
            return self.oc_flow_forward(conditions, batch_size)
        else:
            raise ValueError(f"Unsupported guidance method: {self.cfg.guidance_method}")

        assert batch_size == 1, "batch_size must be 1 for no guidance"

        (u0, uT) = conditions

        t_start = time.time()

        solver = ConditionedODESolver(
            self.flow_model,
            value=None,
            ode_method=self.cfg.ode_solver,
        )

        x = torch.randn(
            batch_size,
            2,
            16,
            128,
            device=self.cfg.device,
        )  # (batch_size, 2, 16, 128)
        x = apply_conditioning(x, u0, uT)

        x, x_chain = solver(
            x,
            t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps + 1, device=x.device
            )[:-1],
            return_chain=True,
        )

        normed_controls = x[:, 1, :10, :]
        controls = self.normalizer.denormalize_f(to_np(normed_controls))

        normed_states = x[:, 0, :11, :]
        states = self.normalizer.denormalize_u(to_np(normed_states))

        values = self.value_model(x)
        values = to_np(values)

        trajectories = Trajectories(controls, states, values)

        control = controls[0]

        x1_estimation = self.x1_estimate(x_chain)
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {"computation_time": computation_time}
        return control, trajectories, x_chain, x1_estimation, info

    def x1_estimate(self, x_chain):

        batch_size, n_steps, _, _, _ = x_chain.shape

        x1_estimation = torch.zeros_like(x_chain)

        with torch.no_grad():
            for k in range(n_steps):
                t = to_torch(k / (n_steps - 1), device=x_chain.device)
                current_x = x_chain[:, k, :, :, :]

                current_v = self.flow_model(current_x, t)

                current_v[:, 0, 0, :] = 0
                current_v[:, 0, 10, :] = 0

                predicted_x = current_x + (1.0 - t) * current_v
                x1_estimation[:, k, :, :] = predicted_x

        return x1_estimation

    def unnormalize_chain(self, x_chain):

        batch_size, n_steps, _, _, _ = x_chain.shape
        is_torch = torch.is_tensor(x_chain)
        if is_torch:
            x_chain_device = x_chain.device
            x_chain_dtype = x_chain.dtype

        for k in range(n_steps):
            current_x = x_chain[:, k, :, :, :]

            normed_states = current_x[:, 0, :11, :]
            normed_controls = current_x[:, 1, :10, :]
            states = self.normalizer.denormalize_u(to_np(normed_states))
            controls = self.normalizer.denormalize_f(to_np(normed_controls))

            if is_torch:
                x_chain[:, k, 0, :11, :] = to_torch(
                    states, dtype=x_chain_dtype, device=x_chain_device
                )
                x_chain[:, k, 1, :10, :] = to_torch(
                    controls, dtype=x_chain_dtype, device=x_chain_device
                )
            else:
                x_chain[:, k, 0, :11, :] = states
                x_chain[:, k, 1, :10, :] = controls

        return x_chain

    def _ensure_safety_bounds(self):
        if self._safety_state_lower is None or self._safety_state_upper is None:
            upper_profile = build_safety_upper_profile(
                spatial_dim=128,
                safety_threshold=self.cfg.safety_threshold,
                profile_type=self.cfg.safety_constraint_profile,
                safety_threshold_margin=self.cfg.safety_threshold_margin,
                temporal_steps=11,
            )
            lower_profile = build_safety_lower_profile(
                spatial_dim=128,
                safety_threshold=self.cfg.safety_threshold,
                profile_type=self.cfg.safety_constraint_profile,
                safety_threshold_margin=self.cfg.safety_threshold_margin,
                temporal_steps=11,
            )
            upper_inner = torch.tensor(upper_profile[1:-1], dtype=torch.float32)
            lower_inner = torch.tensor(lower_profile[1:-1], dtype=torch.float32)
            upper_norm = self.normalizer.normalize_u(upper_inner)
            lower_norm = self.normalizer.normalize_u(lower_inner)
            self._safety_state_upper = upper_norm.detach().cpu().numpy().reshape(-1)
            self._safety_state_lower = lower_norm.detach().cpu().numpy().reshape(-1)

    def _dof_to_full_tensor(self, dof_vec, u0_vec, uT_vec, dtype=None):
        dtype = dtype or dof_vec.dtype
        full = np.zeros((1, 2, 16, 128), dtype=dtype)
        state = dof_vec[: 9 * 128].reshape(9, 128)
        control = dof_vec[9 * 128 :].reshape(10, 128)
        full[0, 0, 0, :] = u0_vec
        full[0, 0, 1:10, :] = state
        full[0, 0, 10, :] = uT_vec
        full[0, 1, :10, :] = control
        return full

    def _full_tensor_to_dof(self, full_tensor):
        state = full_tensor[0, 0, 1:10, :].reshape(-1)
        control = full_tensor[0, 1, :10, :].reshape(-1)
        return np.concatenate([state, control], axis=-1)

    def _dof_to_full_tensor_torch(self, dof_tensor, u0_tensor, uT_tensor):
        if dof_tensor.dim() == 1:
            dof_tensor = dof_tensor.unsqueeze(0)
        if u0_tensor.dim() == 1:
            u0_tensor = u0_tensor.unsqueeze(0)
        if uT_tensor.dim() == 1:
            uT_tensor = uT_tensor.unsqueeze(0)

        batch = dof_tensor.shape[0]
        full = torch.zeros(
            batch,
            2,
            16,
            128,
            device=dof_tensor.device,
            dtype=dof_tensor.dtype,
        )
        state = dof_tensor[:, : 9 * 128].reshape(batch, 9, 128)
        control = dof_tensor[:, 9 * 128 :].reshape(batch, 10, 128)
        full[:, 0, 0, :] = u0_tensor
        full[:, 0, 1:10, :] = state
        full[:, 0, 10, :] = uT_tensor
        full[:, 1, :10, :] = control
        return full

    def _constrained_flow_torch(self, dof_tensor, t_scalar, u0_tensor, uT_tensor):
        full = self._dof_to_full_tensor_torch(dof_tensor, u0_tensor, uT_tensor)
        t_tensor = to_torch(t_scalar, device=full.device)
        v = self.flow_model(full, t_tensor)
        v[:, 0, 0, :] = 0.0
        v[:, 0, 10, :] = 0.0
        v[:, 0, 11:, :] = 0.0
        v[:, 1, 10:, :] = 0.0
        state_v = v[:, 0, 1:10, :].reshape(full.shape[0], -1)
        control_v = v[:, 1, :10, :].reshape(full.shape[0], -1)
        return torch.cat([state_v, control_v], dim=-1)

    def _apply_safety_constraints(self, opti, X_var, X_index_selector=None):

        assert (
            self.cfg.constraint == "safety_score"
        ), f"Unsupported constraint: {self.cfg.constraint}"

        if X_index_selector is None:
            dof = X_var[-1, :]
        elif X_index_selector == "projection":
            dof = X_var
        else:
            raise ValueError

        spatial_dim = 128
        temporal_steps = 11

        full_upper_profile = build_safety_upper_profile(
            spatial_dim=spatial_dim,
            safety_threshold=self.cfg.safety_threshold,
            profile_type=self.cfg.safety_constraint_profile,
            safety_threshold_margin=self.cfg.safety_threshold_margin,
            temporal_steps=temporal_steps,
        )
        full_lower_profile = build_safety_lower_profile(
            spatial_dim=spatial_dim,
            safety_threshold=self.cfg.safety_threshold,
            profile_type=self.cfg.safety_constraint_profile,
            safety_threshold_margin=self.cfg.safety_threshold_margin,
            temporal_steps=temporal_steps,
        )

        inner_upper = full_upper_profile[1:-1]
        inner_lower = full_lower_profile[1:-1]

        safety_threshold_normalized_ub = (
            self.normalizer.normalize_u(inner_upper).cpu().numpy()
        )
        safety_threshold_normalized_lb = (
            self.normalizer.normalize_u(inner_lower).cpu().numpy()
        )

        state = dof[: 9 * spatial_dim]

        upper_bounds = safety_threshold_normalized_ub.reshape(1, -1)
        lower_bounds = safety_threshold_normalized_lb.reshape(1, -1)

        opti.subject_to(state.T >= lower_bounds)

        opti.subject_to(state.T <= upper_bounds)

    def _apply_pde_constraints(
        self,
        opti,
        X_var,
        u0_normalized,
        uT_normalized,
        T: float = 1.0,
        visc_bounds: tuple = (0.0, 0.02),
        X_index_selector=None,
    ):

        N, Nt = 128, 10
        dx = 1.0 / (N + 1)
        dt = T / Nt
        Nx = N + 2  # padded

        visc_var = opti.variable()
        opti.subject_to(visc_var >= visc_bounds[0])
        opti.subject_to(visc_var <= visc_bounds[1])

        if X_index_selector is None:
            dof = cs.vec(X_var[-1, :])
        elif X_index_selector == "projection":
            dof = cs.vec(X_var)
        else:
            raise ValueError

        state_flat = dof[: (Nt - 1) * N]
        control_flat = dof[(Nt - 1) * N :]

        u0 = cs.vec(self.normalizer.denormalize_u(u0_normalized))
        uT = cs.vec(self.normalizer.denormalize_u(uT_normalized))
        S = self.normalizer.denormalize_u(state_flat)
        F = self.normalizer.denormalize_f(control_flat)

        U_rows = [u0] + [S[k * N : (k + 1) * N] for k in range(Nt - 1)] + [uT]
        G_rows = [F[k * N : (k + 1) * N] for k in range(Nt)]

        inv2dx = 1.0 / (2.0 * dx)
        invdx2 = 1.0 / (dx * dx)

        D1 = cs.DM.zeros(Nx, Nx)
        for i in range(1, Nx - 1):
            D1[i, i - 1] = -1.0
            D1[i, i + 1] = 1.0
        D1[0, 0], D1[0, 1] = -3.0, 4.0
        D1[-1, -2], D1[-1, -1] = -4.0, 3.0
        D1 *= inv2dx

        D2 = cs.DM.zeros(Nx, Nx)
        for i in range(1, Nx - 1):
            D2[i, i - 1] = 1.0
            D2[i, i] = -2.0
            D2[i, i + 1] = 1.0
        D2[0, 0], D2[0, 1], D2[0, 2] = 2.0, -5.0, 4.0
        D2[-1, -3], D2[-1, -2], D2[-1, -1] = 4.0, -5.0, 2.0

        P = cs.DM.zeros(Nx, N)
        P[1 : N + 1, :] = cs.DM.eye(N)
        R = cs.DM.zeros(N, Nx)
        R[:, 1 : N + 1] = cs.DM.eye(N)

        for k in range(1, Nt + 1):
            u_prev = U_rows[k - 1]
            u_curr = U_rows[k]
            f_k = G_rows[k - 1]

            u_curr_pad = P @ u_curr
            conv_full = 0.5 * (D1 @ cs.power(u_curr_pad, 2))
            diff_full = (visc_var * invdx2) * (D2 @ u_curr_pad)

            conv = R @ conv_full
            diff = R @ diff_full

            rhs = u_prev + dt * (-conv + diff + f_k)
            opti.subject_to(u_curr == rhs)

    def _generate_energy_objective(
        self,
        opti,
        X_var,
        X_index_selector=None,
    ):

        if X_index_selector is None:
            dof = cs.vec(X_var[-1, :])
        elif X_index_selector == "projection":
            dof = cs.vec(X_var)
        else:
            raise ValueError

        control_flat = dof[9 * 128 :]

        self.oc_energy_cost = 0.5 * cs.sumsqr(control_flat)

    def projection_formulate(
        self,
        print_level=2,
        constraint="",
    ):
        """
        Directly project the states to the feasible set.
        """

        assert (
            self.cfg.guidance_method == "projection"
        ), f"guidance_method must be projection, but got {self.cfg.guidance_method}"

        self.oc_cs_opti = cs.Opti()

        self.oc_N_steps = self.cfg.ode_t_steps
        self.oc_dof = 9 * 128 + 10 * 128

        self.oc_u0_param = self.oc_cs_opti.parameter(1, 128)
        self.oc_uT_param = self.oc_cs_opti.parameter(1, 128)

        self.oc_X_single = self.oc_cs_opti.variable(self.oc_dof)
        self.oc_X_single_ref = self.oc_cs_opti.parameter(self.oc_dof)

        self.oc_control_cost = 0.5 * cs.sumsqr(self.oc_X_single - self.oc_X_single_ref)

        if constraint in ["safety_score"]:
            self._apply_safety_constraints(
                self.oc_cs_opti, self.oc_X_single, X_index_selector="projection"
            )
            self._apply_pde_constraints(
                self.oc_cs_opti,
                self.oc_X_single,
                self.oc_u0_param,
                self.oc_uT_param,
                X_index_selector="projection",
            )
        else:
            raise ValueError(f"Unsupported constraint: {constraint}")

        self.oc_cs_opti.minimize(self.oc_control_cost)

        solver_opts = {
            "ipopt.print_level": print_level,
            "ipopt.hessian_approximation": "limited-memory",
        }
        self.oc_cs_opti.solver("ipopt", solver_opts)

    def projection_relaxed_formulate(
        self,
        print_level=2,
        constraint="",
    ):
        """
        Inexact projection.
        """

        self._ensure_safety_bounds()
        self.oc_N_steps = self.cfg.ode_t_steps
        self.oc_dof = 9 * 128 + 10 * 128

    def hardflow_formulate(
        self,
        print_level=2,
        constraint="",
        objective="",
    ):
        """
        Operate on the predicted terminal state.
        """

        assert self.cfg.guidance_method in (
            "hardflow",
            "hardflow_new",
        ), f"guidance_method must be hardflow or hardflow_new, but got {self.cfg.guidance_method}"

        self.oc_cs_opti = cs.Opti()

        self.oc_N_steps = self.cfg.ode_t_steps
        self.oc_dof = 9 * 128 + 10 * 128

        self.oc_u0_param = self.oc_cs_opti.parameter(1, 128)
        self.oc_uT_param = self.oc_cs_opti.parameter(1, 128)

        self.oc_t_param = self.oc_cs_opti.parameter(1)

        self.oc_X_terminal_predicted_ref = self.oc_cs_opti.parameter(self.oc_dof)
        self.oc_X_terminal_predicted = self.oc_cs_opti.variable(self.oc_dof)

        self.oc_control_cost = (
            0.5
            * cs.sumsqr(self.oc_X_terminal_predicted - self.oc_X_terminal_predicted_ref)
            * self.oc_t_param**2
        )

        if constraint in ["safety_score"]:
            self._apply_safety_constraints(
                self.oc_cs_opti,
                self.oc_X_terminal_predicted,
                X_index_selector="projection",
            )
            self._apply_pde_constraints(
                self.oc_cs_opti,
                self.oc_X_terminal_predicted,
                self.oc_u0_param,
                self.oc_uT_param,
                X_index_selector="projection",
            )
        else:
            raise ValueError(f"Unsupported constraint: {constraint}")

        if objective == "":
            self.oc_cs_opti.minimize(self.oc_control_cost)
        elif objective == "control_energy":
            self._generate_energy_objective(
                self.oc_cs_opti,
                self.oc_X_terminal_predicted,
                X_index_selector="projection",
            )
            self.oc_cs_opti.minimize(
                self.oc_control_cost
                + self.oc_energy_cost * self.cfg.hardflow_cost_scale
            )
        else:
            raise ValueError(f"Unsupported objective: {objective}")

        solver_opts = {
            "ipopt.print_level": print_level,
            "ipopt.hessian_approximation": "limited-memory",
        }
        self.oc_cs_opti.solver("ipopt", solver_opts)

    def oc_flow_formulate(
        self,
        print_level=2,
        constraint="",
    ):
        self.oc_N_steps = self.cfg.ode_t_steps
        self.oc_dof = 9 * 128 + 10 * 128

    def warmstart(self, conditions):

        (u0, uT) = conditions

        x = torch.randn(
            self.cfg.warmstart_batch,
            2,
            16,
            128,
            device=self.cfg.device,
        )  # (batch_size, 2, 16, 128)
        x = apply_conditioning(x, u0, uT)

        x_chain = [x.clone()]
        dt = 1.0 / self.oc_N_steps
        for k in range(self.oc_N_steps):
            t = k / self.oc_N_steps
            dx_dt = self.flow_model(x, to_torch(t, device=x.device))

            # zero out velocity at conditioned points
            dx_dt[:, 0, 0, :] = 0.0
            dx_dt[:, 0, 10, :] = 0.0

            # zero out velocity at padded points
            dx_dt[:, 0, 11:, :] = 0.0
            dx_dt[:, 1, 10:, :] = 0.0

            x = x + dx_dt * dt
            x_chain.append(x)

        values = self.value_model(x_chain[-1]).view(-1)
        best_idx = torch.argmax(values).item()

        best_x_chain = torch.stack(
            [x_step[best_idx] for x_step in x_chain], dim=0
        )  # (oc_N_steps+1, 2, 16, 128)

        best_u0_np = to_np(u0)
        best_uT_np = to_np(uT)

        best_state_chain = best_x_chain[:, 0, :11, :]
        best_control_chain = best_x_chain[:, 1, :10, :]

        best_state_chain = best_state_chain.reshape(self.oc_N_steps + 1, -1)
        best_state_chain = best_state_chain[:, 128 : 10 * 128]

        best_control_chain = best_control_chain.reshape(self.oc_N_steps + 1, -1)

        best_dof_chain = torch.cat([best_state_chain, best_control_chain], dim=-1)
        best_dof_chain_np = to_np(best_dof_chain)

        return best_u0_np, best_uT_np, best_dof_chain_np

    def projection_forward(self, conditions, batch_size=1):
        assert batch_size == 1, "batch_size must be 1 for optimal control"
        assert (
            self.cfg.guidance_method == "projection"
        ), f"guidance_method must be projection, but got {self.cfg.guidance_method}"

        best_u0_np, best_uT_np, best_dof_chain_np = self.warmstart(conditions)
        best_final_dof = best_dof_chain_np[-1, :]

        t_start = time.time()

        self.oc_cs_opti.set_value(self.oc_u0_param, best_u0_np)
        self.oc_cs_opti.set_value(self.oc_uT_param, best_uT_np)

        X_optimized = [best_dof_chain_np[0, :]]
        U_optimized = []
        dt = 1.0 / self.oc_N_steps
        best_u0_vec = best_u0_np.reshape(-1)
        best_uT_vec = best_uT_np.reshape(-1)

        for k in range(self.oc_N_steps):
            t_k = k * dt
            x_k = X_optimized[k]

            x_k_state = x_k[: 9 * 128].reshape(9, 128)
            x_k_control = x_k[9 * 128 :].reshape(10, 128)

            x_k_full = np.zeros((1, 2, 16, 128), dtype=x_k.dtype)
            x_k_full[0, 0, 0, :] = best_u0_vec
            x_k_full[0, 0, 1:10, :] = x_k_state
            x_k_full[0, 0, 10, :] = best_uT_vec
            x_k_full[0, 1, :10, :] = x_k_control

            x = to_torch(x_k_full, device=self.cfg.device)

            if self.value_model is not None and self.cfg.projection_gradient_steps > 0:
                for _ in range(self.cfg.projection_gradient_steps):
                    inputs = x.detach().clone().requires_grad_(True)
                    t_tensor = to_torch(t_k, device=inputs.device)
                    dx_dt = self.flow_model(inputs, t_tensor)

                    dx_dt[:, 0, 0, :] = 0.0
                    dx_dt[:, 0, 10, :] = 0.0
                    dx_dt[:, 0, 11:, :] = 0.0
                    dx_dt[:, 1, 10:, :] = 0.0
                    predicted_x1 = inputs + dx_dt * (1.0 - t_k)
                    loss = -self.value_model(predicted_x1)
                    if loss.ndim > 0:
                        loss = loss.mean()
                    (g,) = torch.autograd.grad(loss, inputs, create_graph=False)

                    g[:, 0, 0, :] = 0.0
                    g[:, 0, 10, :] = 0.0
                    g[:, 0, 11:, :] = 0.0
                    g[:, 1, 10:, :] = 0.0

                    x = (inputs - self.cfg.projection_gradient_lr * g).detach()

            x_k_full = to_np(x.cpu()).reshape(1, 2, 16, 128)

            updated_state = x_k_full[0, 0, 1:10, :].reshape(-1)
            updated_control = x_k_full[0, 1, :10, :].reshape(-1)
            x_k = np.concatenate([updated_state, updated_control], axis=-1).astype(
                x_k.dtype, copy=False
            )

            v_k = self.constrained_flow_fn(x_k, t_k, best_u0_np, best_uT_np)
            x_next_ref = x_k + np.array(v_k).squeeze() * dt

            projection_flag = True
            if self.cfg.projection_option == "all":
                pass
            elif self.cfg.projection_option == "late":
                if k < self.oc_N_steps // 2:
                    projection_flag = False
            else:
                raise ValueError(
                    f"Unsupported projection_option: {self.cfg.projection_option}"
                )

            if projection_flag:
                self.oc_cs_opti.set_value(self.oc_X_single_ref, x_next_ref)
                self.oc_cs_opti.set_initial(self.oc_X_single, x_next_ref)
                try:
                    sol = self.oc_cs_opti.solve_limited()
                    x_next = sol.value(self.oc_X_single)
                except RuntimeError as e:
                    print("Solver failed, returning last available value.")
                    x_next = self.oc_cs_opti.debug.value(self.oc_X_single)
            else:
                x_next = x_next_ref

            x_next = np.array(x_next).flatten()

            u_k = (x_next - x_next_ref) / dt
            U_optimized.append(u_k)

            X_optimized.append(x_next)

        optimized_final_dof = X_optimized[-1]

        optimized_dof = np.array(X_optimized)  # (oc_N_steps + 1, dof)
        optimized_state = optimized_dof[:, : 9 * 128]

        optimized_state = optimized_state.reshape(self.oc_N_steps + 1, 9, 128)

        # broadcast u0 and uT from (1, 128) to (11, 1, 128)
        best_u0_expanded = np.broadcast_to(
            best_u0_np[None, :], (self.oc_N_steps + 1, 1, 128)
        )
        best_uT_expanded = np.broadcast_to(
            best_uT_np[None, :], (self.oc_N_steps + 1, 1, 128)
        )
        optimized_state = np.concatenate(
            [best_u0_expanded, optimized_state, best_uT_expanded],
            axis=1,
        )

        optimized_control = optimized_dof[:, 9 * 128 :]
        optimized_control = optimized_control.reshape(self.oc_N_steps + 1, 10, 128)

        optimized_x_chain = np.zeros(
            (1, self.oc_N_steps + 1, 2, 16, 128), dtype=optimized_state.dtype
        )
        optimized_x_chain[0, :, 0, :11, :] = optimized_state
        optimized_x_chain[0, :, 1, :10, :] = optimized_control

        print()
        print(
            f"    Norm of Control Inputs: {np.linalg.norm(np.array(U_optimized)):.6f}",
            flush=True,
        )
        print()

        optimized_final_x = optimized_x_chain[:, -1, :, :, :]

        normed_controls = optimized_final_x[:, 1, :10, :]
        controls = self.normalizer.denormalize_f(to_np(normed_controls))

        normed_states = optimized_final_x[:, 0, :11, :]
        states = self.normalizer.denormalize_u(to_np(normed_states))

        values = self.value_model(to_torch(optimized_final_x))
        values = to_np(values)

        trajectories = Trajectories(controls, states, values)

        control = controls[0]

        x1_estimation = self.x1_estimate(to_torch(optimized_x_chain))
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(optimized_x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {"computation_time": computation_time}
        return control, trajectories, x_chain, x1_estimation, info

    def projection_relaxed_forward(self, conditions, batch_size=1):
        assert batch_size == 1, "batch_size must be 1 for optimal control"
        assert (
            self.cfg.guidance_method == "projection_relaxed"
        ), f"guidance_method must be projection_relaxed, but got {self.cfg.guidance_method}"
        assert (
            self.cfg.constraint == "safety_score"
        ), f"projection_relaxed supports only safety_score constraint, got {self.cfg.constraint}"

        best_u0_np, best_uT_np, best_dof_chain_np = self.warmstart(conditions)

        t_start = time.time()

        lower_bounds = self._safety_state_lower
        upper_bounds = self._safety_state_upper

        X_optimized = [best_dof_chain_np[0, :]]
        U_optimized = []
        dt = 1.0 / self.oc_N_steps

        best_u0_vec = best_u0_np.reshape(-1)
        best_uT_vec = best_uT_np.reshape(-1)

        lam_upper = np.zeros_like(upper_bounds, dtype=np.float32)
        lam_lower = np.zeros_like(lower_bounds, dtype=np.float32)

        al_steps = int(getattr(self.cfg, "oc_inexact_al_steps", 10))
        al_lr = float(getattr(self.cfg, "oc_inexact_al_lr", 0.1))
        rho = float(getattr(self.cfg, "oc_inexact_al_rho", 10.0))

        state_slice = slice(0, 9 * 128)

        for k in range(self.oc_N_steps):
            t_k = k * dt
            x_k = X_optimized[k]
            dof_dtype = x_k.dtype

            x_k_full = self._dof_to_full_tensor(
                x_k, best_u0_vec, best_uT_vec, dtype=x_k.dtype
            )
            x = to_torch(x_k_full, device=self.cfg.device)

            if self.value_model is not None and self.cfg.projection_gradient_steps > 0:
                for _ in range(self.cfg.projection_gradient_steps):
                    inputs = x.detach().clone().requires_grad_(True)
                    t_tensor = to_torch(t_k, device=inputs.device)
                    dx_dt = self.flow_model(inputs, t_tensor)

                    dx_dt[:, 0, 0, :] = 0.0
                    dx_dt[:, 0, 10, :] = 0.0
                    dx_dt[:, 0, 11:, :] = 0.0
                    dx_dt[:, 1, 10:, :] = 0.0
                    predicted_x1 = inputs + dx_dt * (1.0 - t_k)
                    loss = -self.value_model(predicted_x1)
                    if loss.ndim > 0:
                        loss = loss.mean()
                    (g,) = torch.autograd.grad(loss, inputs, create_graph=False)

                    g[:, 0, 0, :] = 0.0
                    g[:, 0, 10, :] = 0.0
                    g[:, 0, 11:, :] = 0.0
                    g[:, 1, 10:, :] = 0.0

                    x = (inputs - self.cfg.projection_gradient_lr * g).detach()

            x_k_full = to_np(x.cpu())
            x_k = self._full_tensor_to_dof(x_k_full).astype(dof_dtype, copy=False)

            v_k = self.constrained_flow_fn(x_k, t_k, best_u0_np, best_uT_np)
            x_next_ref = x_k + np.array(v_k).squeeze() * dt

            x_var = x_next_ref.astype(np.float32, copy=True)

            for _ in range(al_steps):
                grad = (x_var - x_next_ref).astype(np.float32)
                state_vals = x_var[state_slice]

                c_upper = state_vals - upper_bounds
                c_lower = lower_bounds - state_vals

                r_upper = np.maximum(c_upper + lam_upper / rho, 0.0)
                r_lower = np.maximum(c_lower + lam_lower / rho, 0.0)

                grad_state = rho * (r_upper - r_lower)
                grad[state_slice] += grad_state

                x_var = x_var - al_lr * grad

                lam_upper = np.maximum(0.0, lam_upper + rho * c_upper)
                lam_lower = np.maximum(0.0, lam_lower + rho * c_lower)

            x_var[state_slice] = np.clip(x_var[state_slice], lower_bounds, upper_bounds)
            x_next = x_var.astype(dof_dtype, copy=False)

            u_k = (x_next - x_next_ref) / dt
            U_optimized.append(u_k)

            X_optimized.append(x_next)

        optimized_final_dof = X_optimized[-1]

        optimized_dof = np.array(X_optimized)
        optimized_state = optimized_dof[:, : 9 * 128].reshape(
            self.oc_N_steps + 1, 9, 128
        )

        best_u0_expanded = np.broadcast_to(
            best_u0_vec[None, :], (self.oc_N_steps + 1, 1, 128)
        )
        best_uT_expanded = np.broadcast_to(
            best_uT_vec[None, :], (self.oc_N_steps + 1, 1, 128)
        )
        optimized_state = np.concatenate(
            [best_u0_expanded, optimized_state, best_uT_expanded], axis=1
        )

        optimized_control = optimized_dof[:, 9 * 128 :].reshape(
            self.oc_N_steps + 1, 10, 128
        )

        optimized_x_chain = np.zeros(
            (1, self.oc_N_steps + 1, 2, 16, 128), dtype=optimized_state.dtype
        )
        optimized_x_chain[0, :, 0, :11, :] = optimized_state
        optimized_x_chain[0, :, 1, :10, :] = optimized_control

        print()
        print(
            f"    Norm of Control Inputs: {np.linalg.norm(np.array(U_optimized)):.6f}",
            flush=True,
        )
        print()

        optimized_final_x = optimized_x_chain[:, -1, :, :, :]

        normed_controls = optimized_final_x[:, 1, :10, :]
        controls = self.normalizer.denormalize_f(to_np(normed_controls))

        normed_states = optimized_final_x[:, 0, :11, :]
        states = self.normalizer.denormalize_u(to_np(normed_states))

        values = self.value_model(to_torch(optimized_final_x))
        values = to_np(values)

        trajectories = Trajectories(controls, states, values)

        control = controls[0]

        x1_estimation = self.x1_estimate(to_torch(optimized_x_chain))
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(optimized_x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {"computation_time": computation_time}
        return control, trajectories, x_chain, x1_estimation, info

    def hardflow_forward(self, conditions, batch_size=1):
        assert batch_size == 1, "batch_size must be 1 for optimal control"
        assert (
            self.cfg.guidance_method == "hardflow"
        ), f"guidance_method must be hardflow, but got {self.cfg.guidance_method}"

        best_u0_np, best_uT_np, best_dof_chain_np = self.warmstart(conditions)
        best_final_dof = best_dof_chain_np[-1, :]

        t_start = time.time()

        self.oc_cs_opti.set_value(self.oc_u0_param, best_u0_np)
        self.oc_cs_opti.set_value(self.oc_uT_param, best_uT_np)

        X_optimized = [best_dof_chain_np[0, :]]
        U_optimized = []
        dt = 1.0 / self.oc_N_steps
        for k in range(self.oc_N_steps):
            t_k = k * dt
            x_k = X_optimized[k]
            v_k = self.constrained_flow_fn(x_k, t_k, best_u0_np, best_uT_np)
            x_next_ref = x_k + np.array(v_k).squeeze() * dt

            control_flag = True
            if self.cfg.hardflow_activation == "all":
                pass
            elif self.cfg.hardflow_activation == "late":
                if k < self.oc_N_steps // 2:
                    control_flag = False
            else:
                raise ValueError(
                    f"Unsupported hardflow_activation: {self.cfg.hardflow_activation}"
                )

            if control_flag:
                x_terminal_predicted_ref = (
                    x_next_ref
                    + (1.0 - t_k - dt)
                    * self.constrained_flow_fn(
                        x_next_ref, t_k + dt, best_u0_np, best_uT_np
                    ).T
                )

                self.oc_cs_opti.set_value(self.oc_t_param, t_k + dt)

                self.oc_cs_opti.set_value(
                    self.oc_X_terminal_predicted_ref, x_terminal_predicted_ref
                )
                self.oc_cs_opti.set_initial(
                    self.oc_X_terminal_predicted, x_terminal_predicted_ref
                )

                try:
                    sol = self.oc_cs_opti.solve_limited()
                    x_terminal_predicted = sol.value(self.oc_X_terminal_predicted)
                except RuntimeError as e:
                    print("Solver failed, returning last available value.")
                    x_terminal_predicted = self.oc_cs_opti.debug.value(
                        self.oc_X_terminal_predicted
                    )

                x_next = x_next_ref + (t_k + dt) * (
                    x_terminal_predicted - x_terminal_predicted_ref
                )
            else:
                x_next = x_next_ref

            x_next = np.array(x_next).flatten()

            u_k = (x_next - x_next_ref) / dt
            U_optimized.append(u_k)

            X_optimized.append(x_next)

        optimized_final_dof = X_optimized[-1]

        optimized_dof = np.array(X_optimized)  # (oc_N_steps + 1, dof)
        optimized_state = optimized_dof[:, : 9 * 128]

        optimized_state = optimized_state.reshape(self.oc_N_steps + 1, 9, 128)

        # broadcast u0 and uT from (1, 128) to (11, 1, 128)
        best_u0_expanded = np.broadcast_to(
            best_u0_np[None, :], (self.oc_N_steps + 1, 1, 128)
        )
        best_uT_expanded = np.broadcast_to(
            best_uT_np[None, :], (self.oc_N_steps + 1, 1, 128)
        )
        optimized_state = np.concatenate(
            [best_u0_expanded, optimized_state, best_uT_expanded],
            axis=1,
        )

        optimized_control = optimized_dof[:, 9 * 128 :]
        optimized_control = optimized_control.reshape(self.oc_N_steps + 1, 10, 128)

        optimized_x_chain = np.zeros(
            (1, self.oc_N_steps + 1, 2, 16, 128), dtype=optimized_state.dtype
        )
        optimized_x_chain[0, :, 0, :11, :] = optimized_state
        optimized_x_chain[0, :, 1, :10, :] = optimized_control

        print()
        print(
            f"    Norm of Control Inputs: {np.linalg.norm(np.array(U_optimized)):.6f}",
            flush=True,
        )
        print()

        optimized_final_x = optimized_x_chain[:, -1, :, :, :]

        normed_controls = optimized_final_x[:, 1, :10, :]
        controls = self.normalizer.denormalize_f(to_np(normed_controls))

        normed_states = optimized_final_x[:, 0, :11, :]
        states = self.normalizer.denormalize_u(to_np(normed_states))

        values = self.value_model(to_torch(optimized_final_x))
        values = to_np(values)

        trajectories = Trajectories(controls, states, values)

        control = controls[0]

        x1_estimation = self.x1_estimate(to_torch(optimized_x_chain))
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(optimized_x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {"computation_time": computation_time}
        return control, trajectories, x_chain, x1_estimation, info

    def constrained_flow_fn_torch(self, dof, t, u0, uT):
        """
        Pure-PyTorch evaluation of the same map that
        ``run.eval.create_constrained_casadi_functions`` builds in CasADi:
        unpack ``dof`` together with the boundary parameters ``u0``/``uT``
        into the (1, 2, 16, 128) tensor expected by the flow U-Net, run a
        forward pass, and pack the inner state/control time slices back into
        the dof-shaped output.

        dof: (1, 9*128 + 10*128) = (1, 19*128)
        u0, uT: (1, 128)
        """
        device = dof.device
        S = 128
        T = 11
        Tp = 16

        dof_state = dof[:, : 9 * S]
        dof_control = dof[:, 9 * S :]

        state = torch.cat([u0, dof_state, uT], dim=-1).reshape(1, T, S)
        state = torch.nn.functional.pad(state, (0, 0, 0, Tp - T))

        control = dof_control.reshape(1, T - 1, S)
        control = torch.nn.functional.pad(control, (0, 0, 0, Tp - (T - 1)))

        x = torch.stack([state, control], dim=1)

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=x.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        flow_output = self.flow_model(x, t)

        state_flow = flow_output[:, 0, 1 : T - 1, :].reshape(1, (T - 2) * S)
        control_flow = flow_output[:, 1, : T - 1, :].reshape(1, (T - 1) * S)

        return torch.cat([state_flow, control_flow], dim=-1)

    def hardflow_new_forward(self, conditions, batch_size=1):
        """
        Variant of hardflow_forward that evaluates the reference flow in
        PyTorch instead of through the l4casadi bridge. The CasADi NLP solved
        at every ODE step is purely algebraic, so the optimization itself does
        not depend on the neural network.
        """
        assert batch_size == 1, "batch_size must be 1 for optimal control"
        assert (
            self.cfg.guidance_method == "hardflow_new"
        ), f"guidance_method must be hardflow_new, but got {self.cfg.guidance_method}"

        best_u0_np, best_uT_np, best_dof_chain_np = self.warmstart(conditions)
        best_final_dof = best_dof_chain_np[-1, :]

        t_start = time.time()

        device = self.cfg.device
        best_u0_torch = to_torch(best_u0_np, device=device).reshape(1, 128)
        best_uT_torch = to_torch(best_uT_np, device=device).reshape(1, 128)

        def flow_eval_np(x_np, t):
            x_torch = to_torch(x_np, device=device).reshape(1, self.oc_dof)
            with torch.no_grad():
                v = self.constrained_flow_fn_torch(
                    x_torch, t, best_u0_torch, best_uT_torch
                )
            return to_np(v).reshape(-1)

        self.oc_cs_opti.set_value(self.oc_u0_param, best_u0_np)
        self.oc_cs_opti.set_value(self.oc_uT_param, best_uT_np)

        X_optimized = [best_dof_chain_np[0, :]]
        U_optimized = []
        dt = 1.0 / self.oc_N_steps
        for k in range(self.oc_N_steps):
            t_k = k * dt
            x_k = X_optimized[k]
            v_k = flow_eval_np(x_k, t_k)
            x_next_ref = x_k + v_k * dt

            control_flag = True
            if self.cfg.hardflow_activation == "all":
                pass
            elif self.cfg.hardflow_activation == "late":
                if k < self.oc_N_steps // 2:
                    control_flag = False
            else:
                raise ValueError(
                    f"Unsupported hardflow_activation: {self.cfg.hardflow_activation}"
                )

            if control_flag:
                v_next = flow_eval_np(x_next_ref, t_k + dt)
                x_terminal_predicted_ref = x_next_ref + (1.0 - t_k - dt) * v_next

                self.oc_cs_opti.set_value(self.oc_t_param, t_k + dt)

                self.oc_cs_opti.set_value(
                    self.oc_X_terminal_predicted_ref, x_terminal_predicted_ref
                )
                self.oc_cs_opti.set_initial(
                    self.oc_X_terminal_predicted, x_terminal_predicted_ref
                )

                try:
                    sol = self.oc_cs_opti.solve_limited()
                    x_terminal_predicted = sol.value(self.oc_X_terminal_predicted)
                except RuntimeError as e:
                    print("Solver failed, returning last available value.")
                    x_terminal_predicted = self.oc_cs_opti.debug.value(
                        self.oc_X_terminal_predicted
                    )

                x_next = x_next_ref + (t_k + dt) * (
                    x_terminal_predicted - x_terminal_predicted_ref
                )
            else:
                x_next = x_next_ref

            x_next = np.array(x_next).flatten()

            u_k = (x_next - x_next_ref) / dt
            U_optimized.append(u_k)

            X_optimized.append(x_next)

        optimized_final_dof = X_optimized[-1]

        optimized_dof = np.array(X_optimized)
        optimized_state = optimized_dof[:, : 9 * 128]
        optimized_state = optimized_state.reshape(self.oc_N_steps + 1, 9, 128)

        best_u0_expanded = np.broadcast_to(
            best_u0_np[None, :], (self.oc_N_steps + 1, 1, 128)
        )
        best_uT_expanded = np.broadcast_to(
            best_uT_np[None, :], (self.oc_N_steps + 1, 1, 128)
        )
        optimized_state = np.concatenate(
            [best_u0_expanded, optimized_state, best_uT_expanded],
            axis=1,
        )

        optimized_control = optimized_dof[:, 9 * 128 :]
        optimized_control = optimized_control.reshape(self.oc_N_steps + 1, 10, 128)

        optimized_x_chain = np.zeros(
            (1, self.oc_N_steps + 1, 2, 16, 128), dtype=optimized_state.dtype
        )
        optimized_x_chain[0, :, 0, :11, :] = optimized_state
        optimized_x_chain[0, :, 1, :10, :] = optimized_control

        print()
        print(
            f"    Norm of Control Inputs: {np.linalg.norm(np.array(U_optimized)):.6f}",
            flush=True,
        )
        print()

        optimized_final_x = optimized_x_chain[:, -1, :, :, :]

        normed_controls = optimized_final_x[:, 1, :10, :]
        controls = self.normalizer.denormalize_f(to_np(normed_controls))

        normed_states = optimized_final_x[:, 0, :11, :]
        states = self.normalizer.denormalize_u(to_np(normed_states))

        values = self.value_model(to_torch(optimized_final_x))
        values = to_np(values)

        trajectories = Trajectories(controls, states, values)

        control = controls[0]

        x1_estimation = self.x1_estimate(to_torch(optimized_x_chain))
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(optimized_x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {"computation_time": computation_time}
        return control, trajectories, x_chain, x1_estimation, info

    def gradient_guidance_forward(self, conditions, batch_size=1):
        """
        Use gradient guidance to generate actions.
        """
        assert (
            self.cfg.guidance_method == "gradient_guidance"
        ), f"guidance_method must be gradient_guidance, but got {self.cfg.guidance_method}"

        (u0, uT) = conditions

        t_start = time.time()

        solver = ConditionedODESolver(
            self.flow_model,
            value=self.value_model,
            ode_method=self.cfg.ode_solver,
            guidance_lr=self.cfg.guidance_lr,
            guide_stpes=self.cfg.guidance_steps,
        )

        x = torch.randn(
            batch_size,
            2,
            16,
            128,
            device=self.cfg.device,
        )  # (batch_size, 2, 16, 128)
        x = apply_conditioning(x, u0, uT)

        x, x_chain = solver(
            x,
            t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps + 1, device=x.device
            )[:-1],
            return_chain=True,
        )

        normed_controls = x[:, 1, :10, :]
        controls = self.normalizer.denormalize_f(to_np(normed_controls))

        normed_states = x[:, 0, :11, :]
        states = self.normalizer.denormalize_u(to_np(normed_states))

        values = self.value_model(x)
        values = to_np(values)

        trajectories = Trajectories(controls, states, values)

        control = controls[0]

        x1_estimation = self.x1_estimate(x_chain)
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {"computation_time": computation_time}
        return control, trajectories, x_chain, x1_estimation, info

    def oc_flow_forward(self, conditions, batch_size=1):
        assert batch_size == 1, "batch_size must be 1 for optimal control"
        assert (
            self.cfg.guidance_method == "oc_flow"
        ), f"guidance_method must be oc_flow, but got {self.cfg.guidance_method}"

        best_u0_np, best_uT_np, best_dof_chain_np = self.warmstart(conditions)

        t_start = time.time()

        device = self.cfg.device
        dt = 1.0 / self.oc_N_steps

        best_u0 = to_torch(best_u0_np, device=device).float()
        best_uT = to_torch(best_uT_np, device=device).float()
        if best_u0.dim() == 1:
            best_u0 = best_u0.unsqueeze(0)
        if best_uT.dim() == 1:
            best_uT = best_uT.unsqueeze(0)

        best_dof_chain = to_torch(best_dof_chain_np, device=device).float()

        U = torch.zeros(
            self.oc_N_steps, self.oc_dof, device=device, dtype=torch.float32
        )

        def eval_terminal_cost(xN):
            full = self._dof_to_full_tensor_torch(xN.unsqueeze(0), best_u0, best_uT)
            cost = -self.value_model(full)
            return cost.squeeze()

        def rollout(U_steps):
            x = best_dof_chain[0].clone()
            X = [x]
            for k in range(self.oc_N_steps):
                t_k = k * dt
                flow = self._constrained_flow_torch(
                    x.unsqueeze(0), t_k, best_u0, best_uT
                ).squeeze(0)
                x = x + dt * (flow + U_steps[k])
                X.append(x)
            return torch.stack(X)

        def gradient_U(U_steps, gamma_u=0.0):
            X = rollout(U_steps)
            xN = X[-1].clone().detach().requires_grad_(True)
            terminal_cost = eval_terminal_cost(xN)
            lam = (
                torch.autograd.grad(terminal_cost, xN, retain_graph=False)[0]
                .detach()
                .clone()
            )

            grad = torch.zeros_like(U_steps)

            for j in reversed(range(self.oc_N_steps)):
                lam_next = lam

                xj = X[j].clone().detach().requires_grad_(True)
                t_j = j * dt
                u_j = U_steps[j].clone().detach()

                def F_j(x_flat):
                    x_local = x_flat.view_as(xj)
                    flow = self._constrained_flow_torch(
                        x_local.unsqueeze(0), t_j, best_u0, best_uT
                    ).squeeze(0)
                    return (x_local + dt * (flow + u_j)).view(-1)

                _, vjp = torch.autograd.functional.vjp(
                    F_j, xj.view(-1), v=lam_next.view(-1)
                )
                lam = vjp.view_as(xj).detach()

                g_j = dt * lam_next + (gamma_u * u_j if gamma_u > 0.0 else 0.0)
                grad[j] = g_j

            return grad, X

        gamma_u = 0.005
        for _ in range(self.cfg.oc_flow_steps):
            grad_U, _ = gradient_U(U, gamma_u=gamma_u)
            U = U - self.cfg.oc_flow_lr * grad_U

        X_optimized = rollout(U)
        U_optimized_np = U.detach().cpu().numpy()
        X_optimized_np = X_optimized.detach().cpu().numpy()

        optimized_state = X_optimized_np[:, : 9 * 128].reshape(
            self.oc_N_steps + 1, 9, 128
        )
        best_u0_expanded = np.broadcast_to(
            best_u0_np.reshape(1, 1, 128), (self.oc_N_steps + 1, 1, 128)
        )
        best_uT_expanded = np.broadcast_to(
            best_uT_np.reshape(1, 1, 128), (self.oc_N_steps + 1, 1, 128)
        )
        optimized_state = np.concatenate(
            [best_u0_expanded, optimized_state, best_uT_expanded], axis=1
        )

        optimized_control = X_optimized_np[:, 9 * 128 :].reshape(
            self.oc_N_steps + 1, 10, 128
        )

        optimized_x_chain = np.zeros(
            (1, self.oc_N_steps + 1, 2, 16, 128), dtype=optimized_state.dtype
        )
        optimized_x_chain[0, :, 0, :11, :] = optimized_state
        optimized_x_chain[0, :, 1, :10, :] = optimized_control

        optimized_final_x = optimized_x_chain[:, -1, :, :, :]

        normed_controls = optimized_final_x[:, 1, :10, :]
        controls = self.normalizer.denormalize_f(to_np(normed_controls))

        normed_states = optimized_final_x[:, 0, :11, :]
        states = self.normalizer.denormalize_u(to_np(normed_states))

        values = self.value_model(to_torch(optimized_final_x))
        values = to_np(values)

        trajectories = Trajectories(controls, states, values)

        control = controls[0]

        x1_estimation = self.x1_estimate(to_torch(optimized_x_chain))
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(optimized_x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {"computation_time": computation_time}
        return control, trajectories, x_chain, x1_estimation, info
