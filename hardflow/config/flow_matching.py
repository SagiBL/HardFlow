from dataclasses import dataclass


@dataclass
class FlowMatchingEvaluationConfig:
    seed: int = 0
    device: str = "cuda:0"

    log_folder: str = "./logs"
    data_path: str = "datasets/burgers"
    flow_exp_name: str = ""
    eval_exp_name: str = ""
    flow_cp: int = 40

    n_test_samples: int = 5
    safety_threshold: float = 0.8
    safety_threshold_margin: float = 0.0
    safety_constraint_profile: str = "quadratic"

    batch_size: int = 1
    ode_solver: str = "euler"
    ode_t_span: tuple = (0, 1)
    ode_t_steps: int = 20

    guidance_method: str = "original"

    ss_batch: int = 32

    value_objective_scale: float = 1.0
    value_objective: str = ""
    value_constraint_scale: float = 1.0

    cost: str = ""
    constraint: str = ""
    dynamics_constraint: bool = False
    projection_option: str = "all"
    projection_gradient_lr: float = 0.1
    projection_gradient_steps: int = 20
    hardflow_activation: str = "all"
    warmstart_batch: int = 1
    solver_print_level: int = 5

    hardflow_cost_scale: float = 1.0
    cost_scale: float = 1.0

    oc_flow_lr: float = 10.0
    oc_flow_steps: int = 20

    guidance_lr: float = 0.1
    guidance_steps: int = 20
