import torch


def _temporal_weights(
    temporal_steps: int,
    profile_type: str = "constant",
    device=None,
    dtype=torch.float32,
):
    device = device or torch.device("cpu")
    if profile_type == "constant":
        return torch.ones(temporal_steps, device=device, dtype=dtype)
    if profile_type == "quadratic":
        tau = torch.linspace(0.0, 1.0, temporal_steps, device=device, dtype=dtype)
        return 2.0 * tau**2 - 2.0 * tau + 1.0
    raise ValueError(f"Unsupported safety profile type: {profile_type}")


def build_safety_upper_profile(
    spatial_dim: int,
    safety_threshold: float,
    profile_type: str = "constant",
    safety_threshold_margin: float = 0.0,
    temporal_steps: int = 1,
    device=None,
    dtype=torch.float32,
):

    device = device or torch.device("cpu")
    weights = _temporal_weights(
        temporal_steps=temporal_steps,
        profile_type=profile_type,
        device=device,
        dtype=dtype,
    )
    base = safety_threshold * weights.view(temporal_steps, 1)
    base = base.expand(temporal_steps, spatial_dim).clone()
    base = base - safety_threshold_margin
    return base.clamp_min(0.0)


def build_safety_lower_profile(
    spatial_dim: int,
    safety_threshold: float,
    profile_type: str = "constant",
    safety_threshold_margin: float = 0.0,
    temporal_steps: int = 1,
    device=None,
    dtype=torch.float32,
):

    upper = build_safety_upper_profile(
        spatial_dim=spatial_dim,
        safety_threshold=safety_threshold,
        profile_type=profile_type,
        safety_threshold_margin=safety_threshold_margin,
        temporal_steps=temporal_steps,
        device=device,
        dtype=dtype,
    )
    return -upper
