from typing import Union
import torch
from torch import nn


def apply_conditioning(x, u0=None, uT=None):
    """
    Args:
        x: (B, 2, 16, 128) - [u, f] channels
        u0: Initial condition
        uT: Final condition
    """
    if u0 is not None:
        x[:, 0, 0, :] = u0
    if uT is not None:
        x[:, 0, 10, :] = uT

    x[:, 0, 11:, :] = 0.0
    x[:, 1, 10:, :] = 0.0

    return x


def apply_conditioning_from_conditioned_x(x, x_conditioned):

    x[:, 0, 0, :] = x_conditioned[:, 0, 0, :]
    x[:, 0, 10, :] = x_conditioned[:, 10, :, :]

    return x


class FlowMatcher:
    def __init__(
        self,
        model: nn.Module,
        flow_matching_type: str = "cfm",
    ):
        self.model = model
        self.flow_matching_type = flow_matching_type
        self.FM = self.get_flow_matcher()

    def get_flow_matcher(self):
        if self.flow_matching_type == "cfm":
            FM = ConditionalFlowMatcher()
        else:
            raise ValueError(
                f"Flow matching type {self.flow_matching_type} not supported"
            )
        return FM

    def loss(self, x, u0=None, uT=None):
        """
        Args:
            x: Data tensor of shape (B, 2, 16, 128) - [u, f] channels
            u0: Initial condition (B, 128) or None
            uT: Final condition (B, 128) or None
        """
        x0 = torch.randn_like(x)

        x0 = apply_conditioning(x0, u0, uT)
        x1 = apply_conditioning(x.clone(), u0, uT)

        if self.flow_matching_type == "cfm":
            t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1, t=None)
        else:
            raise ValueError(
                f"Flow matching type {self.flow_matching_type} not supported"
            )

        vt = self.model(xt, t)
        loss = torch.nn.functional.mse_loss(vt, ut)

        infos = {"loss": loss.item()}
        return loss, infos


def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatcher:
    def __init__(self, sigma: Union[float, int] = 0.0):
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t must have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut
