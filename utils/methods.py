import os
import time
import json
from absl import flags
import torch

import RectifiedFlow.datasets as datasets

from .method_utils import (
    get_img,
    embed_to_latent,
    save_img,
    generate_traj,
    generate_traj_with_guidance,
)


import warnings

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS


def sanitize_prompt_for_filename(prompt):
    prompt_mapping = {
        "A photo of an old face.": "old",
        "A photo of a sad face.": "sad",
        "A photo of a smiling face.": "smile",
        "A photo of an angry face.": "angry",
        "A photo of a face with curly hair.": "curly",
    }
    return prompt_mapping.get(
        prompt,
        prompt.replace("A photo of ", "").replace(".", "").replace(" ", "_").lower(),
    )


def oc_flow_optimization(
    z0,
    u_ind,
    dynamics,
    generate_traj,
    L_N,
    x_ref,  # reference image for LPIPS
    lpips_model,
    N=100,
    number_of_iterations=25,
    lr=2.5,
    lpips_threshold=0.1,
    constraint_margin=0.0,
    weight_decay=0.995,
):
    device, shape = z0.device, z0.shape
    batch_size = shape[0]
    assert batch_size == 1

    eps = 1e-3
    dt = 1.0 / N

    # helper: penalty for constraint violation
    def penalty(xN):

        # constraint: g(xN) = LPIPS(xN, x_ref) - threshold + margin <= 0
        d = lpips_model(xN, x_ref).view(xN.shape[0])
        g = d - lpips_threshold + constraint_margin

        hinge = torch.clamp(g, min=0.0)

        return (hinge**2).mean()

    u = {j: torch.zeros_like(z0, device=device) for j in u_ind}
    for j in u:
        u[j].requires_grad = False
        u[j].grad = torch.zeros_like(u[j], device=device)

    L_best, opt_u = -float("inf"), {j: u[j].clone() for j in u}

    for iter in range(number_of_iterations):
        z_traj = generate_traj(dynamics, z0, u=u, N=N)

        xN = z_traj[-1].to(device).detach().clone().requires_grad_(True)

        V_N = -L_N(xN) - penalty(xN)
        lam = torch.autograd.grad(V_N, xN)[0].detach().clone()  # \lambda_N

        L_curr = V_N.item()
        if L_curr > L_best:
            L_best = L_curr
            for j in u:
                opt_u[j] = u[j].detach().clone()

        for j in reversed(range(N)):  # N-1, ..., 0

            lam_next = lam  # \lambda_{j+1}

            xj = z_traj[j].to(device).detach().clone().requires_grad_(True)
            tj = torch.full((batch_size,), j / N * (1.0 - eps) + eps, device=device)

            def F_j(x_flat):
                x = x_flat.view(shape)
                return (x + dynamics(x, tj * 999) * dt).view(-1)

            _, vjp = torch.autograd.functional.vjp(
                F_j, xj.view(-1), v=lam_next.view(-1)
            )
            lam = vjp.view(shape).detach()  # \lambda_j

            u[j].grad = lam_next.detach().clone()

        for j in u:
            u[j] = weight_decay * u[j] + batch_size * lr * u[j].grad

    return opt_u


def hardflow_optimization(
    z0,
    dynamics,
    L_N,
    x_ref,  # reference image for LPIPS
    lpips_model,
    N=100,
    gradient_steps=25,
    dual_update_frequency=5,
    lr=2.5,
    lpips_threshold=0.1,
    constraint_margin=0.0,
    rho=5.0,
    rho_mult=1.5,
    regularization=0.0,
):
    device, shape = z0.device, z0.shape
    batch_size = shape[0]
    assert batch_size == 1

    eps = 1e-3
    dt = 1.0 / N

    z = z0.detach().clone()
    traj = []
    traj.append(z.detach().clone())

    # helper: terminal augmented Lagrangian value
    def V_AL(xN, lam_dual_given, rho_given):
        V_base = -L_N(xN)

        # constraint: g(xN) = LPIPS(xN, x_ref) - threshold + margin <= 0
        d = lpips_model(xN, x_ref).view(xN.shape[0])
        g = d - lpips_threshold + constraint_margin

        hinge = torch.clamp(g, min=0.0)

        return V_base - lam_dual_given * g.mean() - 0.5 * rho_given * (hinge**2).mean()

    for i in range(N):
        t = torch.ones(batch_size, device=device) * i / N * (1.0 - eps) + eps

        with torch.no_grad():
            vt = dynamics(z, t * 999)
            z_prediction_ref = z + vt * (1.0 - t)
            z_next_ref = z + vt * dt

        inputs = z_prediction_ref.detach().requires_grad_(True)

        use_constraint = i >= N // 2
        lam_dual = torch.tensor(0.0, device=device) if use_constraint else None
        rho_step = rho if use_constraint else None

        def get_objective(inputs_val, lam=None, rho_val=None):
            reg_term = (
                (t + dt) ** 2
                * ((inputs_val - z_prediction_ref) ** 2).mean()
                * regularization
            )
            if use_constraint:
                return V_AL(inputs_val, lam, rho_val) + reg_term
            else:
                return -L_N(inputs_val) + reg_term

        for j in range(gradient_steps):
            objective = get_objective(inputs, lam_dual, rho_step)
            (grad,) = torch.autograd.grad(
                objective,
                inputs,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )
            inputs = (inputs + lr * grad).detach().requires_grad_(True)
            del grad

            if use_constraint and (j + 1) % dual_update_frequency == 0:
                with torch.no_grad():
                    d_cur = lpips_model(inputs, x_ref).view(inputs.shape[0])
                    g_cur = float(
                        (d_cur - lpips_threshold + constraint_margin).mean().item()
                    )
                lam_dual = torch.clamp(lam_dual + rho_step * g_cur, min=0.0)
                if g_cur > 0.02:
                    rho_step = rho_step * max(1.0, rho_mult)

        z = z_next_ref + (t + dt) * (inputs - z_prediction_ref)

        traj.append(z.detach().clone())

    return traj


def gradient_guidance_optimization(
    z0,
    dynamics,
    L_N,
    x_ref,  # reference image for LPIPS
    lpips_model,
    N=100,
    gradient_steps=25,
    lr=2.5,
    lpips_threshold=0.1,
    constraint_margin=0.0,
):
    device, shape = z0.device, z0.shape
    batch_size = shape[0]
    assert batch_size == 1

    def penalty(xN):

        d = lpips_model(xN, x_ref).view(xN.shape[0])
        g = d - lpips_threshold + constraint_margin

        hinge = torch.clamp(g, min=0.0)

        return (hinge**2).mean()

    full_L_N = lambda x: L_N(x) + penalty(x)

    traj = generate_traj_with_guidance(
        dynamics, z0, N=N, L_N=full_L_N, alpha_L=lr, guidance_steps=gradient_steps
    )

    return traj


def projection_relaxed_optimization(
    z0,
    dynamics,
    L_N,
    x_ref,  # reference image for LPIPS
    lpips_model,
    N=100,
    gradient_steps=25,
    lr=2.5,
    lpips_threshold=0.1,
    constraint_margin=0.0,
    lr_al=2.5,
    rho=5.0,
    rho_mult=1.0,
    al_steps=5,
):
    device, shape = z0.device, z0.shape
    batch_size = shape[0]
    assert batch_size == 1

    def V_AL(x_new, x_current, lam_dual_given, rho_given):
        V_base = -((x_new - x_current) ** 2).mean()

        d = lpips_model(x_new, x_ref).view(x_new.shape[0])
        g = d - lpips_threshold + constraint_margin

        hinge = torch.clamp(g, min=0.0)

        return V_base - lam_dual_given * g.mean() - 0.5 * rho_given * (hinge**2).mean()

    eps = 1e-3
    dt = 1.0 / N

    z = z0.detach().clone()
    traj = [z.clone()]

    for i in range(N):
        t = (i / N) * (1.0 - eps) + eps
        t_vec = torch.full((batch_size,), t, device=device)
        t_b = t_vec.view(batch_size, *([1] * (z.ndim - 1)))
        use_constraint = i >= N // 2
        if i == N // 2:
            lam_dual = torch.tensor(0.0, device=device)

        if L_N is not None and gradient_steps > 0:
            for _ in range(gradient_steps):
                inputs = z.detach().clone().requires_grad_(True)
                vt = dynamics(inputs, t_vec * 999.0)
                loss = L_N(inputs + vt * (1.0 - t_b))
                if loss.ndim > 0:
                    loss = loss.mean()
                (g,) = torch.autograd.grad(loss, inputs, create_graph=False)

                z = z - lr * g.detach()

        if use_constraint:
            x_current = z.detach().clone()
            inputs = z.detach().clone().requires_grad_(True)
            for _ in range(al_steps):
                objective = V_AL(inputs, x_current, lam_dual, rho)
                (grad,) = torch.autograd.grad(
                    objective,
                    inputs,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )
                inputs = (inputs + lr_al * grad).detach().requires_grad_(True)
                del grad

            with torch.no_grad():
                d_cur = lpips_model(inputs, x_ref).view(inputs.shape[0])
                g_cur = float(
                    (d_cur - lpips_threshold + constraint_margin).mean().item()
                )

            lam_dual = torch.clamp(lam_dual + rho * g_cur, min=0.0)
            if g_cur > 0.02:
                rho = rho * max(1.0, rho_mult)

            z = inputs.detach().clone()

        with torch.no_grad():
            vt = dynamics(z, t_vec * 999.0).detach()
            z = z + vt * dt

        traj.append(z.clone())

    return traj


def image_edit(config, models, image_path, text_prompt, output_dir):

    # models
    model_fn = models["model_fn"]
    clip_loss_1 = models["clip_loss_1"]
    lpips_model = models["lpips_model"]

    N = 100

    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    prompt_suffix = sanitize_prompt_for_filename(text_prompt)
    opt_img_path = os.path.join(output_dir, f"{img_name}_{prompt_suffix}.png")
    original_img_path = os.path.join(output_dir, f"{img_name}.png")
    image = get_img(image_path)
    original_img = image.to(config.device)

    clip_loss_1.set_ref_img(original_img)
    clip_loss_1.set_text(text_prompt)

    scaled_img = scaler(original_img)
    save_img(inverse_scaler(scaled_img), path=original_img_path)

    latent = embed_to_latent(model_fn, scaler(original_img))
    traj = generate_traj(model_fn, latent, N=N)

    print(f"Optimization starts: {image_path} -> {opt_img_path}")

    t_start = time.time()

    if FLAGS.method == "oc_flow":
        u_ind = [_ for _ in range(N)]
        u_opt = oc_flow_optimization(
            z0=latent,
            u_ind=u_ind,
            dynamics=model_fn,
            generate_traj=generate_traj,
            L_N=clip_loss_1.L_N,
            x_ref=traj[-1].detach(),
            lpips_model=lpips_model,
            N=N,
            number_of_iterations=25,
            lr=2.5,
            lpips_threshold=0.06,
            constraint_margin=0.02,
        )
        traj_opt = generate_traj(model_fn, z0=latent, u=u_opt, N=N)
    elif FLAGS.method == "gradient_guidance":
        traj_opt = gradient_guidance_optimization(
            z0=latent,
            dynamics=model_fn,
            L_N=clip_loss_1.L_N,
            x_ref=traj[-1].detach(),
            lpips_model=lpips_model,
            N=N,
            gradient_steps=25,
            lr=2.5,
            lpips_threshold=0.06,
            constraint_margin=0.02,
        )
    elif FLAGS.method == "projection_relaxed":
        traj_opt = projection_relaxed_optimization(
            z0=latent,
            dynamics=model_fn,
            L_N=clip_loss_1.L_N,
            x_ref=traj[-1].detach(),
            lpips_model=lpips_model,
            N=N,
            gradient_steps=25,
            lr=2.5,
            lpips_threshold=0.06,
            constraint_margin=0.02,
            lr_al=2.5,
            rho=5.0,
            rho_mult=1.02,
            al_steps=10,
        )
    elif FLAGS.method == "hardflow":
        traj_opt = hardflow_optimization(
            z0=latent,
            dynamics=model_fn,
            L_N=clip_loss_1.L_N,
            x_ref=traj[-1].detach(),
            lpips_model=lpips_model,
            N=N,
            gradient_steps=40,
            dual_update_frequency=8,
            lr=2.5,
            lpips_threshold=0.06,
            constraint_margin=0.02,
            rho=5.0,
            rho_mult=2.0,
            regularization=0.0005,
        )
    else:
        traj_opt = traj

    t_end = time.time()
    total_time = t_end - t_start

    print(f"Optimization ends")

    save_img(inverse_scaler(traj_opt[-1]), path=opt_img_path)

    with torch.no_grad():

        clip_loss_orig = clip_loss_1.L_N(traj[-1]).item()
        lpips_score_orig = lpips_model(traj[-1], traj[-1]).item()
        clip_loss_opt = clip_loss_1.L_N(traj_opt[-1]).item()
        lpips_score_opt = lpips_model(traj_opt[-1], traj[-1]).item()
        print(
            f"clip score (original): {-clip_loss_orig:.4f}, lpips score (original): {lpips_score_orig:.4f}\n"
            f"clip score (optimized): {-clip_loss_opt:.4f}, lpips score (optimized): {lpips_score_opt:.4f}\n"
            f"total time: {total_time:.4f} s"
        )

        metric = {
            "clip_score": float(-clip_loss_opt),
            "lpips_score": float(lpips_score_opt),
            "clip_score_original": float(-clip_loss_orig),
            "lpips_score_original": float(lpips_score_orig),
            "time": float(total_time),
        }

    return metric
