from itertools import cycle
import os
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import tqdm

from hardflow.datasets.burgers import BurgersDataset
from hardflow.models_flow.flow_matcher import FlowMatcher
from hardflow.models_flow.unet import TemporalUnet
from run.utils import deterministic


def train(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    deterministic(args.seed)

    log_subfolder = os.path.join(args.log_folder, "burgers", "flow", args.exp_name)
    os.makedirs(log_subfolder, exist_ok=True)

    dataset = BurgersDataset(split="train", root_path=args.data_path)

    train_loader = cycle(
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    )

    unet = TemporalUnet(
        output_channels=2,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        n_groups=1,
    ).to(device)

    flow_matcher = FlowMatcher(model=unet, flow_matching_type="cfm")

    optimizer = torch.optim.Adam(
        unet.parameters(), lr=args.learning_rate, betas=(0.9, 0.99)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10000, eta_min=0
    )
    ema = torch.optim.swa_utils.AveragedModel(
        unet,
        avg_fn=lambda avg, new, num: args.ema_decay * avg + (1 - args.ema_decay) * new,
    )

    writer = SummaryWriter(log_dir=os.path.join(log_subfolder, "tensorboard_logs"))

    for i in tqdm.tqdm(range(args.n_train_steps)):
        batch = next(train_loader).to(device)  # (B, 2, 16, 128)

        if args.use_conditioning:
            u0 = batch[:, 0, 0, :].clone()  # initial condition
            uT = batch[:, 0, 10, :].clone()  # final condition
            loss, infos = flow_matcher.loss(batch, u0=u0, uT=uT)
        else:
            loss, infos = flow_matcher.loss(batch)

        loss.backward()

        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Update EMA every 10 steps
        if i % 10 == 0:
            ema.update_parameters(unet)

        if i % args.save_freq == 0 and i > 0:
            checkpoint_num = i // args.save_freq
            torch.save(
                unet.state_dict(),
                os.path.join(log_subfolder, f"model_{checkpoint_num}.pth"),
            )
            torch.save(
                ema.module.state_dict(),
                os.path.join(log_subfolder, f"model_ema_{checkpoint_num}.pth"),
            )
            print(f"\nSaved checkpoint {checkpoint_num} at step {i}")
        writer.add_scalar("loss", loss, i)

    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Burgers flow matching model")

    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/burgers",
        help="Path to Burgers datasets directory",
    )
    parser.add_argument(
        "--use_conditioning",
        type=bool,
        default=True,
        help="Whether to condition on u0 and uT",
    )

    parser.add_argument(
        "--n_train_steps", type=int, default=200000, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument("--ema_decay", type=float, default=0.995, help="EMA decay rate")
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--save_freq", type=int, default=5000, help="Checkpoint saving frequency"
    )

    parser.add_argument(
        "--log_folder",
        type=str,
        default="./logs",
        help="Base folder for logs and checkpoints",
    )
    parser.add_argument(
        "--exp_name", type=str, default="burgers", help="Experiment name"
    )

    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train(args)
