import torch
from torch.utils.data import Dataset
import h5py
import os
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class BurgersDataNormalizer:

    # state normalization parameters
    u_min: float = -2.472626
    u_max: float = 2.365398

    # control normalization parameters
    f_min: float = -7.308461
    f_max: float = 6.910266

    def normalize_u(self, u_data):
        return 2 * (u_data - self.u_min) / (self.u_max - self.u_min) - 1

    def normalize_f(self, f_data):
        return 2 * (f_data - self.f_min) / (self.f_max - self.f_min) - 1

    def denormalize_u(self, u_normalized):
        return (u_normalized + 1) * (self.u_max - self.u_min) / 2 + self.u_min

    def denormalize_f(self, f_normalized):
        return (f_normalized + 1) * (self.f_max - self.f_min) / 2 + self.f_min

    def get_u_range(self):
        return self.u_min, self.u_max

    def get_f_range(self):
        return self.f_min, self.f_max


class BurgersDataset(Dataset):

    def __init__(
        self,
        split: str = "train",
        root_path: str = "datasets/burgers",
        normalizer: Optional[BurgersDataNormalizer] = None,
    ):
        """
        Args:
            split: Data split ('train', 'test')
            root_path: Path to datasets directory
            normalizer: For data scaling
        """
        self.split = split
        self.root_path = root_path

        self.nt_total = 11
        self.nx = 128
        self.pad_size = 16

        self.normalizer = normalizer or BurgersDataNormalizer()

        self._load_data()

    def _load_data(self):
        if self.split == "train":

            train_dir = os.path.join(self.root_path, "train")
            if os.path.exists(train_dir):
                u_list = []
                f_list = []

                for i in range(20):
                    split_file = os.path.join(train_dir, f"train_{i:02d}.h5")
                    if not os.path.exists(split_file):
                        raise FileNotFoundError(f"Missing split file: {split_file}")

                    with h5py.File(split_file, "r") as f:
                        u_list.append(torch.tensor(f["u"][:], dtype=torch.float32))
                        f_list.append(torch.tensor(f["f"][:], dtype=torch.float32))

                self.u_data = torch.cat(u_list, dim=0)
                self.f_data = torch.cat(f_list, dim=0)
                print(f"Loaded {len(self.u_data)} training samples from 20 split files")
            else:
                raise FileNotFoundError(f"Training data not found at {train_dir}")
        else:
            file_path = os.path.join(self.root_path, f"{self.split}.h5")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset not found at {file_path}")

            with h5py.File(file_path, "r") as f:
                # (N, 11, 128) for u, (N, 10, 128) for f
                self.u_data = torch.tensor(
                    f[self.split]["pde_11-128"][:], dtype=torch.float32
                )
                self.f_data = torch.tensor(
                    f[self.split]["pde_11-128_f"][:], dtype=torch.float32
                )

    def __len__(self):
        return len(self.u_data)

    def __getitem__(self, idx):
        """
        Returns:
            Tensor of shape (2, 16, 128) containing [u, f] channels,
            normalized and padded.
        """
        u = self.u_data[idx]  # (11, 128)
        f = self.f_data[idx]  # (10, 128)

        # Apply normalization
        u_normalized = self.normalizer.normalize_u(u)
        f_normalized = self.normalizer.normalize_f(f)

        # Pad temporal dimension to 16
        # u: 11 -> 16 (pad 5 at end)
        # f: 10 -> 16 (pad 6 at end)
        u_padded = F.pad(
            u_normalized, (0, 0, 0, self.pad_size - self.nt_total), "constant", 0
        )
        f_padded = F.pad(
            f_normalized, (0, 0, 0, self.pad_size - self.nt_total + 1), "constant", 0
        )

        data = torch.stack([u_padded, f_padded], dim=0)  # (2, 16, 128)

        return data

    def get_condition(self, idx):
        """
        Returns:
            Dict with 'u0' and 'uT' tensors of shape (128,), normalized.
        """
        u = self.u_data[idx]  # (11, 128)
        u0_normalized = self.normalizer.normalize_u(u[0, :])
        ut_normalized = self.normalizer.normalize_u(u[-1, :])
        return {"u0": u0_normalized, "uT": ut_normalized}
