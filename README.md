# HardFlow — Maze Navigation

This branch (`maze2d`) reproduces Section VII.B of [HardFlow: Hard-Constrained
Sampling for Flow-Matching Models via Trajectory Optimization](https://arxiv.org/abs/2511.08425).

The task is closed-loop point-mass navigation in the `maze2d-large-v1`
environment with two extra obstacles. A flow-matching model is trained on
offline trajectories from the dataset; at inference, HardFlow steers the
sampler so the planned trajectory avoids all obstacles and reaches the goal
as fast as possible. A PD controller tracks the planned positions in the
simulation.

## Branches in this repo

The four experiments from the paper each live on a separate branch. Switch
with `git checkout <branch>` and follow that branch's README.

| Branch    | Section in paper | Task                              |
|-----------|------------------|-----------------------------------|
| `d3il`    | VII.A            | Robotic Manipulation              |
| `maze2d`  | VII.B            | Maze Navigation (this branch)     |
| `burgers` | VII.C            | PDE Control                       |
| `image`   | VII.D            | Text-Guided Image Editing         |

## Algorithms (`--guidance_method`)

| Flag                 | Name in the paper            |
|----------------------|------------------------------|
| `original`           | Original                     |
| `gradient_guidance`  | Gradient Guidance            |
| `oc_flow`            | OC-Flow                      |
| `projection`         | Projection-All/Late*         |
| `projection_relaxed` | Projection-Relaxed           |
| `hardflow`           | **HardFlow**                 |
| `hardflow_new`       | **HardFlow (l4casadi-free)** |

\*Use `--projection_option {all,late}`. For
"+ Gradient Guidance" combinations, also set `--projection_gradient_steps 5`.

`hardflow` and `hardflow_new` solve the same surrogate problem and produce
matching numerical results; the only difference is how the reference flow is
evaluated between IPOPT solves. The
[`l4casadi`](https://github.com/Tim-Salzmann/l4casadi) bridge was initially
introduced to solve the full-horizon optimal control problem with neural
dynamics. For HardFlow, however, this dependency is unnecessary. Instead,
`hardflow_new` evaluates the flow directly in PyTorch, removing the
`l4casadi` dependency and avoiding the associated overhead.

## Setup

D4RL needs MuJoCo 2.0:

```bash
# Install MuJoCo 2.0.0 from https://www.roboti.us/download.html
# Place files and key in ~/.mujoco/
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin' >> ~/.bashrc

conda env create -f environment.yml
conda activate hardflow
pip install -r requirements.txt
pip install mujoco-py==2.0.2.8
pip install -e .
# l4casadi (CUDA) is required for hardflow and the projection baselines;
# the hardflow_new version does not need it. Follow
# https://github.com/Tim-Salzmann/l4casadi for the installation.
```

The pretrained flow-matching checkpoint is expected at
`logs/maze2d-large-v1/flow/H384_1e6steps/model_ema_20.pth`. An example model
is available
[here](https://drive.google.com/file/d/1_VXAPbO6pk_UT0GMvxGr518xGUfvsRu-/view?usp=sharing).
To train it from scratch, run:

```bash
bash run_scripts/train.sh
```

A linear dynamics model fitted from the training data is used as a
physical-fidelity constraint at inference time. To fit it, run:

```bash
python run/fit_dynamics.py
```

### Troubleshooting

```bash
# fatal error: GL/osmesa.h: No such file or directory
sudo apt-get install libosmesa6-dev

# version `GLIBCXX_3.4.30' not found
conda install -c conda-forge gcc=12.1.0
```

## Run

```bash
bash run_scripts/eval_original.sh           # Original
bash run_scripts/eval_gradient_guidance.sh  # Gradient Guidance
bash run_scripts/eval_oc_flow.sh            # OC-Flow
bash run_scripts/eval_projection.sh         # Projection-All / Projection-Late
bash run_scripts/eval_projection_relaxed.sh # Projection-Relaxed
bash run_scripts/eval_hardflow.sh           # HardFlow
bash run_scripts/eval_hardflow_new.sh       # HardFlow (l4casadi-free)
```

Each script writes `trajectories.csv` under `logs/maze2d-large-v1/eval/<exp_name>/`.
Use `notebooks/collect_results.ipynb` to aggregate results across runs.

## Acknowledgement

The maze2d benchmark setup on this branch is adapted from
[SafeDiffuser](https://github.com/Weixy21/SafeDiffuser).
Structure is also inspired by
[flow_guidance](https://github.com/AI4Science-WestlakeU/flow_guidance).
Some PyTorch–CasADi bridges are provided by
[l4casadi](https://github.com/Tim-Salzmann/l4casadi).

## Citation

```bibtex
@article{li2025hardflow,
  title={HardFlow: Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization},
  author={Li, Zeyang and Alim, Kaveh and Azizan, Navid},
  journal={arXiv preprint arXiv:2511.08425},
  year={2025}
}
```
