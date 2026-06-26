<h1 align="center">HardFlow</h1>

<p align="center">
  <b><a href="https://arxiv.org/abs/2511.08425">Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization</a></b>
  <br><br>
  Zeyang Li &nbsp;·&nbsp; Kaveh Alim &nbsp;·&nbsp; Navid Azizan
  <br>
  <i>Massachusetts Institute of Technology</i>
  <br><br>
  <b>IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026</b>
</p>

---

## PDE Control (Section VII.C)

This branch (`burgers`) reproduces the PDE control experiments from the
paper. The task is control of the 1-D viscous Burgers' equation.
A flow-matching model is trained to generate (state, control) trajectories
that satisfy given initial/terminal boundary conditions. At inference,
HardFlow steers the sampler so the state trajectory respects a
hard time-varying bound `|u(t, s)| ≤ 0.8 · (2t² − 2t + 1)` while minimizing
control energy.

## Branches in this repo

The four experiments from the paper each live on a separate branch. Switch
with `git checkout <branch>` and follow that branch's README.

| Branch    | Section in paper | Task                              |
|-----------|------------------|-----------------------------------|
| `d3il`    | VII.A            | Robotic Manipulation              |
| `maze2d`  | VII.B            | Maze Navigation                   |
| `burgers` | VII.C            | PDE Control (this branch)         |
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
"+ Gradient Guidance" combinations, also set `--projection_gradient_steps 20`.

`hardflow` and `hardflow_new` solve the same surrogate problem and produce
matching numerical results; the only difference is how the reference flow is
evaluated between IPOPT solves. The
[`l4casadi`](https://github.com/Tim-Salzmann/l4casadi) bridge was initially
introduced to solve the full-horizon optimal control problem with neural
dynamics. For HardFlow, however, this dependency is unnecessary. Instead,
`hardflow_new` evaluates the flow directly in PyTorch, removing the
`l4casadi` dependency and avoiding the associated overhead.

## Setup

```bash
conda env create -f environment.yml
conda activate hardflow
pip install -r requirements.txt
pip install -e .
# l4casadi (CUDA) is required for hardflow and the projection baselines;
# the hardflow_new version does not need it. Follow
# https://github.com/Tim-Salzmann/l4casadi for the installation.
```

The Burgers training/test data is included at `datasets/burgers/`. The
pretrained flow-matching checkpoint is expected at
`logs/burgers/flow/4e5steps/model_ema_40.pth`. An example model is available
[here](https://drive.google.com/file/d/1eczftOmCMrZm-nj2flfJnBgoIB6loRWf/view?usp=sharing).
To train it from scratch, run:

```bash
bash run_scripts/train.sh
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

Each script writes `results.json` under `logs/burgers/eval/<exp_name>/`. To
aggregate across runs, see `notebooks/collect_results.ipynb`.

## Acknowledgement

The Burgers benchmark setup on this branch is adapted from
[safediffcon](https://github.com/AI4Science-WestlakeU/safediffcon).
Structure is also inspired by
[flow_guidance](https://github.com/AI4Science-WestlakeU/flow_guidance).
Some PyTorch–CasADi bridges are provided by
[l4casadi](https://github.com/Tim-Salzmann/l4casadi).

## Citation

```bibtex
@article{li2025hardflow,
  title={HardFlow: Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization},
  author={Li, Zeyang and Alim, Kaveh and Azizan, Navid},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  publisher={IEEE}
}
```
