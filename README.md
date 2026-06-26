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

## Text-Guided Image Editing (Section VII.D)

This branch (`image`) reproduces the text-guided image editing experiments
from the paper. Given an input face image and a text prompt
(e.g., "A photo of an old face."), the goal is to produce an edit whose CLIP
score with the prompt is high while the LPIPS distance to the input
image stays below 0.06 (hard constraint).

This branch uses a separate codebase from the other tasks. The
corresponding conda environment is `hardflow_image` (the other three
branches use `hardflow`).

## Branches in this repo

The four experiments from the paper each live on a separate branch. Switch
with `git checkout <branch>` and follow that branch's README.

| Branch    | Section in paper | Task                                    |
|-----------|------------------|-----------------------------------------|
| `d3il`    | VII.A            | Robotic Manipulation                    |
| `maze2d`  | VII.B            | Maze Navigation                         |
| `burgers` | VII.C            | PDE Control                             |
| `image`   | VII.D            | Text-Guided Image Editing (this branch) |

## Algorithms (`--method`)

Section VII.D evaluates four methods on this task:

| `--method`           | Name in the paper                      |
|----------------------|----------------------------------------|
| `gradient_guidance`  | Gradient Guidance                      |
| `oc_flow`            | OC-Flow                                |
| `projection_relaxed` | Projection-Relaxed + Gradient Guidance |
| `hardflow`           | **HardFlow**                           |

(The "Original" baseline is omitted: with no editing it trivially satisfies
the LPIPS constraint by returning the input unchanged, and the CLIP score is
not meaningful.)

## Setup

```bash
conda env create -f environment.yml
conda activate hardflow_image
pip install -r requirements.txt
```

Place the pretrained flow-matching checkpoint at `./pretrained.pth`.
An example model is available
[here](https://drive.google.com/file/d/1YqTinyDGfwslFYKNui6wJnCkcmlpCO3a/view?usp=sharing).
Place the images to be edited in `./experiment/`.

## Run

```bash
# Quick test on the three reference images in demo/:
python main.py --folder demo --method hardflow

# Run a single method on the full inputs in experiment/:
python main.py --folder experiment --method hardflow

# Or run all four methods sequentially:
bash run.sh
```

Each call writes one `metrics.json` per (image, prompt) pair to
`output/<image_id>/<method>/`, together with the edited images. To
aggregate results across methods and prompts, run `python visualization.py`.

## Acknowledgement

The image-editing codebase on this branch is adapted from
[Guided-Flow-Matching-with-Optimal-Control](https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control).

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
