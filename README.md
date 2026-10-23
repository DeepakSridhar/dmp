# Diffusion Meta-Prompting (DMP)

Minimal reference code for **"Diffusion Meta-Prompting and Steering for Generalizable Foundation
Model Adaptation"** ([arXiv:2610.11067](https://arxiv.org/abs/2610.11067)).

DMP treats learned prompts as samples from a distribution. A diffusion model is trained on a
*repository* of previously learned prompts, conditioned on a text description of each task, and then
synthesizes prompts for new tasks in one shot, without the original task data or task losses.

This repository contains the classification pipeline (DMPClass / DMPCoOp, Sec. 4.1):

| paper | code |
| --- | --- |
| prompt autoencoder (2048 -> 8x16 latent) | `dmp/autoencoder.py`, `scripts/train_autoencoder.py` |
| DMP training, Eq. 1-2 | `dmp/diffusion.py`, `scripts/train_dmp.py` |
| DDIM sampling, Eq. 3; composition / negation, Eq. 5 | `dmp/sampler.py`, `scripts/sample.py` |
| anchor selection, Eq. 6; test-time steering, Eq. 7 | `scripts/select_anchors.py`, `scripts/sample.py --anchors` |
| class-name conditioning (one CLIP token per class) | `dmp/data.py`, `scripts/encode_classnames.py` |

Evaluating a sampled prompt on a dataset is done with the
prompt-learning codebase the repository came from (e.g. [CoOp](https://github.com/KaiyangZhou/CoOp)).

## Install

```bash
pip install -r requirements.txt
```

## Smoke test

Runs the whole pipeline on a synthetic repository (toy data -> autoencoder -> DMP -> anchors ->
plain / steered / composite / composed sampling) and checks the outputs.

```bash
bash scripts/smoke_test.sh
```

## Using your own prompt repository

**1. Repository file** - a `torch.save`d dict:

```python
{"prompts": FloatTensor[N, K, d],          # one learned prompt per (task, seed)
 "tasks": [str] * N, "seeds": [int] * N,   # task id and prompt-learning seed of every prompt
 "task_classnames": {task: [class names]}} # the text description of each task
```

**2. Text encodings** of every class name (plus `""`, the unconditional embedding):

```bash
python scripts/encode_classnames.py --classnames classnames.json --out text_encodings.pt
```

**3. Train** the autoencoder and the DMP (`configs/dmp_coop.yaml` holds the paper's architecture for
CoOp prompts with K=4, d=512):

```bash
python scripts/train_autoencoder.py --config configs/dmp_coop.yaml --repository repository.pt --out runs/coop/autoencoder.pt
python scripts/train_dmp.py --config configs/dmp_coop.yaml --repository repository.pt \
    --text-encodings text_encodings.pt --autoencoder runs/coop/autoencoder.pt --out runs/coop/dmp.pt
```

Omit `--autoencoder` to model short prompts directly in prompt space (set the UNet `in_channels` /
`image_size` to K / d).

To give every task its own autoencoder, train one per task with `--tasks` and pass them as `task=path`
entries; each prompt is then encoded with the autoencoder of its task:

```bash
for T in eurosat oxford_flowers ...; do
    python scripts/train_autoencoder.py --config configs/dmp_coop.yaml --repository repository.pt \
        --tasks $T --out runs/coop_per_task/ae_$T.pt
done
python scripts/train_dmp.py --config configs/dmp_coop.yaml --repository repository.pt \
    --text-encodings text_encodings.pt --autoencoder eurosat=runs/coop_per_task/ae_eurosat.pt ... \
    --out runs/coop_per_task/dmp.pt
```

At sampling time pass the same `task=path` list to `--autoencoder`. A sample is decoded with the
autoencoder of `--decode-task` (default: the first of `--tasks`), and the steering anchor of a
composite task is the average of the task anchors, each encoded with its own autoencoder.

**4. Pick anchors** from the training accuracy of each repository prompt (CSV `task,seed,train_acc`):

```bash
python scripts/select_anchors.py --repository repository.pt --accuracies train_acc.csv --out runs/coop/anchors.pt
```

**5. Sample**:

```bash
COMMON="--dmp runs/coop/dmp.pt --autoencoder runs/coop/autoencoder.pt --text-encodings text_encodings.pt --repository repository.pt"
# single task with test-time steering (lambda = 0.2)
python scripts/sample.py $COMMON --tasks eurosat --anchors runs/coop/anchors.pt --out eurosat.pt
# composite task: the classes of both datasets in one condition, average anchor
python scripts/sample.py $COMMON --tasks eurosat oxford_flowers --anchors runs/coop/anchors.pt --out pair.pt
# composition / negation with Eq. 5: one guidance weight per task
python scripts/sample.py $COMMON --tasks eurosat oxford_flowers --compose --scale 4.5 -2.0 --out mix.pt
# arbitrary class list
python scripts/sample.py $COMMON --classnames "Forest" "River" "sunflower" --out custom.pt
```

The output holds `prompts` of shape `[n_samples, K, d]`. `--coop-template <prompt_learner/model.pth.tar-50>
--coop-out <path>` additionally writes the first sample as a CoOp checkpoint for evaluation.

## Converting pytorch-lightning checkpoints

```bash
python scripts/convert_checkpoints.py dmp --ckpt dmp.ckpt --config configs/dmp_coop.yaml --scale-factor <s> --out dmp.pt
python scripts/convert_checkpoints.py autoencoder --ckpt autoencoder.ckpt --config configs/dmp_coop.yaml --out autoencoder.pt
python scripts/convert_checkpoints.py repository --prompts ctx_values_dict.pt --classnames classnames.json --out repository.pt
```

## Citation

```bibtex
@article{sridhar2026dmp,
  title   = {Diffusion Meta-Prompting and Steering for Generalizable Foundation Model Adaptation},
  author  = {Sridhar, Deepak and Li, Yi and Bhardwaj, Kartikeya and Liu, Shuangjun and Jing, Taotao and
             Li, Yuan and Zhang, Shuai and Lyu, Jiancheng and Gao, Dashan and Vasconcelos, Nuno},
  journal = {arXiv preprint arXiv:2610.11067},
  year    = {2026}
}
```

## Acknowledgements

The UNet and autoencoder blocks are adapted from
[latent-diffusion](https://github.com/CompVis/latent-diffusion) (MIT, see `LICENSES/`).
