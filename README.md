# radiology-clip-mini

Minimal CLIP-style baseline on **Open-I (IU X-Ray)** (~50 paired cases) for image↔report retrieval with a single activation heatmap (“Grad-CAM-style”) example.

## TL;DR

* **Image encoder:** ResNet-18 (ImageNet) → MLP head → **256-D**
* **Text encoder:** DistilBERT → MLP head → **256-D**
* **Loss:** symmetric InfoNCE with learnable temperature τ
* **Eval:** Recall@1/5/10, Median Rank, nDCG@10
* **Hardware:** GTX-1650 or Kaggle T4; RAM ≥ 8 GB


## Repo layout

```
radiology-clip-mini/
├─ configs/    # YAML configs (explicit hyper-params)
├─ notebooks/  # thin wrappers calling library code
├─ results/   # run outputs (metrics, ckpts, figs) — git-ignored
├─ src/rclip/   # library code (data, models, train, eval, viz)
└─ scripts/   # small utilities
```

## Quickstart (local, Bash)

```bash
git clone https://github.com/MahshadSa/radiology-clip-mini.git
cd radiology-clip-mini
python -m venv .venv && source .venv/bin/activate   # Windows below
pip install -r requirements.txt

# Train (writes results/<YYYYMMDD-HHMMSS>/ and updates results/latest.txt)
python -m rclip.train --config configs/tiny.yaml

# Evaluate + visualise using the pointer above
python -m rclip.eval --ckpt "$(cat results/latest.txt)"
python -m rclip.viz  --ckpt "$(cat results/latest.txt)"
```

**Windows (PowerShell)**

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m rclip.train --config configs/tiny.yaml
$ckpt = Get-Content results/latest.txt
python -m rclip.eval --ckpt "$ckpt"
python -m rclip.viz  --ckpt "$ckpt"
```

**Kaggle Notebook (Bash cells)**

```bash
git clone https://github.com/MahshadSa/radiology-clip-mini.git
cd radiology-clip-mini
pip install -r requirements.txt
python -m rclip.train --config configs/tiny.yaml
CKPT=$(cat results/latest.txt)
python -m rclip.eval --ckpt "$CKPT"
python -m rclip.viz  --ckpt "$CKPT"
```

## Data

* **Source:** Open-I (IU X-Ray) via 🤗 Datasets: `"ykumards/open-i"`.
* **Text priority:** findings → impression → report → text (case-insensitive, cleaned).
* **Split policy:** **patient-level** (prevents leakage).
* **Subset:** ~50 patients for speed (configurable in `configs/tiny.yaml`).

## Configuration (example)

```yaml
seed: 42
data:
  hf_dataset: ykumards/open-i
  max_patients: 50
  val_frac: 0.1
  test_frac: 0.1
  cache_dir: data/hf
train:
  epochs: 8
  batch_size: 16
  lr: 1.0e-3
  weight_decay: 1.0e-4
  grad_clip: 1.0
  amp: true
model:
  embed_dim: 256
  text_encoder: distilbert-base-uncased
  temperature_init: 0.07
paths:
  results_dir: results
```

## Results (thumbnail)

* *Image→Text:* R@1 ***x.xx***, R@5 ***x.xx***, R@10 ***x.xx***, MedR ***x***, nDCG@10 ***x.xx***
* *Text→Image:* R@1 ***x.xx***, R@5 ***x.xx***, R@10 ***x.xx***, MedR ***x***, nDCG@10 ***x.xx***

![retrieval grid](viz/retrieval_grid.png)

> Notes: short generic phrases (“no acute abnormality”) over-match; activation maps are qualitative, not localisation.

## Repro notes

* Deterministic seeds; `results/<run>/run.json` stores config + git hash.
* Metrics in `results/<run>/metrics.json`.
* Figures in `viz/` (e.g., `retrieval_grid.png`, `gradcam_example.png`).

## Lessons learned

* Reports are messy. normalising headers/case helped zero-shot retrieval.
* Patient-wise splits matter. file-wise inflated R@k.
* Tokeniser/versions can mismatch.  lock DistilBERT and assert lengths.

## Cite

* Radford et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, 2021.
* Demner-Fushman et al., *Preparing a Collection of Radiology Examinations for Distribution and Retrieval*, 2016 (IU X-Ray).

## License & ethics

MIT. Research/education only; not for clinical use.
