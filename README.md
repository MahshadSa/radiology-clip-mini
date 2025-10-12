# radiology-clip-mini

Minimal CLIP-style baseline on IU X-Ray (≈50 paired cases) for image↔report retrieval and a single Grad-CAM example.

## TL;DR
- **Image encoder:** ResNet-18 (ImageNet) → MLP head → **256-D**
- **Text encoder:** DistilBERT → MLP head → **256-D**
- **Loss:** symmetric InfoNCE with learnable temperature τ
- **Eval:** Recall@1/5/10, Median Rank, nDCG@10
- **Hardware:** GTX-1650 or Kaggle T4; RAM ≥ 8 GB

## Repo layout
radiology-clip-mini/
├─ configs/ # YAML configs (explicit hyper-params)
├─ notebooks/ # thin wrappers calling library code
├─ results/ # run outputs (metrics, ckpts, figs); git-ignored
├─ src/rclip/ # library code (data, models, train, eval, viz)
└─ scripts/ # small utilities 


## Quickstart (local)

git clone https://github.com/MahshadSa/radiology-clip-mini.git
cd radiology-clip-mini
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train (writes results/<YYYYMMDD-HHMMSS>/ and updates results/latest.txt)
python -m rclip.train --config configs/tiny.yaml

# Evaluate + visualize using the pointer above
python -m rclip.eval --ckpt "$(cat results/latest.txt)"
python -m rclip.viz  --ckpt "$(cat results/latest.txt)"



**Quickstart (Kaggle)**

!git clone https://github.com/MahshadSa/radiology-clip-mini.git
%cd radiology-clip-mini
!pip install -r requirements.txt

!python -m rclip.train --config configs/tiny.yaml
!python -m rclip.eval  --ckpt $(cat results/latest.txt)
!python -m rclip.viz   --ckpt $(cat results/latest.txt)


## Data

Dataset: IU X-Ray (OpenI) via datasets (iu_xray).
Text field priority: findings → impression → report → text.
Split: patient-level (avoid leakage).
Uses a ~50-patient subset for speed (adjust in configs/tiny.yaml).

## Configuration
seed: 42
data:
  dataset: iu_xray
  text_field: findings
  max_samples: 50
  val_frac: 0.1
  test_frac: 0.1
train:
  epochs: 8
  batch_size: 16
  lr_image_head: 1.0e-3
  lr_text: 3.0e-5
  temperature: 0.07
model:
  image_encoder: resnet18
  text_encoder: distilbert-base-uncased
  embed_dim: 256
paths:
  results_dir: results

## Repro notes
Deterministic seeds; config + git hash saved to results/<run>/run.json.
Metrics saved to results/<run>/metrics.json.
A small sample of figures lives under results/sample/.

## Cite
Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP), 2021.
Demner-Fushman et al., Preparing a Collection of Radiology Examinations for Distribution and Retrieval, 2016 (IU X-Ray).

## License & ethics

MIT. Research/education only; not for clinical use.
