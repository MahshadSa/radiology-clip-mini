# radiology-clip-mini

Tiny, reproducible vision–language pipeline for chest X-rays.
Goal: learn CLIP-style alignment between images and findings text on a ~50-case subsample of IU X-Ray (OpenI) and show retrieval + Grad-CAMs.

# TL;DR

Encoders: ResNet-18 (image) + DistilBERT (text) → 256-D shared space
Loss: symmetric InfoNCE w/ temperature
Eval: Image↔Text Recall@K, Median Rank, nDCG@10
Viz: Retrieval grids + Grad-CAM overlays
Run local or on Kaggle (no manual data uploads; datasets pulls IU X-Ray)

# Dataset

IU X-Ray (OpenI) via 🤗 datasets ("iu_xray").
Text field: findings (fallback: impression).
Subsample: max_samples=50 (patient-level) for fast iteration.

## Structure
radiology-clip-mini/
├─ README.md
├─ requirements.txt
├─ setup.cfg
├─ .gitignore
├─ src/
│  ├─ rclip/__init__.py
│  ├─ data.py              # load/clean IU X-Ray, split, dataloaders
│  ├─ text.py              # report → findings section, cleaning/tokenization
│  ├─ models.py            # image encoder (ViT/ResNet), text encoder (DistilBERT), projection heads
│  ├─ train.py             # contrastive training loop (InfoNCE), logging, checkpoints
│  ├─ eval.py              # Recall@K, median rank, nDCG
│  ├─ viz.py               # Grad-CAM/attention maps, retrieval viz
│  └─ utils.py             # seed, config, small helpers
├─ notebooks/
│  ├─ 00_data_preview.ipynb
│  ├─ 10_train_rclip.ipynb
│  ├─ 20_eval_retrieval.ipynb
│  └─ 30_viz_gradcam.ipynb
├─ configs/
│  └─ tiny.yaml            # batch_size, lr, epochs, encoders, text field choice
├─ data/                   # (empty; Kaggle will download on the fly)
├─ results/                # figures, metrics, checkpoints (gitignored)
└─ scripts/
   ├─ make_splits.py
   └─ export_minimal_weights.py

# Quickstart (local)
git clone https://github.com/<you>/radiology-clip-mini.git
cd radiology-clip-mini
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# train (tiny config)
python -m src.train --config configs/tiny.yaml

# evaluate and visualize (uses latest checkpoint by default)
python -m src.eval  --ckpt results/*/checkpoint.pt
python -m src.viz   --ckpt results/*/checkpoint.pt

# Quickstart (Kaggle Notebook)
!git clone https://github.com/<you>/radiology-clip-mini.git
%cd radiology-clip-mini
!pip install -r requirements.txt

!python -m src.train --config configs/tiny.yaml
!python -m src.eval  --ckpt $(ls -t results/*/checkpoint.pt | head -n1)
!python -m src.viz   --ckpt $(ls -t results/*/checkpoint.pt | head -n1)

# What’s inside (modules)

src/data.py:
   Loads IU X-Ray, extracts (image, findings_text), patient-level splits, transforms.

src/text.py:
   Findings extraction & cleanup; DistilBertTokenizerFast tokenization.

src/models.py:
   ResNet-18 (pretrained) → projection head; DistilBERT → projection head; L2-norm to shared space.

src/train.py:
   Symmetric contrastive loss (image→text & text→image), temperature τ, mixed precision, checkpointing.

src/eval.py:
   Recall@1/5/10, Median Rank, nDCG@10 for both directions; saves metrics.json.

src/viz.py:
   Retrieval grids; Grad-CAM on last conv block of ResNet-18; overlays saved as PNGs.

# Citation / Related

   CLIP: Radford et al., 2021
   IU X-Ray (OpenI): Demner-Fushman et al., 2016