# Transformer-based multi-label emotion detection in Russian messages

This project addresses multi‑label emotion recognition in Russian‑language texts. 

The **RuIzardEmotions** dataset — comprising 30,000 Reddit comments annotated with ten emotion categories — is used for training and evaluation. The baseline `ruBERT‑tiny2` model is reproduced from the [authors' repository](https://github.com/Djacon/russian-emotion-detection), and several enhancements are explored. In particular, we replace the SoftMax activation with Sigmoid at the output layer, which is more suitable for multi‑label classification.

Three pre‑trained Russian language models — `ruBERT‑base`, `ruBERT‑base‑cased‑conversational`, and `ruRoBERTa‑large` — are fine‑tuned and compared. The best performance is achieved by `ruRoBERTa‑large`, reaching an F1‑macro score of 0.7134 and an F1‑micro score of 0.8619 (authors’ implementation). 

## Models & Dataset

- **Models** (Hugging Face):
  - [cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2)
  - [ai-forever/ruBert-base](https://huggingface.co/ai-forever/ruBert-base)
  - [DeepPavlov/rubert-base-cased-conversational](https://huggingface.co/DeepPavlov/rubert-base-cased-conversational)
  - [ai-forever/ruRoberta-large](https://huggingface.co/ai-forever/ruRoberta-large)
- **Dataset**: [RuIzardEmotions](https://huggingface.co/datasets/Djacon/ru-izard-emotions) 

## Overall

Project structure:
```
mlops-ru-izard-emotion-detection/
│
├── configs/                 # Hydra configuration files
│   ├── model/               # Model configurations
│   ├── train/               # Training configurations
│   ├── logger/              # MLflow logging configuration
│   └── paths/               # Project paths
│
├── emotion_detection/
│   ├── scripts/
│   │   ├── download_data.py # Dataset download
│   │   ├── train.py         # Model training
│   │   ├── infer.py         # Batch inference
│   │   └── predict.py       # Single-text prediction
│   │
│   └── src/
│       └── utils/
│           ├── focal_loss.py
│           └── metrics.py
│
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

### 1. Clone Repository
```bash
git clone <repository_url>
cd mlops-ru-izard-emotion-detection/emotion_detection
```

### 2. Install uv

#### Linux / macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Create Environment and Install Dependencies
```bash
uv sync
```

## Train

### Data Download

Download the dataset:
```bash
uv run python scripts/download_data.py
```

The dataset will be stored in the `emotion_detection/data` directory.

### Training

#### Available Models
- `ruroberta_large`
- `rubert_tiny2`
- `rubert_base`
- `rubert-base-cased-conv`.  

#### Train Model

```bash
uv run python -m scripts.train model=ruroberta_large
```

#### Experiment Tracking
The project uses [https://mlflow.org/](MLflow) for experiment tracking.
Start MLflow UI::
```bash
uv run mlflow ui --backend-store-uri file:./mlflow
```

Open the link:
```
http://127.0.0.1:5000
```

## Inference
### Batch Inferenc 
```bash
uv run python -m scripts.infer model=ruroberta_large
```

### Single Text Prediction

Example:
```bash
uv run python scripts/predict.py --model_path ./models/ai-forever_ruRoberta-large --text "Сегодня отличный день!"
```

## Results
### Results with SoftMax activation

| Metric | ruBERT-tiny2 (authors' code) | ruBERT-tiny2 (authors' description) | ruBERT-base | ruBERT-cased-conv | ruRoBERTa-large |
|--------|------------------------------|--------------------------------------|-------------|-------------------|-----------------|
| F1-macro (torch) | 0.3343 | 0.3408 | 0.3991 | 0.3984 | **0.4103** |
| F1-micro (torch) | 0.4464 | 0.4518 | 0.4931 | 0.4876 | **0.5022** |
| F1-macro (authors) | 0.6254 | 0.6277 | 0.6578 | 0.6570 | **0.6643** |
| F1-micro (authors) | 0.8628 | 0.8606 | 0.8651 | 0.8638 | **0.8682** |
| Precision | 0.4867 | 0.4683 | 0.6254 | 0.6741 | **0.7110** |
| Recall | 0.2607 | 0.2725 | 0.3178 | 0.3188 | **0.3288** |

### Results with Sigmoid activation

| Metric | ruBERT-tiny2 (authors' code) | ruBERT-tiny2 (authors' description) | ruBERT-base | ruBERT-cased-conv | ruRoBERTa-large |
|--------|----------------|------------------------|-------------|-------------------|-----------------|
| F1-macro (torch) | 0.4590 | 0.4708 | 0.5150 | 0.5087 | **0.5180** |
| F1-micro (torch) | 0.5381 | 0.5416 | 0.5707 | 0.5624 | **0.5732** |
| F1-macro (authors) | 0.6862 | 0.6889 | 0.7071 | 0.7036 | **0.7134** |
| F1-micro (authors) | **0.8641** | 0.8564 | 0.8510 | 0.8489 | 0.8619 |
| Precision | 0.5686 | 0.5193 | 0.5365 | 0.5295 | **0.5784** |
| Recall | 0.4024 | 0.4383 | **0.5108** | 0.5060 | 0.4847 |