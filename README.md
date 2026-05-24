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
multi-label-emotion-detection/
├── configs/                          # Hydra configs
│   ├── model/
│   ├── data/
│   ├── train/
│   ├── paths/
│   ├── logger/
│   └── config.yaml
├── data/                             # Dataset
├── emotion_detection/                # Main source code package
│
│   ├── train/                        # Training scripts
│   │   ├── train.py                  # Training entry point
│   │   ├── multilabel_classifier.py  # Lightning model
│   │   ├── focal_loss.py             # Custom loss function
│   │
│   ├── eval/                         # Evaluation scripts
│   │   ├── evaluate.py               # Full model evaluation on test set
│   │   ├── metrics.py                # Custom F1 scores
│   │   ├── plot_metrics.py           # Plotting training curves
│   │
│   ├── infer/                        # Inference scripts
│   │   ├── infer.py                  # Checkpoint inference
│   │   ├── infer_triton.py           # Triton inference client
│   │   ├── triton_client.py          # Triton gRPC/HTTP wrapper
│   │
│   ├── export/                       # Model export tools
│   │   ├── export_onnx.py            # Export HF -> ONNX
│   │   ├── export_tensorrt.sh        # ONNX -> TensorRT engine build
│   │
│   ├── triton_server/                # Triton deployment
│   │   ├── build_triton_repo.py      # Builds model repo
│   │   ├── model_repository/         # Final Triton model repo
│   │   ├── docker-compose.yml        # Triton server setup
│   │   ├── Dockerfile                # Triton image config
│   │
│   ├── models/                       # Saved HF checkpoints
│   │   ├── <model_name>/<timestamp>/
│
│   ├── onnx_models/                  # Exported ONNX models
│   ├── tensorrt_models/              # TensorRT engines
│
│   ├── data/                         # Dataset module
│   │   ├── emotion_datamodule.py     # Lightning DataModule
│   │   ├── download_data.py          # Dataset download script
│
│   ├── utils/                        # Helper utilities
│   │   ├── dvc_pull.py               # DVC dataset/model pull
│   │   ├── model_paths.py            # Path helpers
│   │   ├── hydra_utils.py            # Hydra helpers
│
│   ├── commands.py                   # CLI entrypoint
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore
```

## Setup

### Clone Repository

```bash
git clone https://github.com/avarya13/multi-label-emotion-detection.git
cd multi-label-emotion-detection
```

### Create Environment and Install Dependencies

```bash
uv sync
```

### Quality Check

Install pre-commit hooks:

```bash
uv run pre-commit install
```

Run all code quality checks:

```bash
uv run pre-commit run -a
```

## Train

### Data Download

Dataset and fine-tuned models is managed using DVC.
To download data and models:

```bash
uv run dvc pull
```

The data will be stored in the `data` directory and the models will be in the `emotion_detection/models`, `emotion_detection/onnx_models` and `emotion_detection/tensorrt_models` directories.

If necessary, you can download the dataset directly from Hugging Face:

```bash
cd emotion_detection
uv run python commands.py download
```

The dataset will be stored in the `/data` directory.

### Experiment Tracking

The project uses [MLflow](https://mlflow.org/) for experiment tracking.
Start MLflow UI:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri file:./mlflow
```

### Training

#### Available Models

- `ruroberta_large`
- `rubert_tiny2`
- `rubert_base`
- `rubert-base-cased-conv`

#### Train Model

```bash
cd emotion_detection
uv run python commands.py train model=ruroberta_large
```

During training, hyperparameters, losses, and evaluation metrics are logged to the `/logs` directory. At the end of training, loss and metric graphs are plotted and saved to the `/plots` directory. The `/logs` and `/plots` directories are created automatically during training.

## Model evaluation

```bash
uv run python commands.py eval model=ruroberta_large
```

To reproduce the evaluation procedure used in the original RuIzardEmotions repository, you can enable Softmax activation::

```bash
uv run python commands.py eval model=ruroberta_large model.activation=softmax
```

Metrics obtained during batch inference are saved to the `/inference_results` directory.

## Model export

### Export to ONNX

```bash
uv run python commands.py export onnx model=rubert_tiny2
```

### Export to TensorRT

```bash
uv run python commands.py export trt model=rubert_tiny2
```

## Inference

### Single Text Prediction

#### Prediction using Triton Inference Server

**Triton requirements:**

- Docker
- NVIDIA Container Toolkit
- GPU

```bash
uv run python commands.py prepare-triton
```

Launch Triton Inference Server:

```bash
docker compose -f triton_server/docker-compose.yml up
```

Run inference:

```bash
uv run python commands.py infer triton model=rubert_tiny2 '+text="Сегодня отличный день!"'
```

#### Prediction without Triton Inference Server

```bash
uv run python commands.py infer ckpt model=rubert_tiny2 '+text="Сегодня отличный день!"'
```

---

## Detailed Project Description

### Problem Statement

The project solves the problem of prototyping a service for evaluating the emotional coloring of user messages when communicating with technical support specialists. The aim of the project is to create a prototype system that automatically detects a set of emotions in a text message, taking into account the possibility of simultaneous presence of several emotions (multi-label classification). The prototype under development includes a reproducible pipeline for training and evaluating the model, as well as an inference service based on NVIDIA Triton Inference Server. The model is deployed as a containerized service.

#### Input and Output Data Format

The input to the system is a text message from the user, presented as a string.
Before accessing the model, the input text is processed on the client side using the HuggingFace tokenizer. As a result, a set of fixed-length numeric tensors is formed, which are then transmitted to the Triton Inference Server via the HTTP API in the numpy array format, serialized by the Triton client.

The request to the inference server consists of two parts: a JSON description of the input and output tensors and binary tensor data. The JSON structure of the request looks like this:

```json
{
  "inputs": [
    {
      "name": "input_ids",
      "shape": [1, 128],
      "datatype": "INT64"
    },
    {
      "name": "attention_mask",
      "shape": [1, 128],
      "datatype": "INT64"
    }
  ],
  "outputs": [
    {
      "name": "logits"
    }
  ]
}
```

The Triton server returns a response in the form of binary data (logits for each class) and a JSON header:

```json
{
  "model_name": "cointegrated-rubert-tiny2",
  "model_version": "1",
  "outputs": [
    {
      "name": "logits",
      "shape": [1, 10],
      "datatype": "FP32",
      "parameters": { "binary_data_size": 40 }
    }
  ]
}
```

#### Metrics

The main evaluation metric is **F1-score (macro-averaged)**, which is calculated as the average F1-score across all emotion classes. This metric is especially useful for imbalanced multi-label classification tasks, since it treats all classes equally regardless of their frequency in the dataset.

In addition, **F1-score (micro-averaged)** is used to measure overall performance by aggregating predictions across all classes. This metric reflects the global quality of classification and is more influenced by frequent labels.

To provide a more complete evaluation, precision and recall are also reported. Precision shows how many predicted labels are correct, while recall reflects how many true labels are successfully identified. Both metrics are averaged across all classes to ensure balanced evaluation.

Alongside threshold-based metrics, ranking-based and probability-based metrics are included. **ROC-AUC** (macro-averaged) evaluates how well the model ranks positive labels higher than negative ones across different thresholds, making it independent of a fixed decision boundary. **Ranking loss** measures the proportion of incorrectly ordered label pairs and reflects the quality of label ranking in multi-label classification.

F1-metrics are computed using two approaches:

- Standard implementation ([`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html))
- Custom implementation: additional F1-score calculations follow the evaluation procedure from the original baseline [notebook](https://github.com/Djacon/russian-emotion-detection/blob/main/model/emotion_detection.ipynb) to ensure comparability with prior results.

The target is to achieve performance not lower than the baseline model [Djacon/rubert-tiny2-russian-emotion-detection](https://huggingface.co/Djacon/rubert-tiny2-russian-emotion-detection), which reports:

- F1-micro >= 0.86
- F1-macro >= 0.62

#### Validation

To ensure reproducibility, a fixed split of the dataset proposed by the authors is used.: 80% of the proposals are for training, 10% of the proposals for validation and 10% of the proposals for testing. This allows us to directly compare the results obtained with the previously published results of the authors of the [Djacon/rubert-tiny2-russian-emotion-detection](https://huggingface.co/Djacon/rubert-tiny2-russian-emotion-detection) model.
The complete reproducibility of the experiment is ensured by fixing a random seed for all stages: mixing data, initializing model weights, etc.

#### Data

For prototyping, the RuIzardEmotions public dataset (2023) is used, which is a high-quality translation of the English-language corpus of go-emotions and other sources. The dataset is distributed under the Apache license‑2.0 and is available on [Hugging Face](https://huggingface.co/datasets/Djacon/ru-izard-emotions).
The dataset contains 30,000 comments from Reddit, translated into Russian using the DeepL system with subsequent post-processing. Each comment is labeled in ten categories: _joy, sadness, anger, enthusiasm, surprise, disgust, fear, guilt, shame, neutral_. It is acceptable to have multiple emotions in one example (multi‑label).
The dataset size is 4.06 MB, and the total number of rows is 24,891.
The RuIzardEmotions dataset already contains a fixed breakdown into training, validation, and test samples in the proportions of 24,000, 3,000, and 3,000 examples, respectively.
A key feature of the dataset is a strong class imbalance. In particular, texts corresponding to the _guilty_ and _shame_ classes make up less than 5% of all texts.

### Modeling

#### Baseline

The pipeline is taken directly from the authors of the RuIzardEmotions dataset and consists of a simple sequence: including text tokenization, training, and validation. As part of this pipeline, the pre‑trained [ruBERT-tiny2](https://huggingface.co/cointegrated/rubert-tiny2) transformer model (29.4 million parameters) is configured on a target dataset to classify emotions with multiple labels. During training, the model learns from the preprocessed data, and validation is performed on the prepared test subset. According to the authors, this basic pipeline provides an F1‑macro score of 0.6180 and an F1‑micro score of 0.8642 in the test subset (author's implementations of functions).

#### Main model

[ruRoBERTa-large](https://huggingface.co/ai-forever/ruRoberta-large) (355 million parameters) is chosen as the main model.
This is a Russian-language version of the RoBERTa architecture, which belongs to the class of large transformers with a large number of parameters, which makes it possible to effectively model complex contextual dependencies in the text.
The model was pre-trained on a large corpus of Russian-language texts using the Adam optimizer, linear scheduler, and token masking.

#### Additional Models (for comparison)

[ruBERT-base](https://huggingface.co/ai-forever/ruBert-base) (178 million parameters)
is the basic version of the BERT architecture for the Russian language. The model is trained on a large corpus of Russian-language texts and is used as a standard baseline transformer of medium complexity.

[ruBERT-base-cased-conversational](https://huggingface.co/DeepPavlov/rubert-base-cased-conversational) (180 million parameters) is a modification of ruBERT-base, additionally trained on dialog and social data (OpenSubtitles, Dirty, Pikabu, Taiga). This model is better adapted to informal and colloquial speech, which is important for analyzing user messages.

#### Hyperparameters for Model Fine-tuning

The hyperparameters used for fine-tuning the models are presented in the table below.

| Parameter     | ruBERT-tiny2 (authors' code) | ruBERT-tiny2 (reported by authors) | ruBERT-base      | ruBERT-cased-conv | ruRoBERTa-large  |
| ------------- | ---------------------------- | ---------------------------------- | ---------------- | ----------------- | ---------------- |
| Epochs        | 10                           | 10                                 | 8                | 8                 | 10               |
| Batch size    | 64                           | 32                                 | 32               | 32                | 32               |
| Learning rate | $1\times10^{-4}$             | $5\times10^{-5}$                   | $2\times10^{-5}$ | $2\times10^{-5}$  | $1\times10^{-5}$ |
| Loss          | BCE                          | BCE                                | BCE              | BCE               | Focal Loss       |
| Scheduler     | LambdaLR                     | LambdaLR                           | Linear + Warmup  | Linear + Warmup   | Linear + Warmup  |

All models were fine-tuned for the multi-label emotion classification task using a classification head with 10 output neurons corresponding to the target emotion classes.

Binary Cross-Entropy (BCE) loss was used for most experiments. For the main model, `ruRoBERTa-large`, Focal Loss ($\gamma = 2.0$) was additionally used to reduce the effect of class imbalance and improve the quality on rare emotion classes.

The AdamW optimizer was used for all experiments. For the baseline models, the learning rate was decreased after each epoch using the LambdaLR scheduler with an exponential decay factor of 0.5. For the remaining models, a linear warmup scheduler followed by linear learning rate decay was applied.

No weight decay was used for `ruBERT-base` and `ruBERT-base-cased-conversational`, while a weight decay value of 0.01 was used for the remaining models.

All experiments were performed on an NVIDIA Titan RTX GPU. To ensure reproducibility, the random seed was fixed to 42.

The authors of the original `ruBERT-tiny2` baseline report different training settings and evaluation results in the repository source code and in the published [README](https://github.com/Djacon/russian-emotion-detection) / [Hugging Face](https://huggingface.co/datasets/Djacon/ru-izard-emotions) card. Because of these inconsistencies, two versions of the `ruBERT-tiny2` baseline were fine-tuned using two different sets of hyperparameters: one corresponding to the repository implementation and another corresponding to the published description.

#### Training and Inference Pipeline

**1. Preprocessing**
Input text messages are tokenized using Hugging Face tokenizers corresponding to each model.
All token sequences are padded or truncated to a fixed length of 128 tokens.
The tokenizer outputs `input_ids` and `attention_mask` tensors, which are used as model inputs.

**2. Fine-tuning**
Model fine-tuning is performed using the Hugging Face Transformers library.
The models are trained for the multi-label emotion classification task using the hyperparameters described above.
Validation is performed after each training epoch using the validation subset.

**3. Postprocessing**
The model outputs logits of size `[batch_size, num_labels]`.
These logits are converted into probabilities using an activation function:

- `sigmoid` for multi-label classification experiments
- `softmax` for comparative experiments (used by dataset authors)

The resulting probabilities are then used for metric computation and final predictions.

**4. Model Saving**
Fine-tuned models are saved in the native Hugging Face Transformers format, including:

- model weights
- configuration files
- tokenizer files

**5. Export to ONNX and TensorRT**
After fine-tuning, models are first saved as Hugging Face checkpoints.
The checkpoints are then exported to the ONNX format, which serves as an intermediate representation for deployment.
Next, TensorRT engines are generated from the ONNX models to optimize inference performance on NVIDIA GPUs.
The resulting TensorRT models are organized in a Triton-compatible model repository structure.

**6. Inference using NVIDIA Triton Inference Server**
Inference is performed through NVIDIA Triton Inference Server deployed in a Docker container.

### Deployment

The final model is deployed as a containerized solution based on NVIDIA Triton Inference Server. The trained model is pre-exported to ONNX format and placed in the Triton Model Repository.

The inference pipeline consists of three stages: preprocessing, inference, and postprocessing. At the preprocessing stage, the input text message is converted into fixed-length numeric tensors using the HuggingFace tokenizer. Based on the received tensors, a request is generated to the Triton Inference Server via the HTTP API with the transmission of the input and output tensors of the model.

At the inference stage, Triton runs the loaded model on the GPU and returns the output logits for each emotion class via the HTTP API. Post-processing is performed on the side of the client inference script and includes the application of the activation function (sigmoid or softmax) and the conversion of logits into probabilities.

The result of the system is a JSON object generated on the client side, which contains the source text, the name of the model, the selected activation function and a dictionary of emotions with the corresponding probabilities.

---

## Results

### Results with SoftMax activation

| Metric             | ruBERT-tiny2 (authors' code) | ruBERT-tiny2 (reported by authors) | ruBERT-base | ruBERT-cased-conv | ruRoBERTa-large |
| ------------------ | ---------------------------- | ---------------------------------- | ----------- | ----------------- | --------------- |
| F1-macro (torch)   | 0.3343                       | 0.3408                             | 0.3991      | 0.3984            | **0.4103**      |
| F1-micro (torch)   | 0.4464                       | 0.4518                             | 0.4931      | 0.4876            | **0.5022**      |
| F1-macro (authors) | 0.6254                       | 0.6277                             | 0.6578      | 0.6570            | **0.6643**      |
| F1-micro (authors) | 0.8628                       | 0.8606                             | 0.8651      | 0.8638            | **0.8682**      |
| Precision          | 0.4867                       | 0.4683                             | 0.6254      | 0.6741            | **0.7110**      |
| Recall             | 0.2607                       | 0.2725                             | 0.3178      | 0.3188            | **0.3288**      |
| ROC-AUC            | **0.8009**                   | 0.7979                             | 0.7792      | 0.7732            | 0.7815          |
| Label Ranking Loss | **0.1357**                   | 0.1451                             | 0.1464      | 0.1536            | 0.1450          |

### Results with Sigmoid activation

| Metric             | ruBERT-tiny2 (authors' code) | ruBERT-tiny2 (reported by authors) | ruBERT-base | ruBERT-cased-conv | ruRoBERTa-large |
| ------------------ | ---------------------------- | ---------------------------------- | ----------- | ----------------- | --------------- |
| F1-macro (torch)   | 0.4590                       | 0.4708                             | 0.5150      | 0.5087            | **0.5180**      |
| F1-micro (torch)   | 0.5381                       | 0.5416                             | 0.5707      | 0.5624            | **0.5732**      |
| F1-macro (authors) | 0.6862                       | 0.6889                             | 0.7071      | 0.7036            | **0.7134**      |
| F1-micro (authors) | **0.8641**                   | 0.8564                             | 0.8510      | 0.8489            | 0.8619          |
| Precision          | 0.5686                       | 0.5193                             | 0.5365      | 0.5295            | **0.5784**      |
| Recall             | 0.4024                       | 0.4383                             | **0.5108**  | 0.5060            | 0.4847          |
| ROC-AUC            | **0.8088**                   | 0.8033                             | 0.7886      | 0.7883            | 0.7869          |
| Label Ranking Loss | **0.1357**                   | 0.1451                             | 0.1464      | 0.1536            | 0.1450          |

**Notes:**

1. For F1-scores, two evaluation procedures are reported:

- `torch` — metrics computed directly with PyTorch/TorchMetrics
- `authors` — metrics computed using the official evaluation script from the original repository of the dataset's authors

2. For Label Ranking Loss, lower value is better.
