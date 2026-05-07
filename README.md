# Training 
```bash
cd /emotion_detection

uv run python -m scripts.train

uv run python -m scripts.train model=rubert_base

uv run python -m scripts.train data.max_length=512 data.batch_size=16

uv run python -m scripts.train model.epochs=5

uv run python -m scripts.train --config-dir=../configs
```

# Inference
```bash
uv run python -m scripts.infer model=rubert_tiny2

uv run python -m scripts.infer model=rubert_base

uv run python -m scripts.infer model=rubert_tiny2 paths.save_dir=./my_models
```

# Предсказание для одного текста
```bash
uv run python scripts/predict.py --text "Сегодня отличный день!"

uv run python scripts/predict.py --model_path ./models/cointegrated_rubert-tiny2 --text "Мне грустно"
```

# Скачивание данных 

```bash
uv run python scripts/download_data.py

uv run python scripts/download_data.py --target_dir ./my_data
```

# MLFlow-logs
```bash
uv run mlflow ui --backend-store-uri file:./mlflow
```