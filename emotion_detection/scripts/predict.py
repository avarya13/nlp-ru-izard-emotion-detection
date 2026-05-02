import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from omegaconf import OmegaConf

cfg_labels = OmegaConf.load("../../configs/data/labels.yaml")
LABELS = cfg_labels.labels
LABELS_RU = cfg_labels.labels_ru

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def predict_emotion(text, tokenizer, model, labels=LABELS):
    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze(0)
    # multilabel: return probabilities for each class
    return {labels[i]: probs[i].item() for i in range(len(labels))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=f"{cfg.paths.save_dir}/{cfg.model.model_name.replace('/', '_')}")
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)
    emotions = predict_emotion(args.text, tokenizer, model)
    # sorting by decreasing
    for label, prob in sorted(emotions.items(), key=lambda x: -x[1]):
        print(f"{label.title().ljust(12)}: {prob:.4f}")

if __name__ == "__main__":
    main()