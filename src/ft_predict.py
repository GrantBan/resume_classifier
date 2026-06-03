# -*- coding: utf-8 -*-
import glob
import os

import fasttext

from config import Config
from text_preprocess import clean_text


def latest_model_path(model_dir):
    pattern = os.path.join(model_dir, "fastText_resume_clean_*.bin")
    model_files = glob.glob(pattern)
    if not model_files:
        raise FileNotFoundError(f"No trained model found: {pattern}")
    return max(model_files, key=os.path.getmtime)


config = Config()
model = fasttext.load_model(latest_model_path(config.model_save_path))


def normalize_prob(prob):
    return float(min(max(prob, 0.0), 1.0))


def predict(data, top_k=3):
    text = clean_text(data["text"])
    labels, probs = model.predict(text, k=top_k)
    data["pred_class"] = labels[0].replace("__label__", "")
    data["pred_prob"] = normalize_prob(probs[0])
    data["top_k"] = [
        {"label": label.replace("__label__", ""), "prob": normalize_prob(prob)}
        for label, prob in zip(labels, probs)
    ]
    return data


if __name__ == "__main__":
    sample = {
        "text": "Responsible for HR recruitment, training, employee relations and office admin work."
    }
    print(predict(sample))
