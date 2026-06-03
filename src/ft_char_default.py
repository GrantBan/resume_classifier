# -*- coding: utf-8 -*-
from collections import Counter
import datetime
import os

import fasttext

from config import Config
from text_preprocess import clean_text


def evaluate_predictions(model, test_file):
    true_counts = Counter()
    pred_counts = Counter()
    correct_counts = Counter()

    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            true_label = parts[0]
            pred_label, _ = model.predict(parts[1], k=1)
            pred_label = pred_label[0]

            true_counts[true_label] += 1
            pred_counts[pred_label] += 1
            if pred_label == true_label:
                correct_counts[true_label] += 1

    class_metrics = []
    for label in sorted(true_counts):
        precision = correct_counts[label] / pred_counts[label] if pred_counts[label] else 0.0
        recall = correct_counts[label] / true_counts[label] if true_counts[label] else 0.0
        class_metrics.append((label, precision, recall, true_counts[label], pred_counts[label]))

    return pred_counts, class_metrics


def train_model(conf):
    return fasttext.train_supervised(
        input=conf.clean_train_data_path,
        lr=1.2,
        dim=80,
        epoch=25,
        minCount=1,
        wordNgrams=2,
        bucket=200000,
        thread=8,
        label="__label__",
        loss="ova",
    )


def main():
    conf = Config()
    model = train_model(conf)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(conf.model_save_path, f"fastText_resume_clean_{current_time}.bin")
    model.save_model(model_path)

    test_text = "Responsible for HR recruitment, training, employee relations and office admin work."
    pred_label, pred_prob = model.predict(clean_text(test_text), k=3)

    print("\n==== Single sample prediction ====")
    print(f"Input: {test_text}")
    for label, prob in zip(pred_label, pred_prob):
        print(f"Prediction: label={label}, prob={prob:.4f}")

    train_res = model.test(conf.clean_train_data_path)
    test_res = model.test(conf.clean_test_data_path)
    pred_counts, class_metrics = evaluate_predictions(model, conf.clean_test_data_path)

    print("\n==== Evaluation ====")
    print(f"Model saved: {model_path}")
    print(f"Train samples: {train_res[0]}, Precision@1: {train_res[1]:.4f}, Recall@1: {train_res[2]:.4f}")
    print(f"Test samples: {test_res[0]}, Precision@1: {test_res[1]:.4f}, Recall@1: {test_res[2]:.4f}")
    print(f"Test prediction distribution: {dict(pred_counts)}")

    print("\n==== Per-class metrics ====")
    print("label\tprecision\trecall\ttrue\tpred")
    for label, precision, recall, true_count, pred_count in class_metrics:
        print(f"{label}\t{precision:.4f}\t{recall:.4f}\t{true_count}\t{pred_count}")

    p3_res = model.test(conf.clean_test_data_path)
    print(f"Top-3 samples: {p3_res[0]}")
    print(f"Precision@3: {p3_res[1]:.4f}")
    print(f"Recall@3 / Top-3 hit rate: {p3_res[2]:.4f}")


if __name__ == "__main__":
    main()
