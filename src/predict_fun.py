import argparse
import os
import time
from functools import lru_cache
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from build_datasets import build_hbert_chunks, clean_text, get_token_counter
from config import Config
from Hierarchical_BERT import HierarchicalBERT_model


PROJECT_DIR = Path(__file__).resolve().parent.parent
BEST_MODEL_PATH = PROJECT_DIR / "model" / "best_hierarchical_bert.pt"


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


class Predictor:
    def __init__(self):
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {BEST_MODEL_PATH}")

        self.conf = Config()
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=self.conf.device)
        state_dict = get_state_dict(checkpoint)

        if isinstance(checkpoint, dict):
            self.conf.class_list = checkpoint.get("class_list", self.conf.class_list)
            self.conf.bert_model_name = checkpoint.get(
                "bert_model_name",
                self.conf.bert_model_name,
            )
            self.conf.max_chunks = int(checkpoint.get("max_chunks", self.conf.max_chunks))
            self.conf.pad_size = int(checkpoint.get("pad_size", self.conf.pad_size))
            self.conf.pooling_type = checkpoint.get(
                "pooling_type",
                self.conf.pooling_type,
            )

        self.count_tokens = get_token_counter(self.conf.bert_model_name)
        self.model = HierarchicalBERT_model(
            self.conf,
            num_labels=len(self.conf.class_list),
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.conf.device)
        self.model.eval()

    def encode_text(self, text: str):
        text = clean_text(text)
        chunks = build_hbert_chunks(
            text=text,
            count_tokens=self.count_tokens,
            max_tokens=self.conf.max_tokens_per_chunk,
            max_chunks=self.conf.max_chunks,
        )

        if not chunks:
            chunks = [""]

        real_chunks = min(len(chunks), self.conf.max_chunks)
        chunks = chunks[: self.conf.max_chunks]
        chunks += [""] * (self.conf.max_chunks - len(chunks))

        token_encode = self.conf.tokenizer(
            chunks,
            padding="max_length",
            max_length=self.conf.pad_size,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = token_encode["input_ids"].view(
            1,
            self.conf.max_chunks,
            self.conf.pad_size,
        )
        attention_mask = token_encode["attention_mask"].view(
            1,
            self.conf.max_chunks,
            self.conf.pad_size,
        )
        chunk_mask = torch.tensor(
            [[1] * real_chunks + [0] * (self.conf.max_chunks - real_chunks)],
            dtype=torch.long,
        )
        return input_ids, attention_mask, chunk_mask

    def predict_single(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Text is empty")

        start = time.time()
        input_ids, attention_mask, chunk_mask = self.encode_text(text)
        input_ids = input_ids.to(self.conf.device)
        attention_mask = attention_mask.to(self.conf.device)
        chunk_mask = chunk_mask.to(self.conf.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, chunk_mask)
            label = int(torch.argmax(logits, dim=1).item())

        return {
            "label": label,
            "category": self.conf.class_list[label],
            "time_ms": round((time.time() - start) * 1000, 2),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict_single(text) for text in texts]


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


def predict(text: str) -> dict:
    return get_predictor().predict_single(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    result = predict(args.text)
    print(f"Label: {result['label']}")
    print(f"Category: {result['category']}")
    print(f"Time: {result['time_ms']} ms")
