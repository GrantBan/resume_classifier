"""
数据集构建
"""
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
from build_datasets import build_hbert_chunks, get_token_counter


config = Config()
_count_tokens = None


def _get_count_tokens():
    global _count_tokens
    if _count_tokens is None:
        _count_tokens = get_token_counter(config.bert_model_name)
    return _count_tokens


def load_raw_data(data_path):
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        header = f.readline()
        if header and header.strip().lower() == "text\tlabel":
            pass
        elif header.strip():
            text, label = header.strip().split("\t", 1)
            data.append((text, int(label)))
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = line.split("\t", 1)
            data.append((text, int(label)))
    return data

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item][0]
        label = self.data[item][1]
        return text, label

# collate_fn: 长简历 -> 多 chunk -> [batch, num_chunks, chunk_size]
def collate_fn(batch):
    texts = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    max_chunks = config.max_chunks
    chunk_size = config.pad_size
    count_tokens = _get_count_tokens()

    flat_chunks = []
    chunk_masks = []
    for text in texts:
        chunks = build_hbert_chunks(
            text,
            count_tokens,
            config.max_tokens_per_chunk,
            max_chunks,
        )
        if not chunks:
            chunks = [""]
        num_real = min(len(chunks), max_chunks)
        chunks = chunks[:max_chunks]
        while len(chunks) < max_chunks:
            chunks.append("")
        flat_chunks.extend(chunks)
        chunk_masks.append(
            [1] * num_real + [0] * (max_chunks - num_real)
        )

    token_encode = config.tokenizer(
        flat_chunks,
        padding="max_length",
        max_length=chunk_size,
        truncation=True,
    )
    batch_size = len(texts)
    input_ids = torch.tensor(token_encode["input_ids"]).view(
        batch_size, max_chunks, chunk_size
    )
    attention_mask = torch.tensor(token_encode["attention_mask"]).view(
        batch_size, max_chunks, chunk_size
    )
    chunk_mask = torch.tensor(chunk_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, chunk_mask, labels




def build_dataloader():
    train_data = load_raw_data(config.train_data_path)
    test_data = load_raw_data(config.test_data_path)

    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    train_loader, test_loader = build_dataloader()
    batch = next(iter(train_loader))
    input_ids, attention_mask, chunk_mask, labels = batch
    print(f"input_ids: {input_ids.shape}")
    print(f"attention_mask: {attention_mask.shape}")
    print(f"chunk_mask: {chunk_mask.shape}")
    print(f"labels: {labels.shape}")
    print(f"chunk_mask sample: {chunk_mask[0].tolist()}")