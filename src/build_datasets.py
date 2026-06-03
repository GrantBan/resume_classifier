"""
为 XGBoost、FastText 和 Hierarchical BERT 构建训练/测试数据。

输入文件格式：
    text<TAB>label

输出文件：
    processed/
        xgboost_train.csv
        xgboost_test.csv
        fasttext_train.txt
        fasttext_test.txt
        hbert_train.jsonl
        hbert_test.jsonl
        label2id.json
        id2label.json
        dataset_info.json

使用方式：
    python src/build_datasets.py

    python src/build_datasets.py ^
        --input data/train.txt ^
        --output-dir data/processed ^
        --test-size 0.1 ^
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


SECTION_TITLES = [
    "Summary",
    "Highlights",
    "Skills",
    "Experience",
    "Education",
    "Education and Training",
    "Accomplishments",
    "Certifications",
    "Work History",
    "Professional Experience",
    "Additional Information",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为简历分类模型构建切分后的数据集。"
    )
    parser.add_argument(
        "--input",
        default="data/train.txt",
        help="Input TSV file. Default: data/train.txt",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory. Default: data/processed",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test split ratio. Default: 0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=4,
        help="Hierarchical BERT 每份简历最多保留的 chunk 数，默认 4。",
    )
    parser.add_argument(
        "--max-tokens-per-chunk",
        type=int,
        default=510,
        help="Max content tokens per chunk. Leave room for [CLS]/[SEP]. Default: 510",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="bert-base-uncased",
        help="Tokenizer name used only if transformers is installed. Default: bert-base-uncased",
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Text column name in input file. Default: text",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Label column name in input file. Default: label",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=200,
        help="过滤过短简历的最小字符数，默认 200。",
    )
    return parser.parse_args()


def clean_text(text: object) -> str:
    """清洗简历文本，同时保留有助于句子切分的标点。"""
    text = str(text)
    text = text.strip().strip('"')

    # 修复数据集中常见的乱码字符。
    text = text.replace("锛?", ",")
    text = text.replace("路", " ")

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*", ", ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mark_section_boundaries(text: str) -> str:
    """在简历常见小标题前后添加换行，帮助后续切分。"""
    for title in sorted(SECTION_TITLES, key=len, reverse=True):
        pattern = rf"\b{re.escape(title)}\b"
        text = re.sub(pattern, f"\n{title}\n", text, flags=re.IGNORECASE)
    return text


def split_sentences(text: str) -> list[str]:
    """将简历文本切分成适合建模的语义片段。

    规则：
    - 简历小标题和换行是强边界。
    - . ? ! ; 作为普通句子边界。
    - 逗号不作为常规边界，只在片段过长时作为兜底切分依据。
    """
    text = mark_section_boundaries(text)
    parts = re.split(r"\n+", text)

    units: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # 当句末标点后面接着新的句子时，在标点后切分。
        sentences = re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9])", part)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                units.append(sentence)

    return units


def get_token_counter(tokenizer_name: str) -> Callable[[str], int]:
    """优先使用 tokenizer 统计 token 数，失败时退回到正则计数。

    正则计数只用于构建 chunk 字符串；真正训练时仍然由 Dataset 中的
    tokenizer 完成精确编码。
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def count_tokens(text: str) -> int:
            return len(
                tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                )
            )

        return count_tokens
    except Exception:

        def count_tokens(text: str) -> int:
            return len(re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text))

        return count_tokens


def split_long_unit(
    unit: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> list[str]:
    """切分过长片段，只有在必要时才使用逗号作为兜底边界。"""
    if count_tokens(unit) <= max_tokens:
        return [unit]

    comma_parts = [part.strip() for part in unit.split(",") if part.strip()]
    if len(comma_parts) <= 1:
        return split_by_words(unit, max_tokens=max_tokens)

    packed: list[str] = []
    current: list[str] = []
    current_text = ""

    for part in comma_parts:
        candidate = part if not current_text else f"{current_text}, {part}"
        if count_tokens(candidate) <= max_tokens:
            current.append(part)
            current_text = candidate
        else:
            if current:
                packed.append(", ".join(current))
            if count_tokens(part) > max_tokens:
                packed.extend(split_by_words(part, max_tokens=max_tokens))
                current = []
                current_text = ""
            else:
                current = [part]
                current_text = part

    if current:
        packed.append(", ".join(current))

    return packed


def split_by_words(text: str, max_tokens: int) -> list[str]:
    """当文本没有可用标点时，最后按词数进行兜底切分。"""
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i : i + max_tokens])
        for i in range(0, len(words), max_tokens)
    ]


def build_hbert_chunks(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    max_chunks: int,
) -> list[str]:
    """为 Hierarchical BERT 构建尽量保持句子完整的 chunks。"""
    units = split_sentences(text)
    expanded_units: list[str] = []

    for unit in units:
        expanded_units.extend(split_long_unit(unit, count_tokens, max_tokens))

    chunks: list[str] = []
    current: list[str] = []
    current_text = ""

    for unit in expanded_units:
        candidate = unit if not current_text else f"{current_text} {unit}"
        if count_tokens(candidate) <= max_tokens:
            current.append(unit)
            current_text = candidate
        else:
            if current:
                chunks.append(" ".join(current))
            current = [unit]
            current_text = unit

    if current:
        chunks.append(" ".join(current))

    return chunks[:max_chunks]


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按类别分层切分训练集和测试集。"""
    rng = random.Random(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in df.groupby(label_col):
        indices = list(group.index)
        rng.shuffle(indices)

        test_count = max(1, round(len(indices) * test_size))
        # 尽量保证每个类别至少有一个样本留在训练集中。
        if len(indices) > 1:
            test_count = min(test_count, len(indices) - 1)
        else:
            test_count = 0

        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    return train_df, test_df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """保存适合 XGBoost 等传统模型使用的 CSV 文件。"""
    df.to_csv(path, index=False, encoding="utf-8")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    """保存 text<TAB>label 格式的数据文件。"""
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")


def fasttext_escape(text: str) -> str:
    """将文本中的换行替换为空格，满足 FastText 单行样本格式。"""
    return text.replace("\n", " ").replace("\r", " ").strip()


def write_fasttext(df: pd.DataFrame, path: Path) -> None:
    """保存 FastText supervised 模型需要的 __label__ 前缀格式。"""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in df.itertuples(index=False):
            label = getattr(row, "label")
            text = fasttext_escape(getattr(row, "text"))
            f.write(f"__label__{label} {text}\n")


def write_hbert_jsonl(
    df: pd.DataFrame,
    path: Path,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    max_chunks: int,
) -> dict[str, float]:
    """保存 Hierarchical BERT 使用的 jsonl 数据，并统计 chunk 数量。"""
    chunk_counts: list[int] = []

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row_id, row in enumerate(df.itertuples(index=False)):
            text = getattr(row, "text")
            label = int(getattr(row, "label"))
            chunks = build_hbert_chunks(
                text=text,
                count_tokens=count_tokens,
                max_tokens=max_tokens,
                max_chunks=max_chunks,
            )
            chunk_counts.append(len(chunks))
            item = {
                "id": row_id,
                "text": text,
                "label": label,
                "chunks": chunks,
                "chunk_count": len(chunks),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if not chunk_counts:
        return {"avg_chunks": 0.0, "max_chunks": 0.0}

    return {
        "avg_chunks": sum(chunk_counts) / len(chunk_counts),
        "max_chunks": float(max(chunk_counts)),
    }


def save_json(data: object, path: Path) -> None:
    """以 UTF-8 编码保存 JSON 文件。"""
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_input(
    path: Path,
    text_col: str,
    label_col: str,
    min_text_chars: int,
) -> pd.DataFrame:
    """读取原始 TSV 数据，并统一整理为 text 和 label 两列。"""
    read_kwargs = {
        "sep": "\t",
        "quoting": csv.QUOTE_NONE,
        "encoding": "utf-8",
        "engine": "python",
    }
    df = pd.read_csv(path, **read_kwargs)

    if text_col not in df.columns or label_col not in df.columns:
        # 有些 train.txt 没有表头，pandas 会把第一条简历误认为表头。
        # 这种情况下需要重新读取，并手动指定列名。
        df = pd.read_csv(
            path,
            header=None,
            names=[text_col, label_col],
            **read_kwargs,
        )

    df = df[[text_col, label_col]].rename(
        columns={text_col: "text", label_col: "label"}
    )
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() >= min_text_chars].reset_index(drop=True)
    return df


def main() -> None:
    """命令行入口：读取原始数据、切分数据集并生成各模型所需文件。"""
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_input(
        input_path,
        args.text_col,
        args.label_col,
        args.min_text_chars,
    )
    train_df, test_df = stratified_split(
        df=df,
        label_col="label",
        test_size=args.test_size,
        seed=args.seed,
    )

    label_values = sorted(df["label"].unique().tolist())
    label2id = {str(label): int(label) for label in label_values}
    id2label = {str(label): str(label) for label in label_values}

    write_csv(train_df, output_dir / "xgboost_train.csv")
    write_csv(test_df, output_dir / "xgboost_test.csv")
    write_tsv(train_df, output_dir / "train_split.txt")
    write_tsv(test_df, output_dir / "test_split.txt")

    write_fasttext(train_df, output_dir / "fasttext_train.txt")
    write_fasttext(test_df, output_dir / "fasttext_test.txt")

    count_tokens = get_token_counter(args.tokenizer_name)
    train_hbert_stats = write_hbert_jsonl(
        train_df,
        output_dir / "hbert_train.jsonl",
        count_tokens=count_tokens,
        max_tokens=args.max_tokens_per_chunk,
        max_chunks=args.max_chunks,
    )
    test_hbert_stats = write_hbert_jsonl(
        test_df,
        output_dir / "hbert_test.jsonl",
        count_tokens=count_tokens,
        max_tokens=args.max_tokens_per_chunk,
        max_chunks=args.max_chunks,
    )

    save_json(label2id, output_dir / "label2id.json")
    save_json(id2label, output_dir / "id2label.json")

    info = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "total_samples": int(len(df)),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "test_size": args.test_size,
        "num_labels": int(len(label_values)),
        "labels": label_values,
        "label_distribution_total": {
            str(k): int(v) for k, v in Counter(df["label"]).items()
        },
        "label_distribution_train": {
            str(k): int(v) for k, v in Counter(train_df["label"]).items()
        },
        "label_distribution_test": {
            str(k): int(v) for k, v in Counter(test_df["label"]).items()
        },
        "hbert": {
            "max_tokens_per_chunk": args.max_tokens_per_chunk,
            "max_chunks": args.max_chunks,
            "train_avg_chunks": train_hbert_stats["avg_chunks"],
            "train_max_chunks": train_hbert_stats["max_chunks"],
            "test_avg_chunks": test_hbert_stats["avg_chunks"],
            "test_max_chunks": test_hbert_stats["max_chunks"],
        },
        "outputs": {
            "train_split": "train_split.txt",
            "test_split": "test_split.txt",
            "xgboost_train": "xgboost_train.csv",
            "xgboost_test": "xgboost_test.csv",
            "fasttext_train": "fasttext_train.txt",
            "fasttext_test": "fasttext_test.txt",
            "hbert_train": "hbert_train.jsonl",
            "hbert_test": "hbert_test.jsonl",
            "label2id": "label2id.json",
            "id2label": "id2label.json",
        },
    }
    save_json(info, output_dir / "dataset_info.json")

    print("Dataset build completed.")
    print(f"Input: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Num labels: {len(label_values)}")
    print("Generated files:")
    for name in info["outputs"].values():
        print(f"  - {output_dir / name}")


if __name__ == "__main__":
    main()
