# -*- coding: utf-8 -*-
import random
from collections import defaultdict

from config import Config
from text_preprocess import clean_fasttext_line, parse_fasttext_line


def stratified_split(lines, test_ratio=0.1, seed=42):
    grouped = defaultdict(list)
    for line in lines:
        label, _ = parse_fasttext_line(line)
        if label:
            grouped[label].append(line)

    rng = random.Random(seed)
    train_data = []
    test_data = []
    for label_lines in grouped.values():
        rng.shuffle(label_lines)
        test_size = max(1, round(len(label_lines) * test_ratio))
        test_data.extend(label_lines[:test_size])
        train_data.extend(label_lines[test_size:])

    rng.shuffle(train_data)
    rng.shuffle(test_data)
    return train_data, test_data


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    config = Config()

    with open(config.res_data_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    train_data, test_data = stratified_split(lines)
    clean_train_data = [clean_fasttext_line(line) for line in train_data]
    clean_test_data = [clean_fasttext_line(line) for line in test_data]

    write_lines(config.train_data_path, train_data)
    write_lines(config.test_data_path, test_data)
    write_lines(config.clean_train_data_path, clean_train_data)
    write_lines(config.clean_test_data_path, clean_test_data)

    print(f"Total: {len(lines)}")
    print(f"Train: {len(train_data)}")
    print(f"Test: {len(test_data)}")
    print(f"Clean train file: {config.clean_train_data_path}")
    print(f"Clean test file: {config.clean_test_data_path}")


if __name__ == "__main__":
    main()
