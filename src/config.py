# -*- coding: utf-8 -*-
from pathlib import Path


class Config:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data"
        self.model_dir = self.project_root / "model"

        self.res_data_path = str(self.data_dir / "fasttext_train_data.txt")
        self.class_data_path = str(self.data_dir / "category_map.txt")

        self.processed_data_path = str(self.data_dir)
        self.train_data_path = str(self.data_dir / "ft_train.txt")
        self.test_data_path = str(self.data_dir / "ft_test.txt")
        self.clean_train_data_path = str(self.data_dir / "ft_train_clean.txt")
        self.clean_test_data_path = str(self.data_dir / "ft_test_clean.txt")

        self.model_save_path = str(self.model_dir)


if __name__ == "__main__":
    config = Config()
    print(config.res_data_path)
