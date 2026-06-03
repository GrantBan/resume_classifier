"""
配置文件
"""

class Config():
    def __init__(self):
        # 原始数据路径
        self.res_data_path = "../data/train.txt"


        # 类别数据路径
        self.class_data_path = "../data/category_map.txt"


        # 预处理后的数据路径
        self.processed_data_path = "../data/"

        # 模型保存路径
        self.model_save_path = "../model/"



        pass


if __name__ == "__main__":
    config = Config()
    import pandas as pd
    data = pd.read_csv(config.res_data_path, sep=",", names=["text", "label"])
    print(f"data.head(5):{data.head(5)}")