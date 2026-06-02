"""
数据分析
"""

import pandas as pd
from config import Config

config = Config()

def data_analysis():
    # 读取数据
    data = pd.read_csv(config.res_data_path, sep=",", names=["text", "label"])
    print(f"data.head(5):{data.head(5)}")
    pass