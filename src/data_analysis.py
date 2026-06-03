"""
数据分析辅助脚本。
"""

from config import Config


config = Config()


def load_category_map():
    """读取类别 id 到类别名称的映射。"""
    id2name = {}
    with open(config.class_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            index, name = line.split("\t")
            id2name[index] = name
    return id2name


if __name__ == "__main__":
    category_map = load_category_map()
    print(f"类别数量: {len(category_map)}")
    print(f"类别映射: {category_map}")
