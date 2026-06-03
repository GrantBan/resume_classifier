# Resume Classification

基于英文简历文本的岗位类别自动分类项目。

本项目使用英文简历数据集，将给定简历自动分类到对应岗位类别中。当前项目路线为：

```text
XGBoost + TF-IDF baseline
-> FastText supervised baseline
-> Base BERT + hierarchical chunk-pooling classifier
```

项目目标是先使用 XGBoost 和 FastText 构建可复现的基线模型，再使用 `bert-base-uncased` 作为底层编码器，结合自定义的层级分块聚合结构处理长简历文本。本文档中的 Hierarchical BERT 指“Base BERT 编码器 + chunk 级编码 + 文档级聚合”的模型结构，不是直接使用现成的 Hierarchical BERT 预训练模型。

## Project Background

在招聘场景中，企业每天会收到大量求职者简历。人工筛选简历不仅耗时，而且容易受到主观因素影响。

本项目希望通过文本分类模型，根据简历内容自动判断候选人的岗位方向，例如：

- Human Resources
- Information Technology
- Sales
- Finance
- Engineering
- Teacher
- Healthcare

该能力可以用于：

- 简历自动归类
- HR 初筛辅助
- 人才库管理
- 岗位推荐
- 招聘系统智能分发

## Dataset

数据集来自 LiveCareer 网站中的英文简历示例，由数据集作者通过网页爬虫采集整理。

数据集中包含 2400+ 份英文简历，每份简历标注了对应岗位类别。

原始 CSV 字段如下：

| Field | Description |
| --- | --- |
| `ID` | Unique identifier and PDF filename |
| `Resume_str` | Resume text in plain string format |
| `Resume_html` | Resume content in HTML format |
| `Category` | Job category label |

本项目 baseline 阶段主要使用：

```text
Resume_str -> Category
```

## Categories

原始数据集包含 24 个岗位类别。为了提高类别平衡性，本项目计划删除样本数量最少的 4 个类别，最终使用 20 个类别进行建模。

| Category | Chinese Meaning |
| --- | --- |
| `HR` | 人力资源 |
| `DESIGNER` | 设计师 |
| `INFORMATION-TECHNOLOGY` | 信息技术 |
| `TEACHER` | 教师 |
| `ADVOCATE` | 法务/律师 |
| `BUSINESS-DEVELOPMENT` | 商务拓展 |
| `HEALTHCARE` | 医疗健康 |
| `FITNESS` | 健身 |
| `SALES` | 销售 |
| `CONSULTANT` | 咨询顾问 |
| `CHEF` | 厨师/餐饮 |
| `FINANCE` | 金融 |
| `APPAREL` | 服装 |
| `ENGINEERING` | 工程 |
| `ACCOUNTANT` | 会计 |
| `CONSTRUCTION` | 建筑施工 |
| `PUBLIC-RELATIONS` | 公共关系 |
| `BANKING` | 银行业务 |
| `ARTS` | 艺术 |
| `AVIATION` | 航空 |

Removed low-sample categories:

| Removed Category | Chinese Meaning | Original Count |
| --- | --- | ---: |
| `BPO` | 业务流程外包/客服外包 | 22 |
| `AUTOMOBILE` | 汽车行业 | 36 |
| `AGRICULTURE` | 农业 | 63 |
| `DIGITAL-MEDIA` | 数字媒体 | 96 |

## Local Data Summary

当前本地数据统计如下：

```text
Samples: 2264
Fields: 4
Classes: 20
```

文本长度统计：

| Metric | Value |
| --- | ---: |
| Mean length | about 6295 characters |
| Median length | about 5886 characters |
| Minimum length in original statistics | 21 characters |
| Maximum length | 38842 characters |

原始统计中 21 字符样本被视为噪声风险样本。当前训练/测试文件中已加入过短样本过滤规则，过滤少于 200 字符或少于 50 token 的样本。

类别存在一定不均衡，例如：

| Category | Count |
| --- | ---: |
| `INFORMATION-TECHNOLOGY` | 120 |
| `BUSINESS-DEVELOPMENT` | 120 |
| `HR` | 110 |
| `APPAREL` | 97 |
删除低样本类别后，剩余类别的样本数量主要分布在 97 到 120 之间，整体更加均衡。因此评估模型时仍然不能只看 Accuracy，还需要重点关注 Macro F1。

## Task Definition

本项目是一个多分类文本分类任务。

Input:

```text
English resume text
```

Output:

```text
One of 20 job categories
```

Example:

```text
Resume:
Software engineer with Python, SQL, Linux, database development and cloud service experience.

Prediction:
INFORMATION-TECHNOLOGY
```

## Model Plan

### 1. XGBoost Baseline

XGBoost 不能直接处理原始文本，因此需要先使用 TF-IDF 提取文本特征。

Pipeline:

```text
Resume_str
-> text cleaning
-> TF-IDF vectorization
-> XGBoost classifier
-> Category prediction
```

Recommended TF-IDF settings:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50000,
    min_df=2,
    max_df=0.9
)
```

If TF-IDF features are too large, `max_features` can be reduced or `TruncatedSVD` can be applied before XGBoost.

### 2. FastText Baseline

FastText supervised 模型使用原始英文文本训练。

FastText training format:

```text
__label__HR HR specialist with recruiting onboarding employee relations and training experience.
__label__INFORMATION-TECHNOLOGY Software engineer with Python SQL Linux and database experience.
```

Pipeline:

```text
Resume_str
-> text cleaning
-> fastText label format
-> FastText supervised model
-> Category prediction
```

### 3. Base BERT + Hierarchical Pooling

由于简历文本较长，普通 BERT 直接截断会丢失大量信息。

本项目深度学习主线不是直接加载现成的 Hierarchical BERT 模型，而是使用 `bert-base-uncased` 作为 base BERT 编码器，并在其外部自定义长文本分块、chunk 聚合和分类层。

Pipeline:

```text
Resume_str
-> text cleaning
-> split into chunks
-> encode each chunk with bert-base-uncased
-> aggregate chunk embeddings
-> classifier
-> Category prediction
```

Input tensor shape:

```text
input_ids:      [batch_size, num_chunks, chunk_size]
attention_mask: [batch_size, num_chunks, chunk_size]
chunk_mask:     [batch_size, num_chunks]
labels:         [batch_size]
```

Recommended initial settings:

| Parameter | Value |
| --- | --- |
| Base encoder | `bert-base-uncased` |
| Chunk size | 512 tokens |
| Max chunks per resume | 4 |
| Max content tokens per resume | 2040 |
| Initial aggregation | mean pooling |
| Advanced aggregation | attention pooling |
| Number of classes | 20 |

In this project, "Hierarchical BERT" means the custom hierarchical long-document classifier built on top of base BERT, not a separate pretrained Hierarchical BERT checkpoint.

## Evaluation

由于类别不均衡，推荐使用以下指标：

| Metric | Purpose |
| --- | --- |
| Accuracy | Overall correctness |
| Macro Precision | Average precision across classes |
| Macro Recall | Average recall across classes |
| Macro F1 | Main metric for imbalanced classes |
| Weighted F1 | Class-frequency weighted F1 |
| Confusion Matrix | Analyze class confusion |

重点关注：

```text
Macro F1
```

## Suggested Project Structure

```text
resume-classification/
│
├── README.md
├── 项目书-英文简历岗位分类基线模型.md
│
├── data/
│   └── Resume.csv
│
├── src/
│   ├── data_analysis.py
│   ├── train_xgboost.py
│   ├── train_fasttext.py
│   ├── train_hierarchical_bert.py
│   ├── evaluate.py
│   └── predict.py
│
├── model/
│   ├── baseline_tfidf_xgboost.pkl
│   ├── baseline_fasttext.bin
│   ├── best_hierarchical_bert.pt
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── result/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── metrics.json
│
├── requirements.txt
└── .gitignore
```

Large data files and model files should not be committed to Git.

## Development Roadmap

| Stage | Task | Output |
| --- | --- | --- |
| 1 | Data analysis and cleaning | Data report |
| 2 | XGBoost baseline | TF-IDF + XGBoost model |
| 3 | FastText baseline | FastText supervised model |
| 4 | Model evaluation | Metrics and confusion matrix |
| 5 | Data augmentation | Balanced training data |
| 6 | Base BERT + hierarchical pooling | Long-document classification model |
| 7 | Deployment | Web demo or API service |

## Notes

- The dataset is collected from resume examples on LiveCareer.
- This project is for learning and research practice.
- The original dataset may contain template bias because it comes from resume examples.
- Some categories have limited samples, so Macro F1 should be emphasized.
- Resume texts are long, so base BERT is used with hierarchical chunk pooling instead of simple truncation.

## Current Status

Current project stage:

```text
Project planning and baseline model preparation
```

Next steps:

```text
1. Build data analysis script
2. Train XGBoost baseline
3. Train FastText baseline
4. Compare baseline metrics
5. Train Base BERT + hierarchical pooling classifier
```
