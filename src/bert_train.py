"""
训练用于英文简历岗位分类的 Base BERT + 层级分块聚合模型。

优化策略：
1. 从训练集中分层切出 dev 集，用 dev Macro F1 保存最优模型。
2. 测试集只在训练结束后评估一次，避免测试集信息泄漏。
3. 同时记录 micro / macro / weighted 指标，并保存混淆矩阵。
"""

import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import Config
from datasets_bd import build_dataloader
from Hierarchical_BERT import HierarchicalBERT_model

try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    get_linear_schedule_with_warmup = None


conf = Config()


def to_device(batch):
    """将一个 batch 中的所有张量移动到当前训练设备。"""
    input_ids, attention_mask, chunk_mask, labels = batch
    return (
        input_ids.to(conf.device, non_blocking=True),
        attention_mask.to(conf.device, non_blocking=True),
        chunk_mask.to(conf.device, non_blocking=True),
        labels.to(conf.device, non_blocking=True),
    )


def get_device_name():
    """安全获取设备名称，避免没有 GPU 时调用 CUDA 接口报错。"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def get_model_save_path():
    """确保模型保存目录存在，并返回最终的模型文件路径。"""
    save_path = Path(conf.model_save_path)
    if save_path.suffix:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        return save_path

    save_path.mkdir(parents=True, exist_ok=True)
    return save_path / "best_hierarchical_bert.pt"


def build_metrics(true_labels, preds):
    """统一计算分类报告、常用指标和混淆矩阵。"""
    labels = list(range(len(conf.class_list)))
    report = classification_report(
        true_labels,
        preds,
        labels=labels,
        target_names=conf.class_list,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(true_labels, preds),
        "micro_precision": precision_score(
            true_labels, preds, average="micro", zero_division=0
        ),
        "micro_recall": recall_score(
            true_labels, preds, average="micro", zero_division=0
        ),
        "micro_f1": f1_score(true_labels, preds, average="micro", zero_division=0),
        "macro_precision": precision_score(
            true_labels, preds, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            true_labels, preds, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(true_labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            true_labels, preds, average="weighted", zero_division=0
        ),
    }
    matrix = confusion_matrix(true_labels, preds, labels=labels)
    return report, metrics, matrix


def evaluate_model(model, data_loader):
    """在 dev/test 集上评估模型，不参与梯度更新。"""
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, chunk_mask, labels = to_device(batch)
            y_pred = model(input_ids, attention_mask, chunk_mask)
            preds.extend(torch.argmax(y_pred, dim=1).cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())

    return build_metrics(true_labels, preds)


def build_scheduler(optimizer, train_dataloader):
    """构建线性 warmup 学习率调度器。"""
    if get_linear_schedule_with_warmup is None or conf.warmup_ratio <= 0:
        return None

    steps_per_epoch = math.ceil(len(train_dataloader) / conf.gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * conf.num_epochs)
    warmup_steps = int(total_steps * conf.warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def train_one_epoch(model, train_dataloader, loss_fn, optimizer, scheduler, epoch):
    """训练一个 epoch，并返回训练集指标。"""
    model.train()
    total_loss = 0.0
    train_batch_count = 0
    train_preds, train_labels = [], []
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_dataloader):
        input_ids, attention_mask, chunk_mask, labels = to_device(batch)

        y_pred = model(input_ids, attention_mask, chunk_mask)
        raw_loss = loss_fn(y_pred, labels)

        loss = raw_loss / conf.gradient_accumulation_steps
        loss.backward()

        should_stop = conf.debug_mode and batch_idx >= 1
        should_step = (
            (batch_idx + 1) % conf.gradient_accumulation_steps == 0
            or (batch_idx + 1) == len(train_dataloader)
            or should_stop
        )

        if should_step:
            if conf.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += raw_loss.item()
        train_batch_count += 1
        preds = torch.argmax(y_pred, dim=1)
        train_preds.extend(preds.cpu().numpy().tolist())
        train_labels.extend(labels.cpu().numpy().tolist())

        if should_stop:
            break

        if conf.log_interval and (batch_idx + 1) % conf.log_interval == 0:
            print(
                f"  epoch {epoch}, batch {batch_idx + 1}/{len(train_dataloader)}, "
                f"loss: {raw_loss.item():.4f}, "
                f"avg: {total_loss / train_batch_count:.4f}",
                flush=True,
            )

    avg_loss = total_loss / max(1, train_batch_count)
    report, metrics, matrix = build_metrics(train_labels, train_preds)
    return avg_loss, report, metrics, matrix


def save_checkpoint(model, save_path, epoch, dev_metrics):
    """保存 dev 集 Macro F1 最优的模型。"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_list": conf.class_list,
            "bert_model_name": conf.bert_model_name,
            "max_chunks": conf.max_chunks,
            "pad_size": conf.pad_size,
            "pooling_type": conf.pooling_type,
            "dev_metrics": dev_metrics,
        },
        save_path,
    )
    print(f"模型已保存: {save_path}")


def load_best_checkpoint(model, save_path):
    """加载 dev 集表现最好的模型权重。"""
    checkpoint = torch.load(save_path, map_location=conf.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_evaluation_outputs(report, matrix, prefix):
    """保存分类报告和混淆矩阵，便于分析弱类。"""
    result_dir = Path(conf.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    report_path = result_dir / f"{prefix}_classification_report.json"
    matrix_path = result_dir / f"{prefix}_confusion_matrix.csv"

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *conf.class_list])
        for class_name, row in zip(conf.class_list, matrix):
            writer.writerow([class_name, *row.tolist()])

    print(f"分类报告已保存: {report_path}")
    print(f"混淆矩阵已保存: {matrix_path}")


def format_metrics(metrics):
    """格式化指标输出。"""
    return (
        f"Acc: {metrics['accuracy']:.4f}, "
        f"Micro F1: {metrics['micro_f1']:.4f}, "
        f"Macro F1: {metrics['macro_f1']:.4f}, "
        f"Macro Recall: {metrics['macro_recall']:.4f}, "
        f"Weighted F1: {metrics['weighted_f1']:.4f}"
    )


def model2train():
    """执行完整训练流程，使用 dev 集选最优模型，最后评估测试集。"""
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader(return_dev=True)

    print(
        f"device: {conf.device} ({get_device_name()}), "
        f"debug_mode: {conf.debug_mode}",
        flush=True,
    )
    print(
        f"batches/epoch: {len(train_dataloader)}, "
        f"batch_size: {conf.batch_size}, "
        f"pooling: {conf.pooling_type}",
        flush=True,
    )

    model = HierarchicalBERT_model(conf, num_labels=len(conf.class_list)).to(conf.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=conf.learning_rate,
        weight_decay=conf.weight_decay,
    )
    scheduler = build_scheduler(optimizer, train_dataloader)

    save_path = get_model_save_path()
    best_dev_macro_f1 = -1.0
    patience_count = 0
    best_train_result = None
    best_dev_result = None

    for epoch in range(1, conf.num_epochs + 1):
        avg_loss, train_report, train_metrics, _ = train_one_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
        )
        dev_report, dev_metrics, dev_matrix = evaluate_model(model, dev_dataloader)

        print(f"Epoch {epoch}/{conf.num_epochs}, Train Loss: {avg_loss:.4f}")
        print(f"Train: {format_metrics(train_metrics)}")
        print(f"Dev:   {format_metrics(dev_metrics)}")

        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            patience_count = 0
            best_train_result = (train_report, train_metrics)
            best_dev_result = (dev_report, dev_metrics, dev_matrix)
            save_checkpoint(model, save_path, epoch, dev_metrics)
        else:
            patience_count += 1
            print(
                f"Dev Macro F1 未提升，early stopping 计数: "
                f"{patience_count}/{conf.early_stopping_patience}",
                flush=True,
            )

        if patience_count >= conf.early_stopping_patience:
            print("触发 early stopping，停止继续训练。")
            break

    checkpoint = load_best_checkpoint(model, save_path)
    print(f"已加载 dev 最优模型，最佳 epoch: {checkpoint['epoch']}")

    test_report, test_metrics, test_matrix = evaluate_model(model, test_dataloader)
    save_evaluation_outputs(test_report, test_matrix, prefix="bert_test")

    return best_train_result, best_dev_result, (test_report, test_metrics, test_matrix)


if __name__ == "__main__":
    train_result, dev_result, test_result = model2train()

    train_report, train_metrics = train_result
    dev_report, dev_metrics, _ = dev_result
    test_report, test_metrics, _ = test_result

    print("=== 训练集结果（dev 最优 epoch 对应） ===")
    print(format_metrics(train_metrics))
    print(train_report)

    print("=== Dev 集结果 ===")
    print(format_metrics(dev_metrics))
    print(dev_report)

    print("=== 测试集结果 ===")
    print(format_metrics(test_metrics))
    print(test_report)
