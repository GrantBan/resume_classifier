"""
训练用于英文简历岗位分类的 Hierarchical BERT 模型。

当前数据只有训练集和测试集，没有单独验证集。因此训练过程中不使用测试集
挑选最优模型，避免测试集信息泄漏；模型在训练结束后保存最终权重，并只在
最后对测试集评估一次。
"""

from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score

from config import Config
from datasets_bd import build_dataloader
from Hierarchical_BERT import HierarchicalBERT_model


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
    return save_path / "final_hierarchical_bert.pt"


def build_metrics(true_labels, preds):
    """统一计算分类报告、F1、准确率和精确率。"""
    labels = list(range(len(conf.class_list)))
    report = classification_report(
        true_labels,
        preds,
        labels=labels,
        target_names=conf.class_list,
        output_dict=True,
        zero_division=0,
    )
    f1 = f1_score(true_labels, preds, average="micro", zero_division=0)
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, average="micro", zero_division=0)
    return report, f1, acc, prec


def evaluate_model(model, data_loader):
    """在测试集上评估模型，不参与梯度更新。"""
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, chunk_mask, labels = to_device(batch)
            y_pred = model(input_ids, attention_mask, chunk_mask)
            preds.extend(torch.argmax(y_pred, dim=1).cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())

    return build_metrics(true_labels, preds)


def save_final_model(model, save_path):
    """保存训练结束后的最终模型。"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_list": conf.class_list,
            "bert_model_name": conf.bert_model_name,
            "max_chunks": conf.max_chunks,
            "pad_size": conf.pad_size,
        },
        save_path,
    )
    print(f"模型已保存: {save_path}")


def model2train():
    """执行完整训练流程，训练结束后在测试集上评估一次。"""
    train_dataloader, test_dataloader = build_dataloader()

    print(
        f"device: {conf.device} ({get_device_name()}), "
        f"debug_mode: {conf.debug_mode}",
        flush=True,
    )
    print(
        f"batches/epoch: {len(train_dataloader)}, "
        f"batch_size: {conf.batch_size}",
        flush=True,
    )

    model = HierarchicalBERT_model(conf, num_labels=len(conf.class_list)).to(conf.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=conf.learning_rate,
        weight_decay=conf.weight_decay,
    )

    save_path = get_model_save_path()
    final_train_result = None

    for epoch in range(conf.num_epochs):
        model.train()
        total_loss = 0.0
        train_batch_count = 0
        train_preds, train_labels = [], []
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids, attention_mask, chunk_mask, labels = to_device(batch)

            y_pred = model(input_ids, attention_mask, chunk_mask)
            raw_loss = loss_fn(y_pred, labels)

            # 梯度累积可以降低单步显存占用，适合长文本 BERT 训练。
            loss = raw_loss / conf.gradient_accumulation_steps
            loss.backward()

            should_stop = conf.debug_mode and batch_idx >= 1

            # 到达累积步数、最后一个 batch 或调试模式结束时再更新参数。
            should_step = (
                (batch_idx + 1) % conf.gradient_accumulation_steps == 0
                or (batch_idx + 1) == len(train_dataloader)
                or should_stop
            )

            if should_step:
                if conf.max_grad_norm:
                    # 梯度裁剪用于缓解梯度爆炸，使训练更稳定。
                    torch.nn.utils.clip_grad_norm_(model.parameters(), conf.max_grad_norm)
                optimizer.step()
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
                    f"  batch {batch_idx + 1}/{len(train_dataloader)}, "
                    f"loss: {raw_loss.item():.4f}, "
                    f"avg: {total_loss / (batch_idx + 1):.4f}",
                    flush=True,
                )

        avg_loss = total_loss / max(1, train_batch_count)
        train_report, train_f1, train_acc, train_prec = build_metrics(
            train_labels,
            train_preds,
        )

        print(f"Epoch {epoch + 1}/{conf.num_epochs}, Train Loss: {avg_loss:.4f}")
        print(
            f"Train F1: {train_f1:.4f}, "
            f"Accuracy: {train_acc:.4f}, "
            f"Precision: {train_prec:.4f}",
        )

        final_train_result = (train_report, train_f1, train_acc, train_prec)

    save_final_model(model, save_path)
    test_report, test_f1, test_acc, test_prec = evaluate_model(model, test_dataloader)

    return (*final_train_result, test_report, test_f1, test_acc, test_prec)


if __name__ == "__main__":
    (
        train_report,
        train_f1,
        train_acc,
        train_prec,
        test_report,
        test_f1,
        test_acc,
        test_prec,
    ) = model2train()

    print("=== 最后一轮训练集结果 ===")
    print(f"Train F1: {train_f1:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}")
    print(train_report)

    print("=== 测试集结果 ===")
    print(f"Test F1: {test_f1:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}")
    print(test_report)
