import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler


class FeatureConfig:
    """特征分类配置参数类"""
    # 特征路径
    # FEATURE_ROOT = "./results/replk_densenet_new/features"
    # RESULT_ROOT = "./classicify/feature_fusion_5"
    FEATURE_ROOT = "./features"
    RESULT_ROOT = "./classicify2/feature_fusion_5"
    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出路径
    LOG_DIR = os.path.join(RESULT_ROOT, "logs")
    FIG_DIR = os.path.join(RESULT_ROOT, "figures")
    CHECKPOINT_DIR = os.path.join(RESULT_ROOT, "checkpoints")

    def __init__(self):
        self.create_directories()

    def create_directories(self):
        """创建输出目录"""
        dirs = [self.RESULT_ROOT, self.LOG_DIR, self.FIG_DIR, self.CHECKPOINT_DIR]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"创建目录: {dir_path}")


config = FeatureConfig()


class GatedFeatureFusionMLP(nn.Module):
    """门控特征融合MLP分类器"""

    def __init__(self, replk_dim=1024, densenet_dim=1024, num_classes=5, hidden_dim=512):
        super(GatedFeatureFusionMLP, self).__init__()

        self.replk_dim = replk_dim
        self.densenet_dim = densenet_dim

        # 门控网络：使用MLP + Sigmoid生成门控向量
        self.gate_network = nn.Sequential(
            nn.Linear(replk_dim + densenet_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, replk_dim),  # 输出维度与单个特征维度相同
            nn.Sigmoid()  # 输出值在0~1之间
        )

        # 特征融合后的MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(replk_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, replk_features, densenet_features):
        """
        前向传播
        Args:
            replk_features: RepLKNet特征, shape: (batch_size, replk_dim)
            densenet_features: DenseNet特征, shape: (batch_size, densenet_dim)
        """
        # 拼接两个特征源
        concatenated_features = torch.cat([replk_features, densenet_features], dim=1)

        # 生成门控向量 g (值在0~1之间)
        g = self.gate_network(concatenated_features)

        # 门控特征融合: g * feat_a + (1 - g) * feat_b
        # 假设replk_dim == densenet_dim，如果不相等需要调整
        if self.replk_dim == self.densenet_dim:
            fused_features = g * replk_features + (1 - g) * densenet_features
        else:
            # 如果维度不相等，使用投影使其匹配
            if self.replk_dim > self.densenet_dim:
                # 对densenet特征进行投影使其维度与replk匹配
                densenet_proj = nn.Linear(self.densenet_dim, self.replk_dim).to(replk_features.device)(
                    densenet_features)
                fused_features = g * replk_features + (1 - g) * densenet_proj
            else:
                # 对replk特征进行投影使其维度与densenet匹配
                replk_proj = nn.Linear(self.replk_dim, self.densenet_dim).to(replk_features.device)(replk_features)
                fused_features = g * replk_proj + (1 - g) * densenet_features

        # 分类
        output = self.classifier(fused_features)
        return output


def load_features(fold_idx, split_name):
    """加载指定折和分割的特征"""
    # 加载RepLKNet特征
    replk_npz_path = os.path.join(config.FEATURE_ROOT, f"fold_{fold_idx + 1}_{split_name}_replk_features.npz")
    replk_data = np.load(replk_npz_path)
    replk_features = replk_data['features']
    labels = replk_data['labels']

    # 加载DenseNet特征
    densenet_npz_path = os.path.join(config.FEATURE_ROOT, f"fold_{fold_idx + 1}_{split_name}_densenet_features.npz")
    densenet_data = np.load(densenet_npz_path)
    densenet_features = densenet_data['features']

    print(f"加载 {split_name} 特征 - Fold {fold_idx + 1}:")
    print(f"RepLKNet特征形状: {replk_features.shape}")
    print(f"DenseNet特征形状: {densenet_features.shape}")
    print(f"标签形状: {labels.shape}")

    return replk_features, densenet_features, labels


def create_dataloader(replk_features, densenet_features, labels, batch_size=32, shuffle=True):
    """创建特征数据加载器"""
    # 转换为Tensor
    replk_tensor = torch.FloatTensor(replk_features)
    densenet_tensor = torch.FloatTensor(densenet_features)
    labels_tensor = torch.LongTensor(labels)

    # 创建数据集
    dataset = TensorDataset(replk_tensor, densenet_tensor, labels_tensor)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    return dataloader


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, fold):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (replk_features, densenet_features, labels) in enumerate(train_loader):
        replk_features = replk_features.to(config.DEVICE)
        densenet_features = densenet_features.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        # 前向传播
        optimizer.zero_grad()
        logits = model(replk_features, densenet_features)
        loss = criterion(logits, labels)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 累计损失与预测结果
        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / ((batch_idx + 1) * config.BATCH_SIZE)
            print(f"Fold {fold + 1} | Epoch {epoch + 1}/{config.EPOCHS} | "
                  f"Batch {batch_idx + 1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

    # 计算训练指标
    avg_train_loss = total_loss / len(train_loader.dataset)
    train_acc = accuracy_score(all_labels, all_preds)
    return avg_train_loss, train_acc


def test_one_epoch(model, test_loader, criterion):
    """测试一个epoch"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for replk_features, densenet_features, labels in test_loader:
            replk_features = replk_features.to(config.DEVICE)
            densenet_features = densenet_features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # 前向传播
            logits = model(replk_features, densenet_features)
            loss = criterion(logits, labels)

            # 累计损失与预测结果
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # 计算测试指标
    avg_test_loss = total_loss / len(test_loader.dataset)
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_test_loss, test_acc, test_precision, test_recall, test_f1, cm


def plot_metrics(fold, train_losses, test_losses, train_accs, test_accs):
    """绘制损失和准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = list(range(1, len(train_losses) + 1))

    # 损失曲线
    ax1.plot(epochs, train_losses, label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(epochs, test_losses, label="Test Loss", color="#3498db", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Fold {fold + 1} Gated Feature Fusion Loss Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, train_accs, label="Train Accuracy", color="#2ecc71", linewidth=2)
    ax2.plot(epochs, test_accs, label="Test Accuracy", color="#f39c12", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Fold {fold + 1} Gated Feature Fusion Accuracy Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 保存图像
    fig_path = os.path.join(config.FIG_DIR, f"fold_{fold + 1}_gated_feature_fusion_metrics.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(fold, cm, classes):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)

    # 设置标签
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 添加文本标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Fold {fold + 1} Gated Feature Fusion Confusion Matrix")
    plt.tight_layout()

    # 保存图像
    cm_path = os.path.join(config.FIG_DIR, f"fold_{fold + 1}_gated_feature_fusion_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_log(fold, log_dict):
    """保存训练日志"""
    log_path = os.path.join(config.LOG_DIR, f"fold_{fold + 1}_gated_feature_fusion_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, ensure_ascii=False, indent=4)


def save_checkpoint(fold, epoch, model, optimizer, best_acc):
    """保存最佳模型参数"""
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"fold_{fold + 1}_gated_feature_fusion_best.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc
    }, checkpoint_path)


def train_fold_with_features(fold_idx):
    """使用特征训练单个折"""
    print(f"\n{'=' * 50}")
    print(f"开始门控特征融合训练第 {fold_idx + 1} 折")
    print(f"{'=' * 50}\n")

    # 加载特征
    train_replk, train_densenet, train_labels = load_features(fold_idx, "train")
    test_replk, test_densenet, test_labels = load_features(fold_idx, "test")

    # 创建数据加载器
    train_loader = create_dataloader(train_replk, train_densenet, train_labels,
                                     batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = create_dataloader(test_replk, test_densenet, test_labels,
                                    batch_size=config.BATCH_SIZE, shuffle=False)

    # 初始化门控特征融合模型
    model = GatedFeatureFusionMLP(
        replk_dim=train_replk.shape[1],
        densenet_dim=train_densenet.shape[1],
        num_classes=5
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 记录训练过程
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    test_precisions = []
    test_recalls = []
    test_f1s = []
    best_acc = 0.0
    best_epoch = 0
    best_cm = None

    # 开始训练
    for epoch in range(config.EPOCHS):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, fold_idx)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试
        test_loss, test_acc, test_precision, test_recall, test_f1, cm = test_one_epoch(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1s.append(test_f1)

        # 打印 epoch 结果
        print(f"Fold {fold_idx + 1} | Epoch {epoch + 1}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_cm = cm
            save_checkpoint(fold_idx, epoch, model, optimizer, best_acc)

    # 绘制并保存指标曲线
    plot_metrics(fold_idx, train_losses, test_losses, train_accs, test_accs)

    # 绘制并保存最佳混淆矩阵
    classes = [f"class_{i + 1}" for i in range(5)]
    plot_confusion_matrix(fold_idx, best_cm, classes)

    # 保存日志
    log_dict = {
        "fold": fold_idx + 1,
        "fusion_type": "gated",
        "best_epoch": best_epoch,
        "best_accuracy": best_acc,
        "best_precision": test_precisions[best_epoch - 1],
        "best_recall": test_recalls[best_epoch - 1],
        "best_f1": test_f1s[best_epoch - 1],
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
        "test_precisions": test_precisions,
        "test_recalls": test_recalls,
        "test_f1s": test_f1s
    }
    save_log(fold_idx, log_dict)

    return {
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "accuracy": best_acc,
        "precision": test_precisions[best_epoch - 1],
        "recall": test_recalls[best_epoch - 1],
        "f1": test_f1s[best_epoch - 1]
    }


def main():
    """主函数：使用特征进行五折交叉验证"""
    print(f"使用设备: {config.DEVICE}")
    print(f"特征路径: {config.FEATURE_ROOT}")
    print(f"结果保存路径: {config.RESULT_ROOT}")
    print(f"融合方案: 门控特征融合\n")

    # 检查特征文件是否存在
    if not os.path.exists(config.FEATURE_ROOT):
        print(f"错误：特征目录不存在: {config.FEATURE_ROOT}")
        print("请先运行特征提取代码生成特征文件")
        return

    # 执行五折交叉验证
    fold_results = []
    for fold_idx in range(5):
        result = train_fold_with_features(fold_idx)
        fold_results.append(result)
        print(f"第 {fold_idx + 1} 折门控特征融合训练完成 - 最佳准确率: {result['accuracy']:.4f}\n")

    # 汇总所有折的结果
    print("\n" + "=" * 50)
    print("门控特征融合五折交叉验证结果汇总")
    print("=" * 50)

    # 计算平均值和标准差
    accuracies = [r["accuracy"] for r in fold_results]
    precisions = [r["precision"] for r in fold_results]
    recalls = [r["recall"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]

    summary = {
        "fusion_type": "gated",
        "fold_results": fold_results,
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "mean_precision": np.mean(precisions),
        "std_precision": np.std(precisions),
        "mean_recall": np.mean(recalls),
        "std_recall": np.std(recalls),
        "mean_f1": np.mean(f1s),
        "std_f1": np.std(f1s)
    }

    # 打印汇总结果
    print(f"平均准确率: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"平均精确率: {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
    print(f"平均召回率: {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
    print(f"平均F1分数: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")

    # 保存汇总结果
    summary_path = os.path.join(config.RESULT_ROOT, "gated_feature_fusion_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"\n汇总结果已保存至: {summary_path}")


if __name__ == "__main__":
    main()