import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')


class FeatureExtractorConfig:
    """特征提取配置参数类"""
    # 数据路径
    DATA_ROOT = "./SIPaKMeD_5_fold"
    FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
    CLASSES = ["class_1", "class_2", "class_3", "class_4", "class_5"]
    NUM_CLASSES = 5

    # 模型参数
    FEATURE_DIM = 1024  # DenseNet121在分类器前的特征维度

    # 特征提取参数
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出路径
    CHECKPOINT_DIR = "./results/densenet_only/checkpoints"
    FEATURE_DIR = "./results/densenet_only/features"

    def __init__(self):
        self.create_directories()

    def create_directories(self):
        """创建输出目录"""
        if not os.path.exists(self.FEATURE_DIR):
            os.makedirs(self.FEATURE_DIR, exist_ok=True)
            print(f"创建特征目录: {self.FEATURE_DIR}")


feature_config = FeatureExtractorConfig()


class DenseNetFeatureExtractor(nn.Module):
    """DenseNet特征提取器 - 修改为直接获取全局平均池化后的特征"""

    def __init__(self, num_classes):
        super(DenseNetFeatureExtractor, self).__init__()
        # 加载预训练的DenseNet121
        self.densenet = models.densenet121(pretrained=True)

        # 保存原始分类器的输入维度
        original_in_features = self.densenet.classifier.in_features

        # 替换分类器（与训练时保持一致）
        self.densenet.classifier = nn.Sequential(
            nn.Linear(original_in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 通过DenseNet的特征部分
        features = self.densenet.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        # 通过分类器获取预测结果
        logits = self.densenet.classifier(out)

        return logits, out  # 返回预测结果和特征


def get_feature_transforms():
    """特征提取的数据预处理转换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_fold_data_for_features(fold_idx):
    """加载指定折的训练集和测试集数据用于特征提取"""
    # 当前折名称
    current_fold = feature_config.FOLDS[fold_idx]

    # 训练集：仅使用当前折的train目录
    train_dir = os.path.join(feature_config.DATA_ROOT, current_fold, "train")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=get_feature_transforms())

    # 测试集：使用当前折的test目录
    test_dir = os.path.join(feature_config.DATA_ROOT, current_fold, "test")
    test_dataset = datasets.ImageFolder(root=test_dir, transform=get_feature_transforms())

    print(f"加载特征提取数据 - {current_fold}:")
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  测试集样本数: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=feature_config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=feature_config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, current_fold, train_dataset.classes, train_dataset.samples


def extract_features(model, data_loader, device):
    """提取特征"""
    model.eval()
    all_features = []
    all_labels = []
    all_preds = []
    all_probs = []
    all_logits = []
    all_image_paths = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播获取输出和特征
            logits, features = model(images)

            # 获取预测结果和概率
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            # 收集数据
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理批次: {batch_idx + 1}/{len(data_loader)}")

    # 合并所有批次的数据
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.vstack(all_probs)
    all_logits = np.vstack(all_logits)

    return all_features, all_labels, all_preds, all_probs, all_logits


def save_features_separate(fold_idx, fold_name, train_features, train_labels, train_preds, train_probs,
                           test_features, test_labels, test_preds, test_probs, class_names):
    """分别保存训练集和测试集特征到npz和csv文件"""

    # 保存训练集特征
    train_npz_path = os.path.join(feature_config.FEATURE_DIR, f"{fold_name}_train_densenet_features.npz")
    np.savez(
        train_npz_path,
        features=train_features,
        labels=train_labels,
        preds=train_preds,
        probs=train_probs,
        class_names=class_names,
        fold_name=fold_name,
        split='train',
        feature_dim=feature_config.FEATURE_DIM
    )
    print(f"训练集特征已保存为NPZ文件: {train_npz_path}")

    # 保存测试集特征
    test_npz_path = os.path.join(feature_config.FEATURE_DIR, f"{fold_name}_test_densenet_features.npz")
    np.savez(
        test_npz_path,
        features=test_features,
        labels=test_labels,
        preds=test_preds,
        probs=test_probs,
        class_names=class_names,
        fold_name=fold_name,
        split='test',
        feature_dim=feature_config.FEATURE_DIM
    )
    print(f"测试集特征已保存为NPZ文件: {test_npz_path}")

    # 保存训练集CSV文件
    train_csv_path = os.path.join(feature_config.FEATURE_DIR, f"{fold_name}_train_densenet_features.csv")
    train_df_data = []
    for i in range(len(train_features)):
        sample_data = {
            'fold': fold_name,
            'split': 'train',
            'true_label': train_labels[i],
            'true_class': class_names[train_labels[i]],
            'pred_label': train_preds[i],
            'pred_class': class_names[train_preds[i]],
            'prob_class_0': train_probs[i][0],
            'prob_class_1': train_probs[i][1],
            'prob_class_2': train_probs[i][2],
            'prob_class_3': train_probs[i][3],
            'prob_class_4': train_probs[i][4],
        }
        # 添加特征
        for j in range(min(feature_config.FEATURE_DIM, train_features.shape[1])):
            sample_data[f'feature_{j}'] = train_features[i][j]
        train_df_data.append(sample_data)

    train_df = pd.DataFrame(train_df_data)
    train_df.to_csv(train_csv_path, index=False)
    print(f"训练集特征已保存为CSV文件: {train_csv_path}")

    # 保存测试集CSV文件
    test_csv_path = os.path.join(feature_config.FEATURE_DIR, f"{fold_name}_test_densenet_features.csv")
    test_df_data = []
    for i in range(len(test_features)):
        sample_data = {
            'fold': fold_name,
            'split': 'test',
            'true_label': test_labels[i],
            'true_class': class_names[test_labels[i]],
            'pred_label': test_preds[i],
            'pred_class': class_names[test_preds[i]],
            'prob_class_0': test_probs[i][0],
            'prob_class_1': test_probs[i][1],
            'prob_class_2': test_probs[i][2],
            'prob_class_3': test_probs[i][3],
            'prob_class_4': test_probs[i][4],
        }
        # 添加特征
        for j in range(min(feature_config.FEATURE_DIM, test_features.shape[1])):
            sample_data[f'feature_{j}'] = test_features[i][j]
        test_df_data.append(sample_data)

    test_df = pd.DataFrame(test_df_data)
    test_df.to_csv(test_csv_path, index=False)
    print(f"测试集特征已保存为CSV文件: {test_csv_path}")

    return train_df, test_df


def extract_features_for_fold(fold_idx):
    """为单个折提取特征"""
    print(f"\n{'=' * 50}")
    print(f"开始提取第 {fold_idx + 1} 折的特征")
    print(f"{'=' * 50}")

    # 加载数据
    train_loader, test_loader, fold_name, class_names, samples = load_fold_data_for_features(fold_idx)

    # 初始化模型
    model = DenseNetFeatureExtractor(num_classes=feature_config.NUM_CLASSES)

    # 加载训练好的权重
    checkpoint_path = os.path.join(feature_config.CHECKPOINT_DIR, f"fold_{fold_idx + 1}_best_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到检查点文件 {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=feature_config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(feature_config.DEVICE)
    print(f"已加载模型权重: {checkpoint_path}")
    print(f"最佳准确率: {checkpoint['best_acc']:.4f}")

    # 提取训练集特征
    print("提取训练集特征...")
    train_features, train_labels, train_preds, train_probs, train_logits = extract_features(
        model, train_loader, feature_config.DEVICE
    )

    # 提取测试集特征
    print("提取测试集特征...")
    test_features, test_labels, test_preds, test_probs, test_logits = extract_features(
        model, test_loader, feature_config.DEVICE
    )

    print(f"特征提取完成:")
    print(f"  训练集特征形状: {train_features.shape}")
    print(f"  测试集特征形状: {test_features.shape}")
    print(f"  特征维度: {train_features.shape[1]}")

    # 分别保存训练集和测试集特征
    train_df, test_df = save_features_separate(
        fold_idx, fold_name, train_features, train_labels, train_preds, train_probs,
        test_features, test_labels, test_preds, test_probs, class_names
    )

    # 计算准确率
    train_acc = (train_preds == train_labels).mean()
    test_acc = (test_preds == test_labels).mean()

    print(f"准确率统计:")
    print(f"  训练集准确率: {train_acc:.4f}")
    print(f"  测试集准确率: {test_acc:.4f}")

    return {
        'fold': fold_idx + 1,
        'fold_name': fold_name,
        'train_features': train_features,
        'test_features': test_features,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_dataframe': train_df,
        'test_dataframe': test_df
    }


def main():
    """主函数：为所有折提取特征"""
    print(f"使用设备: {feature_config.DEVICE}")
    print(f"数据集路径: {feature_config.DATA_ROOT}")
    print(f"特征保存路径: {feature_config.FEATURE_DIR}")
    print(f"检查点路径: {feature_config.CHECKPOINT_DIR}\n")

    # 检查检查点文件是否存在
    print("检查检查点文件...")
    for fold_idx in range(5):
        checkpoint_path = os.path.join(feature_config.CHECKPOINT_DIR, f"fold_{fold_idx + 1}_best_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            print(f"  ✓ 找到检查点文件: fold_{fold_idx + 1}_best_checkpoint.pth")
        else:
            print(f"  ✗ 找不到检查点文件: fold_{fold_idx + 1}_best_checkpoint.pth")
            return

    # 为每个折提取特征
    all_results = []
    for fold_idx in range(5):
        result = extract_features_for_fold(fold_idx)
        if result is not None:
            all_results.append(result)

    # 汇总结果
    if all_results:
        print("\n" + "=" * 50)
        print("特征提取完成汇总")
        print("=" * 50)

        train_accs = [r['train_accuracy'] for r in all_results]
        test_accs = [r['test_accuracy'] for r in all_results]

        print(f"训练集平均准确率: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
        print(f"测试集平均准确率: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

        # 保存汇总信息
        summary = {
            'fold_results': [
                {
                    'fold': r['fold'],
                    'fold_name': r['fold_name'],
                    'train_accuracy': float(r['train_accuracy']),
                    'test_accuracy': float(r['test_accuracy']),
                    'train_samples': len(r['train_features']),
                    'test_samples': len(r['test_features'])
                } for r in all_results
            ],
            'mean_train_accuracy': float(np.mean(train_accs)),
            'std_train_accuracy': float(np.std(train_accs)),
            'mean_test_accuracy': float(np.mean(test_accs)),
            'std_test_accuracy': float(np.std(test_accs))
        }

        summary_path = os.path.join(feature_config.FEATURE_DIR, "feature_extraction_summary.json")
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        print(f"汇总结果已保存至: {summary_path}")

        print(f"\n所有特征文件已保存到: {feature_config.FEATURE_DIR}")
        print("生成的文件包括:")
        for fold_name in feature_config.FOLDS:
            print(f"  - {fold_name}_train_densenet_features.npz")
            print(f"  - {fold_name}_train_densenet_features.csv")
            print(f"  - {fold_name}_test_densenet_features.npz")
            print(f"  - {fold_name}_test_densenet_features.csv")


if __name__ == "__main__":
    main()