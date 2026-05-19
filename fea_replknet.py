import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import warnings
from replknet import create_RepLKNet31B

# 忽略警告信息
warnings.filterwarnings('ignore')


class FeatureExtractionConfig:
    """特征提取配置参数类"""
    # 数据路径
    DATA_ROOT = "./SIPaKMeD_5_fold"
    FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
    CLASSES = ["class_1", "class_2", "class_3", "class_4", "class_5"]
    NUM_CLASSES = 5

    # 特征提取参数
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输入输出路径
    CHECKPOINT_DIR = "./results/replknet_only/checkpoints"
    FEATURES_DIR = "./results/replknet_only/features"

    def __init__(self):
        self.create_directories()

    def create_directories(self):
        """创建输出目录"""
        if not os.path.exists(self.FEATURES_DIR):
            os.makedirs(self.FEATURES_DIR, exist_ok=True)
            print(f"创建目录: {self.FEATURES_DIR}")


config = FeatureExtractionConfig()


class RepLKNetWrapper(nn.Module):
    """RepLKNet包装器 - 与训练时相同的结构"""

    def __init__(self, num_classes=5):
        super(RepLKNetWrapper, self).__init__()
        # 使用与训练时相同的结构
        self.replknet = create_RepLKNet31B(
            num_classes=num_classes,
            use_checkpoint=False,
            small_kernel_merged=False
        )

        # 增强分类头（与训练时相同）
        original_head = self.replknet.head
        self.replknet.head = nn.Sequential(
            nn.Dropout(0.5),
            original_head
        )

    def forward(self, x):
        return self.replknet(x)

    def forward_features(self, x):
        """提取特征的方法"""
        return self.replknet.forward_features(x)


def get_feature_transforms():
    """特征提取用的数据预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_fold_datasets(fold_idx):
    """加载指定折的训练集和测试集"""
    current_fold = config.FOLDS[fold_idx]

    # 训练集
    train_dir = os.path.join(config.DATA_ROOT, current_fold, "train")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=get_feature_transforms())

    # 测试集
    test_dir = os.path.join(config.DATA_ROOT, current_fold, "test")
    test_dataset = datasets.ImageFolder(root=test_dir, transform=get_feature_transforms())

    print(f"Fold {fold_idx + 1} ({current_fold}):")
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  测试集: {len(test_dataset)} 个样本")

    return train_dataset, test_dataset, current_fold


def load_trained_model(checkpoint_path, num_classes=5):
    """从checkpoint加载训练好的模型"""
    # 使用与训练时相同的包装器类
    model = RepLKNetWrapper(num_classes=num_classes).to(config.DEVICE)

    # 加载checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

        # 检查checkpoint中保存的模型状态字典
        model_state_dict = checkpoint["model_state_dict"]

        # 直接加载权重
        model.load_state_dict(model_state_dict)
        print(f"成功加载模型权重: {checkpoint_path}")

        # 打印模型信息
        print(f"模型类别数: {num_classes}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print(f"警告: 找不到checkpoint文件 {checkpoint_path}")
        return None

    model.eval()
    return model


def extract_features_from_dataset(model, dataset, dataset_name, fold_idx, split_name):
    """从数据集提取特征"""
    data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    all_features = []
    all_labels = []
    all_image_paths = []

    print(f"正在提取 {dataset_name} 特征...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(config.DEVICE)

            # 使用forward_features方法提取特征
            features = model.forward_features(images)  # [B, C, H, W]

            # 全局平均池化得到 [B, C]
            features_pooled = torch.mean(features, dim=[2, 3])

            all_features.append(features_pooled.cpu().numpy())
            all_labels.append(labels.numpy())

            # 获取当前批次的图像路径
            start_idx = batch_idx * config.BATCH_SIZE
            end_idx = start_idx + len(images)
            batch_paths = [dataset.samples[i][0] for i in range(start_idx, min(end_idx, len(dataset)))]
            all_image_paths.extend(batch_paths)

            # 打印进度
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(data_loader):
                print(f"Fold {fold_idx + 1} | {dataset_name} | 批次: {batch_idx + 1}/{len(data_loader)}")

    # 合并所有批次
    if all_features:
        features_array = np.vstack(all_features)
        labels_array = np.concatenate(all_labels)
        print(f"{dataset_name} 特征提取完成: {features_array.shape}")
        return features_array, labels_array, all_image_paths
    else:
        print(f"{dataset_name} 没有提取到特征")
        return None, None, None


def save_split_features(fold_idx, split_name, features, labels, image_paths, fold_name):
    """保存单个分割（训练集或测试集）的特征"""

    # 1. 保存为 .npz 文件
    npz_filename = f"fold_{fold_idx + 1}_{split_name}_replk_features.npz"
    npz_path = os.path.join(config.FEATURES_DIR, npz_filename)

    np.savez(npz_path,
             features=features,
             labels=labels,
             image_paths=image_paths,
             fold_name=fold_name,
             split_name=split_name)

    # 2. 保存为 .csv 文件
    csv_filename = f"fold_{fold_idx + 1}_{split_name}_replk_features.csv"
    csv_path = os.path.join(config.FEATURES_DIR, csv_filename)

    # 创建DataFrame
    feature_columns = [f'feat_{i:04d}' for i in range(features.shape[1])]

    df = pd.DataFrame(features, columns=feature_columns)

    # 添加元数据列
    df['label'] = labels
    df['image_path'] = image_paths
    df['fold'] = fold_name
    df['split'] = split_name
    df['class_name'] = [config.CLASSES[label] for label in labels]

    # 添加样本ID
    df['sample_id'] = [f"{fold_name}_{split_name}_{i:04d}" for i in range(len(features))]

    # 保存CSV
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"  {split_name}特征文件已保存:")
    print(f"    NPZ文件: {npz_path}")
    print(f"    CSV文件: {csv_path}")
    print(f"    特征维度: {features.shape}")
    print(f"    样本数量: {len(features)}")


def extract_fold_features(fold_idx):
    """提取单个折的特征"""
    print(f"\n{'=' * 60}")
    print(f"开始提取第 {fold_idx + 1} 折的特征")
    print(f"{'=' * 60}")

    try:
        # 1. 加载数据
        train_dataset, test_dataset, fold_name = load_fold_datasets(fold_idx)

        # 2. 加载模型
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"fold_{fold_idx + 1}_best_checkpoint.pth")
        model = load_trained_model(checkpoint_path)
        if model is None:
            return False

        # 3. 分别提取训练集和测试集特征
        print(f"\n提取训练集特征...")
        train_features, train_labels, train_paths = extract_features_from_dataset(
            model, train_dataset, "训练集", fold_idx, "train"
        )

        print(f"\n提取测试集特征...")
        test_features, test_labels, test_paths = extract_features_from_dataset(
            model, test_dataset, "测试集", fold_idx, "test"
        )

        if train_features is None or test_features is None:
            print("特征提取失败")
            return False

        # 4. 分别保存训练集和测试集特征
        print(f"\n保存特征文件...")
        save_split_features(fold_idx, "train", train_features, train_labels, train_paths, fold_name)
        save_split_features(fold_idx, "test", test_features, test_labels, test_paths, fold_name)

        print(f"\nFold {fold_idx + 1} 特征汇总:")
        print(f"  训练集: {train_features.shape} - {len(train_features)} 个样本")
        print(f"  测试集: {test_features.shape} - {len(test_features)} 个样本")
        print(f"  总样本数: {len(train_features) + len(test_features)}")
        print(f"  特征维度: {train_features.shape[1]}")

        return True

    except Exception as e:
        print(f"Fold {fold_idx + 1} 特征提取出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def verify_checkpoints():
    """验证所有checkpoint文件是否存在"""
    print("检查checkpoint文件...")
    missing_checkpoints = []

    for fold_idx in range(5):
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"fold_{fold_idx + 1}_best_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            print(f"  ✓ Fold {fold_idx + 1}: {checkpoint_path}")
        else:
            print(f"  ✗ Fold {fold_idx + 1}: 文件不存在")
            missing_checkpoints.append(fold_idx + 1)

    return missing_checkpoints


def extract_all_folds(folds_to_extract=None):
    """提取所有折的特征"""
    print("=" * 60)
    print("RepLKNet 特征提取工具")
    print("=" * 60)

    print(f"设备: {config.DEVICE}")
    print(f"Checkpoint目录: {config.CHECKPOINT_DIR}")
    print(f"特征保存目录: {config.FEATURES_DIR}")

    # 验证checkpoint文件
    missing_checkpoints = verify_checkpoints()
    if missing_checkpoints:
        print(f"\n警告: 以下fold的checkpoint文件缺失: {missing_checkpoints}")
        response = input("是否继续提取其他fold的特征? (y/n): ")
        if response.lower() != 'y':
            return

    # 确定要提取的fold
    if folds_to_extract is None:
        folds_to_extract = list(range(5))
        print(f"\n将提取所有5个fold的特征")
    else:
        folds_to_extract = [f for f in folds_to_extract if 0 <= f <= 4]
        if not folds_to_extract:
            print("错误: 没有有效的fold索引！")
            return
        print(f"\n将提取fold: {[f + 1 for f in folds_to_extract]}的特征")

    # 提取特征
    successful_folds = []
    for fold_idx in folds_to_extract:
        try:
            success = extract_fold_features(fold_idx)
            if success:
                successful_folds.append(fold_idx + 1)
                print(f"✓ Fold {fold_idx + 1} 特征提取完成\n")
            else:
                print(f"✗ Fold {fold_idx + 1} 特征提取失败\n")
        except Exception as e:
            print(f"✗ Fold {fold_idx + 1} 特征提取出错: {e}\n")

    # 汇总结果
    print("=" * 60)
    print("特征提取完成汇总")
    print("=" * 60)
    if successful_folds:
        print(f"成功提取的fold: {successful_folds}")
        print(f"特征文件保存在: {config.FEATURES_DIR}")

        # 显示特征文件信息
        print(f"\n生成的特征文件:")
        for fold in successful_folds:
            train_npz = os.path.join(config.FEATURES_DIR, f"fold_{fold}_train_replk_features.npz")
            test_npz = os.path.join(config.FEATURES_DIR, f"fold_{fold}_test_replk_features.npz")

            if os.path.exists(train_npz):
                data = np.load(train_npz)
                print(f"  Fold {fold} 训练集: {data['features'].shape}")
            if os.path.exists(test_npz):
                data = np.load(test_npz)
                print(f"  Fold {fold} 测试集: {data['features'].shape}")
    else:
        print("没有成功提取任何fold的特征")


# 加载特征文件的辅助函数
def load_features(fold_idx, split_name):
    """加载指定fold和分割的特征"""
    npz_path = os.path.join(config.FEATURES_DIR, f"fold_{fold_idx}_{split_name}_replk_features.npz")
    csv_path = os.path.join(config.FEATURES_DIR, f"fold_{fold_idx}_{split_name}_replk_features.csv")

    if not os.path.exists(npz_path):
        print(f"错误: 找不到特征文件 {npz_path}")
        return None

    # 加载.npz文件
    data = np.load(npz_path, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    image_paths = data['image_paths']

    # 加载.csv文件
    df = pd.read_csv(csv_path)

    print(f"Fold {fold_idx} {split_name} 特征加载成功:")
    print(f"  特征维度: {features.shape}")
    print(f"  样本数量: {len(features)}")

    return {
        'features': features,
        'labels': labels,
        'image_paths': image_paths,
        'dataframe': df
    }


if __name__ == "__main__":
    # 使用方法：

    # 1. 提取所有fold的特征
    extract_all_folds()

    # 2. 提取指定fold的特征
    # extract_all_folds(folds_to_extract=[0, 2, 4])  # 提取fold 1, 3, 5

    # 3. 提取单个fold的特征
    # extract_all_folds(folds_to_extract=[0])  # 只提取fold 1

    # 4. 加载特征进行分析
    # train_features = load_features(1, "train")  # 加载fold 1的训练集特征
    # test_features = load_features(1, "test")    # 加载fold 1的测试集特征