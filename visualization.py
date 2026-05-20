#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
融合模型可视化脚本（论文高质量版）
- 高分辨率输出 (600 DPI)
- 学术字体 (Times New Roman)
- 优化的颜色、点大小、图例和布局
- 支持 PDF 矢量图（可选）
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms, models
from PIL import Image
import cv2
import matplotlib

# ======================== 高质量 matplotlib 全局配置 ========================
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.manifold import TSNE
import pandas as pd

try:
    from replknet import create_RepLKNet31B
except ImportError:
    print("错误：未找到 replknet 模块，请将 replknet.py 放在当前目录下。")
    sys.exit(1)


# ======================== 全局配置 ========================
class Config:
    REPLK_CKPT = "./results/replknet_only/checkpoints/fold_1_best_checkpoint.pth"
    DENSE_CKPT = "./results/densenet_only/checkpoints/fold_1_best_checkpoint.pth"
    FUSION_CKPT = "./classicify2/feature_fusion_5/checkpoints/fold_1_gated_feature_fusion_best.pth"
    DATA_ROOT = "./SIPaKMeD_5_fold"
    FEATURE_DIR = "./features"
    LOG_DIR = "./classicify2/feature_fusion_5/logs"
    FIG_DIR = "./classicify2/feature_fusion_5/figures"
    OUTPUT_DIR = "./fusion_visualizations000"

    NUM_CLASSES = 5
    CLASS_NAMES = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic",
                   "im_Parabasal", "im_Superficial-Intermediate"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    IMG_SIZE = 224
    SAMPLES_PER_CLASS = 3
    TSNE_SAMPLES = 500

    # 新增：高质量图片参数
    FIG_DPI = 600                     # 输出分辨率（印刷级）
    SAVE_FORMAT = 'png'               # 可选 'png', 'pdf', 'svg'；若选 pdf 需调整后端
    FIG_SIZE_FACTOR = 1.2             # 整体尺寸放大系数


# 高质量 matplotlib 参数设置
rcParams.update({
    'font.size': 12,                  # 全局字体大小
    'axes.titlesize': 14,            # 子图标题大小
    'axes.labelsize': 12,            # 坐标轴标签大小
    'legend.fontsize': 11,           # 图例字体大小
    'xtick.labelsize': 11,           # X轴刻度大小
    'ytick.labelsize': 11,           # Y轴刻度大小
    'figure.dpi': Config.FIG_DPI,    # 显示 DPI（仅影响屏幕显示）
    'savefig.dpi': Config.FIG_DPI,   # 保存 DPI
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'image.cmap': 'jet',             # 热力图颜色映射
    'axes.grid': False,
    'font.family': 'serif',          # 衬线字体，适合学术论文
    'font.serif': ['Times New Roman'],  # 常用学术字体
    'lines.linewidth': 2.0,          # 线条加粗
    'lines.markersize': 4,           # 标记点大小
    'patch.linewidth': 1.2,          # 箱线图边框宽度
    'grid.alpha': 0.3,               # 网格透明度
})


# ======================== 模型定义（保持不变） ========================
class RepLKNetWrapper(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.replknet = create_RepLKNet31B(num_classes=num_classes, use_checkpoint=False, small_kernel_merged=False)
        original_head = self.replknet.head
        self.replknet.head = nn.Sequential(nn.Dropout(0.5), original_head)

    def forward(self, x):
        return self.replknet(x)

    def forward_features(self, x):
        return self.replknet.forward_features(x)


class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        densenet = models.densenet121(pretrained=False)
        self.features = densenet.features
        original_in_features = densenet.classifier.in_features
        self.classifier = nn.Sequential(
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
        features = self.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits, out


class GatedFeatureFusionMLP(nn.Module):
    def __init__(self, replk_dim=1024, densenet_dim=1024, num_classes=5, hidden_dim=512):
        super().__init__()
        self.replk_dim = replk_dim
        self.densenet_dim = densenet_dim
        self.gate_network = nn.Sequential(
            nn.Linear(replk_dim + densenet_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, replk_dim),
            nn.Sigmoid()
        )
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
        if replk_dim != densenet_dim:
            if replk_dim > densenet_dim:
                self.densenet_proj = nn.Linear(densenet_dim, replk_dim)
            else:
                self.replk_proj = nn.Linear(replk_dim, densenet_dim)
        else:
            self.densenet_proj = None
            self.replk_proj = None

    def forward(self, replk_features, densenet_features):
        concat = torch.cat([replk_features, densenet_features], dim=1)
        g = self.gate_network(concat)
        if self.replk_dim == self.densenet_dim:
            fused = g * replk_features + (1 - g) * densenet_features
        else:
            if self.replk_dim > self.densenet_dim:
                densenet_proj = self.densenet_proj(densenet_features)
                fused = g * replk_features + (1 - g) * densenet_proj
            else:
                replk_proj = self.replk_proj(replk_features)
                fused = g * replk_proj + (1 - g) * densenet_features
        out = self.classifier(fused)
        return out, g


class EndToEndGatedFusionClassifier(nn.Module):
    def __init__(self, replk_model, densenet_model, fusion_model):
        super().__init__()
        self.replk_extractor = replk_model
        self.densenet_extractor = densenet_model
        self.fusion = fusion_model

    def forward(self, x):
        feat_map = self.replk_extractor.forward_features(x)
        replk_feat = torch.mean(feat_map, dim=[2, 3])
        _, densenet_feat = self.densenet_extractor(x)
        logits, g = self.fusion(replk_feat, densenet_feat)
        return logits, g


# ======================== 模型加载（保持不变） ========================
def load_models():
    print("加载模型权重...")
    replk_extractor = RepLKNetWrapper(num_classes=Config.NUM_CLASSES)
    densenet_extractor = DenseNetFeatureExtractor(num_classes=Config.NUM_CLASSES)
    fusion_model = GatedFeatureFusionMLP(replk_dim=1024, densenet_dim=1024, num_classes=Config.NUM_CLASSES)

    for name, ckpt_path, model in [("RepLKNet", Config.REPLK_CKPT, replk_extractor),
                                   ("DenseNet", Config.DENSE_CKPT, densenet_extractor),
                                   ("FusionMLP", Config.FUSION_CKPT, fusion_model)]:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{name} checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location='cpu')
        state_dict = state.get('model_state_dict', state)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {name} from {ckpt_path}")

    model = EndToEndGatedFusionClassifier(replk_extractor, densenet_extractor, fusion_model)
    model = model.to(Config.DEVICE)
    model.eval()
    return model, replk_extractor, densenet_extractor, fusion_model


# ======================== 数据加载（保持不变） ========================
transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class CellImageDataset(Dataset):
    def __init__(self, root, fold, split):
        self.dir = os.path.join(root, fold, split)
        self.imgs, self.labels, self.class_img = [], [], {i: [] for i in range(5)}
        for i, cname in enumerate(Config.CLASS_NAMES):
            cdir = os.path.join(self.dir, cname)
            if not os.path.isdir(cdir): continue
            for f in os.listdir(cdir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.imgs.append(os.path.join(cdir, f))
                    self.labels.append(i)
                    self.class_img[i].append(os.path.join(cdir, f))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        return transform(img), self.labels[idx]

    def get_samples_per_class(self, n=2):
        sel = []
        for c in range(5):
            pool = self.class_img[c]
            if pool:
                chosen = random.sample(pool, min(n, len(pool)))
                for p in chosen:
                    sel.append((p, c))
        return sel


def load_feature_loader():
    fold_str = "fold_1"
    replk_path = os.path.join(Config.FEATURE_DIR, f"{fold_str}_test_replk_features.npz")
    dense_path = os.path.join(Config.FEATURE_DIR, f"{fold_str}_test_densenet_features.npz")
    if not os.path.exists(replk_path) or not os.path.exists(dense_path):
        raise FileNotFoundError("特征文件缺失，请先运行特征提取脚本。")
    replk = torch.FloatTensor(np.load(replk_path)['features'])
    dense = torch.FloatTensor(np.load(dense_path)['features'])
    labels = torch.LongTensor(np.load(replk_path)['labels'])
    return DataLoader(TensorDataset(replk, dense, labels), batch_size=Config.BATCH_SIZE, shuffle=False)


# ======================== Grad‑CAM 工具（保持不变） ========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.act, self.grad = None, None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, m, i, o):
        self.act = o.detach()

    def _bwd_hook(self, m, i, o):
        self.grad = o[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        weights = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam.squeeze().cpu().numpy()


def find_last_conv_layer(model):
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("未找到卷积层，无法生成热图")
    return last_conv


# ======================== 方案一：融合注意力热图（优化） ========================
def generate_fusion_heatmaps(image_path, model, replk_ext, dense_ext, true_label, save_path, report_file):
    img_pil = Image.open(image_path).convert('RGB')
    orig_np = np.array(img_pil.resize((Config.IMG_SIZE, Config.IMG_SIZE)))
    input_t = transform(img_pil).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        logits, g = model(input_t)
        pred_cls = logits.argmax(dim=1).item()
        g_mean = g.mean().item()
        g_values = g.mean(dim=1).cpu().numpy()[0]

    target_dense = dense_ext.features.denseblock4
    target_replk = find_last_conv_layer(replk_ext)

    def get_cam(model, input_t, target_layer):
        gradcam = GradCAM(model, target_layer)
        return gradcam.generate(input_t)

    heat_rep = get_cam(replk_ext, input_t, target_replk)
    heat_dense = get_cam(dense_ext, input_t, target_dense)

    avg_rep = heat_rep.mean()
    avg_dense = heat_dense.mean()

    heat_rep = cv2.resize(heat_rep, (Config.IMG_SIZE, Config.IMG_SIZE))
    heat_dense = cv2.resize(heat_dense, (Config.IMG_SIZE, Config.IMG_SIZE))
    heat_fused = g_mean * heat_rep + (1 - g_mean) * heat_dense
    heat_fused = np.clip(heat_fused, 0, 1)
    avg_fused = heat_fused.mean()

    with open(report_file, 'a', encoding='utf-8') as f:
        f.write(
            f"\n[热力图] {os.path.basename(image_path)}  True:{Config.CLASS_NAMES[true_label]}  Pred:{Config.CLASS_NAMES[pred_cls]}\n")
        f.write(f"  门控权重 g (RepLKNet偏好): {g_mean:.4f}\n")
        f.write(f"  平均激活强度 - RepLKNet: {avg_rep:.4f}  DenseNet: {avg_dense:.4f}  融合后: {avg_fused:.4f}\n")

    def overlay(img, h, ax):
        # 使用更高质量的颜色映射（论文常用 jet 或 inferno）
        heatmap = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        im = ax.imshow(heatmap_rgb, alpha=0.45)   # 稍微提高透明度，使原图更清晰
        return im

    # 调整子图尺寸，适应放大因子
    fig, axes = plt.subplots(1, 4, figsize=(18 * Config.FIG_SIZE_FACTOR, 5 * Config.FIG_SIZE_FACTOR))
    axes[0].imshow(orig_np)
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    im1 = overlay(orig_np, heat_rep, axes[1])
    axes[1].set_title('RepLKNet', fontsize=14)
    axes[1].axis('off')
    im2 = overlay(orig_np, heat_dense, axes[2])
    axes[2].set_title('DenseNet', fontsize=14)
    axes[2].axis('off')
    im3 = overlay(orig_np, heat_fused, axes[3])
    axes[3].set_title(f'Fused (g={g_mean:.2f})', fontsize=14)
    axes[3].axis('off')

    # 添加颜色条，并设置字体
    cbar = fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label('Activation Intensity', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.suptitle(f'True: {Config.CLASS_NAMES[true_label]}   Pred: {Config.CLASS_NAMES[pred_cls]}', fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()


def run_plan1(model, replk_ext, dense_ext, report_file):
    print("\n===== 方案一：融合注意力热图 =====")
    with open(report_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 50 + "\n方案一：融合注意力热图\n" + "=" * 50 + "\n")

    dataset = CellImageDataset(Config.DATA_ROOT, "fold_1", "test")
    samples = dataset.get_samples_per_class(Config.SAMPLES_PER_CLASS)
    for idx, (img_path, label) in enumerate(samples):
        cname = Config.CLASS_NAMES[label]
        save_p = os.path.join(Config.OUTPUT_DIR, 'heatmaps', f'heatmap_{cname}_{idx}.{Config.SAVE_FORMAT}')
        generate_fusion_heatmaps(img_path, model, replk_ext, dense_ext, label, save_p, report_file)


# ======================== 方案二：门控权重分布（优化） ========================
def run_plan2(fusion_mlp, report_file):
    print("\n===== 方案二：门控权重分布 =====")
    loader = load_feature_loader()
    fusion_mlp.eval()
    g_list, lbl_list = [], []
    with torch.no_grad():
        for r, d, l in loader:
            r, d = r.to(Config.DEVICE), d.to(Config.DEVICE)
            _, g = fusion_mlp(r, d)
            g_mean = g.mean(dim=1).cpu().numpy()
            g_list.extend(g_mean)
            lbl_list.extend(l.numpy())
    g_arr = np.array(g_list)
    lbl_arr = np.array(lbl_list)
    g_dense = 1 - g_arr

    with open(report_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 50 + "\n方案二：门控权重分布统计\n" + "=" * 50 + "\n")
        f.write(f"总样本数: {len(g_arr)}\n")
        f.write(
            f"{'类别':<30} {'RepLKNet均值':<15} {'DenseNet均值':<15} {'RepLKNet标准差':<15} {'DenseNet标准差':<15}\n")
        for c in range(5):
            mask = lbl_arr == c
            if mask.sum() == 0:
                continue
            r_mean = g_arr[mask].mean()
            d_mean = g_dense[mask].mean()
            r_std = g_arr[mask].std()
            d_std = g_dense[mask].std()
            f.write(f"{Config.CLASS_NAMES[c]:<30} {r_mean:<15.4f} {d_mean:<15.4f} {r_std:<15.4f} {d_std:<15.4f}\n")

        f.write("\n详细百分位数 (RepLKNet偏好):\n")
        f.write(f"{'类别':<30} {'Q1':<10} {'中位数':<10} {'Q3':<10}\n")
        for c in range(5):
            mask = lbl_arr == c
            vals = g_arr[mask]
            if len(vals):
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                f.write(f"{Config.CLASS_NAMES[c]:<30} {q1:<10.4f} {med:<10.4f} {q3:<10.4f}\n")

    plt.figure(figsize=(12 * Config.FIG_SIZE_FACTOR, 5 * Config.FIG_SIZE_FACTOR))
    positions_rep = np.arange(5) * 2
    positions_dense = positions_rep + 0.7
    data_rep = [g_arr[lbl_arr == c] for c in range(5)]
    data_dense = [g_dense[lbl_arr == c] for c in range(5)]

    # 箱线图优化：增加线宽、填充色透明度、离群点样式
    bp1 = plt.boxplot(data_rep, positions=positions_rep, widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', linewidth=1.5),
                      whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=1.5, color='darkblue'),
                      flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))
    bp2 = plt.boxplot(data_dense, positions=positions_dense, widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='lightcoral', linewidth=1.5),
                      whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=1.5, color='darkred'),
                      flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))

    plt.xticks(positions_rep + 0.35, Config.CLASS_NAMES, rotation=45, ha='right', fontsize=12)
    plt.ylabel('Gating Weight', fontsize=13)
    plt.title('Distribution of Gating Weights per Class', fontsize=15)
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['RepLKNet preference', 'DenseNet preference'],
               loc='upper right', fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    save_p = os.path.join(Config.OUTPUT_DIR, f'gating_weight_distribution.{Config.SAVE_FORMAT}')
    plt.savefig(save_p, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"门控权重分布图已保存: {save_p}")


# ======================== 方案三：t‑SNE 特征对比（优化） ========================
def run_plan3(model, replk_ext, dense_ext, fusion_mlp, report_file):
    print("\n===== 方案三：t‑SNE 特征对比 =====")
    dataset = CellImageDataset(Config.DATA_ROOT, "fold_1", "test")
    n_samples = min(Config.TSNE_SAMPLES, len(dataset))

    all_labels = np.array(dataset.labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    ratios = counts / counts.sum()
    per_class = np.maximum(np.round(ratios * n_samples).astype(int), 1)
    diff = n_samples - per_class.sum()
    for i in range(abs(diff)):
        if diff > 0:
            per_class[i % len(unique)] += 1
        elif per_class[i % len(unique)] > 1:
            per_class[i % len(unique)] -= 1

    indices = []
    for cls, n in zip(unique, per_class):
        cls_idx = np.where(all_labels == cls)[0]
        chosen = np.random.choice(cls_idx, size=n, replace=False)
        indices.extend(chosen)
    indices = np.array(indices)
    np.random.shuffle(indices)

    loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=Config.BATCH_SIZE, shuffle=False)

    replk_feats, dense_feats, fused_feats, labels = [], [], [], []
    replk_ext.eval()
    dense_ext.eval()
    fusion_mlp.eval()
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(Config.DEVICE)
            r_map = replk_ext.forward_features(imgs)
            r_vec = torch.mean(r_map, dim=[2, 3])
            _, d_vec = dense_ext(imgs)
            _, g = fusion_mlp(r_vec, d_vec)
            f_vec = g * r_vec + (1 - g) * d_vec
            replk_feats.append(r_vec.cpu().numpy())
            dense_feats.append(d_vec.cpu().numpy())
            fused_feats.append(f_vec.cpu().numpy())
            labels.append(labs.numpy())
    r_all = np.concatenate(replk_feats)
    d_all = np.concatenate(dense_feats)
    f_all = np.concatenate(fused_feats)
    lbl_all = np.concatenate(labels)

    print("Computing t‑SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    r_2d = tsne.fit_transform(r_all)
    d_2d = tsne.fit_transform(d_all)
    f_2d = tsne.fit_transform(f_all)

    # 保存 CSV
    df_tsne = pd.DataFrame({
        'replk_x': r_2d[:, 0],
        'replk_y': r_2d[:, 1],
        'densenet_x': d_2d[:, 0],
        'densenet_y': d_2d[:, 1],
        'fusion_x': f_2d[:, 0],
        'fusion_y': f_2d[:, 1],
        'label': lbl_all
    })
    csv_path = os.path.join(Config.OUTPUT_DIR, 'tsne_coords.csv')
    df_tsne.to_csv(csv_path, index=False)
    print(f"t‑SNE 坐标已保存为 CSV: {csv_path}")

    with open(report_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 50 + "\n方案三：t‑SNE 特征对比\n" + "=" * 50 + "\n")
        f.write(f"抽取样本总数: {n_samples}\n")
        for cls, n in zip(unique, per_class):
            f.write(f"  {Config.CLASS_NAMES[cls]}: {n} 张\n")
        f.write(f"二维坐标已保存至: {csv_path}\n")

    # 绘图：增大点尺寸、透明度、图例字体
    fig, axes = plt.subplots(1, 3, figsize=(18 * Config.FIG_SIZE_FACTOR, 5 * Config.FIG_SIZE_FACTOR))
    point_size = 15   # 适合印刷的点大小
    alpha_val = 0.7

    for ax, (data, title) in zip(axes, [(r_2d, 'RepLKNet'), (d_2d, 'DenseNet'), (f_2d, 'Gated Fusion')]):
        for c in range(5):
            ax.scatter(data[lbl_all == c, 0], data[lbl_all == c, 1],
                       label=Config.CLASS_NAMES[c], alpha=alpha_val, s=point_size, edgecolors='none')
        ax.set_title(title, fontsize=15)
        ax.legend(markerscale=2, fontsize=10, loc='best')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    plt.tight_layout()
    save_p = os.path.join(Config.OUTPUT_DIR, f'fusion_tsne.{Config.SAVE_FORMAT}')
    plt.savefig(save_p, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"t‑SNE 图已保存: {save_p}")


# ======================== 方案四：优化图表（含混淆矩阵重构） ========================
def run_plan4(report_file):
    print("\n===== 方案四：优化训练曲线与混淆矩阵 =====")
    log_path = os.path.join(Config.LOG_DIR, "fold_1_gated_feature_fusion_log.json")
    if not os.path.exists(log_path):
        print("未找到训练日志，跳过曲线重绘。")
        with open(report_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 50 + "\n方案四：训练曲线\n" + "=" * 50 + "\n")
            f.write("未找到训练日志文件。\n")
        return

    with open(log_path) as f:
        log = json.load(f)
    epochs = range(1, len(log['train_losses']) + 1)

    final_train_acc = log['train_accs'][-1] if log['train_accs'] else None
    final_test_acc = log['test_accs'][-1] if log['test_accs'] else None
    final_train_loss = log['train_losses'][-1] if log['train_losses'] else None
    final_test_loss = log['test_losses'][-1] if log['test_losses'] else None
    best_epoch = np.argmax(log['test_accs']) + 1 if log['test_accs'] else None
    best_test_acc = max(log['test_accs']) if log['test_accs'] else None

    with open(report_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 50 + "\n方案四：训练曲线与混淆矩阵\n" + "=" * 50 + "\n")
        f.write(f"总 Epoch 数: {len(epochs)}\n")
        if final_train_acc is not None:
            f.write(f"最终训练准确率: {final_train_acc:.4f}\n")
            f.write(f"最终测试准确率: {final_test_acc:.4f}\n")
            f.write(f"最佳测试准确率: {best_test_acc:.4f} (Epoch {best_epoch})\n")
            f.write(f"最终训练损失: {final_train_loss:.4f}\n")
            f.write(f"最终测试损失: {final_test_loss:.4f}\n")

    # 训练曲线优化：线宽、标记、网格
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14 * Config.FIG_SIZE_FACTOR, 5 * Config.FIG_SIZE_FACTOR))
    ax1.plot(epochs, log['train_losses'], label='Train Loss', lw=2.5, marker='o', markersize=4, markevery=5)
    ax1.plot(epochs, log['test_losses'], label='Test Loss', lw=2.5, marker='s', markersize=4, markevery=5)
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Loss', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=11)

    ax2.plot(epochs, log['train_accs'], label='Train Acc', lw=2.5, marker='o', markersize=4, markevery=5)
    ax2.plot(epochs, log['test_accs'], label='Test Acc', lw=2.5, marker='s', markersize=4, markevery=5)
    ax2.set_xlabel('Epoch', fontsize=13)
    ax2.set_ylabel('Accuracy', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=11)

    plt.tight_layout()
    save_p = os.path.join(Config.OUTPUT_DIR, f'improved_metrics_fold1.{Config.SAVE_FORMAT}')
    plt.savefig(save_p, dpi=Config.FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"优化曲线已保存: {save_p}")

    # 尝试重新生成高质量混淆矩阵（如果存在预测结果）
    # 此处假设可以从日志或特征中获取测试集预测；若无则复制原有图片
    # 为了提高论文质量，建议在训练脚本中保存混淆矩阵数据；这里提供一个简单版：若已有原始混淆矩阵图片，重新用 matplotlib 绘制。
    cm_orig = os.path.join(Config.FIG_DIR, "fold_1_gated_feature_fusion_confusion_matrix.png")
    if os.path.exists(cm_orig):
        # 为了保持高质量，可以复制原始图片（但原始可能分辨率低）
        cm_new = os.path.join(Config.OUTPUT_DIR, f'improved_confmat_fold1.{Config.SAVE_FORMAT}')
        # 尝试使用 matplotlib 重新渲染（如果有原始数据），避免直接复制
        # 如果没有保存混淆矩阵数值，则只能复制；这里提供复制备选
        import shutil
        shutil.copy2(cm_orig, cm_new)
        print(f"混淆矩阵已复制（如需重绘，请在训练过程中保存混淆矩阵数据）: {cm_new}")
    else:
        print("未找到原有混淆矩阵图片。")


# ======================== 主流程 ========================
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    report_file = os.path.join(Config.OUTPUT_DIR, 'summary_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("融合模型可视化数值报告（论文高质量版）\n")
        f.write(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    model, replk_ext, dense_ext, fusion_mlp = load_models()
    run_plan1(model, replk_ext, dense_ext, report_file)
    run_plan2(fusion_mlp, report_file)
    run_plan3(model, replk_ext, dense_ext, fusion_mlp, report_file)
    run_plan4(report_file)

    print("\n全部可视化完成！结果保存在:", Config.OUTPUT_DIR)
    print(f"数值报告: {report_file}")


if __name__ == "__main__":
    main()