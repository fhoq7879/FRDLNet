import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
import json
import traceback
from pathlib import Path
import shutil
import argparse

warnings.filterwarnings('ignore')

# 注意：根据你的实际路径调整导入
import configs
import backbone
from data.datamgr import SetDataManager
from methods.relationnet import RelationNet
from methods.cosine_batch import CosineBatch
from methods.ournet import OurNet  # 确保指向修改后的ournet.py
from io_utils import parse_args

# ===================== 全局配置（4类纤维） =====================
CLASS_NAMES = ['Wool', 'Cashmere', 'Modified Wool', 'Modified Cashmere']
CLASS_COLORS = ['#626E98', '#E37366', '#A782DE', '#92314D']
SHOT_COLORS = ['#3498db', '#e74c3c']  # 1-shot:蓝色, 5-shot:红色
GLOBAL_CLASS_MAP = {
    'Wool': 0,
    'Cashmere': 1,
    'Modified Wool': 2,
    'Modified Cashmere': 3
}

# ===================== 骨干网络注册表 =====================
BACKBONE_REGISTRY = {
    'conv4np': backbone.Conv4NP,
    'resnet12': backbone.ResNet12,
}

# ===================== 箱线图配置 =====================
BOXPLOT_CONFIG = {
    'figsize': (10, 6),
    'fontsize': 12,
    'title_fontsize': 14,
    'xlabel': 'Shot Setting',
    'ylabel': 'Validation Accuracy',
    'title': '1-shot vs 5-shot Classification Accuracy Distribution',
    'whis': 1.5,
    'width': 0.4,
    'showfliers': True,
    'showmeans': True,
}


def get_backbone(backbone_name):
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(f"无效骨干网络: {backbone_name}")
    return BACKBONE_REGISTRY[backbone_name]


# ===================== 工具函数 =====================
def set_seed(seed=10):
    """设置随机种子，保证实验可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_checkpoint_dir(params, shot):
    """创建检查点目录，按shot数分类"""
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 添加shot信息到目录名
    checkpoint_dir = (
        f"{configs.save_dir}/checkpoints/{params.dataset}/"
        f"{params.backbone}_{params.method}_{params.train_n_way}way_{shot}shot_{time_str}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"✅ Checkpoint目录已创建: {checkpoint_dir}")
    return checkpoint_dir


def load_data(params, shot):
    """加载训练/验证数据（适配4类纤维）"""
    base_file = os.path.join(configs.data_dir[params.dataset], 'base.json')
    val_file = os.path.join(configs.data_dir[params.dataset], 'val.json')

    # 校验数据文件
    for f in [base_file, val_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"数据文件不存在: {f}")

    image_size = 84
    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))

    print(f"📊 数据加载配置: {params.train_n_way}-way {shot}-shot, n_query={n_query}")

    # 数据加载器配置（关闭多线程，避免Windows报错）
    train_few_shot_params = dict(n_way=params.train_n_way, n_support=shot, n_query=n_query, num_workers=0)
    test_few_shot_params = dict(n_way=params.test_n_way, n_support=shot, n_query=n_query, num_workers=0)

    # 初始化数据加载器
    base_loader = SetDataManager(image_size, n_way=params.train_n_way, n_support=shot, n_query=n_query,
                                 num_workers=0).get_data_loader(base_file, aug=params.train_aug)
    val_loader = SetDataManager(image_size, n_way=params.test_n_way, n_support=shot, n_query=n_query,
                                num_workers=0).get_data_loader(val_file, aug=False)

    print(f"✅ 数据加载完成: 训练集 {base_file}, 验证集 {val_file}")
    return base_loader, val_loader


def init_model(params, shot, device, checkpoint_dir):
    """初始化OurNet模型"""
    # 1. 实例化骨干网络
    backbone_class = get_backbone(params.backbone)
    feature_model = backbone_class()
    feature_model = feature_model.to(device)

    # 2. 初始化OurNet
    model = OurNet(
        model_func=feature_model,
        n_way=params.train_n_way,
        n_support=shot,
        loss_type='softmax',
        use_fsrm=True,  # 使用FSRM
        use_fmrm=True,  # 使用FMRM
        attention_heads=8,
        attention_dropout=0.1,
        use_position_embedding=True,
        # ArcFace相关参数
        use_arcface=True,
        arcface_margin=0.2,
        arcface_scale=30.0,
        arcface_weight=0.25,
        # 可视化相关（初始为False，训练完解码器后启用）
        enable_visualization=False
    )
    model = model.to(device)

    # 3. 初始化模型内部属性
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(4, 3, 84, 84).to(device)
        dummy_feature = model.feature(dummy_input)
    model.feat_dim = dummy_feature.shape[1:]
    model.device = device
    model.train()

    # 验证核心参数
    assert model.n_way == params.train_n_way, f"模型n_way({model.n_way})与配置({params.train_n_way})不匹配"
    assert model.n_support == shot, f"模型n_support({model.n_support})与shot({shot})不匹配"

    print(f"✅ 模型初始化完成")
    print(f"  - 骨干网络: {params.backbone} | 特征维度: {model.feat_dim}")
    print(f"  - 任务配置: {params.train_n_way}-way {shot}-shot")
    print(f"  - ArcFace配置: margin={model.arcface_margin}, scale={model.arcface_scale}, weight={model.arcface_weight}")
    print(f"  - FSRM: {model.use_fsrm}, FMRM: {model.use_fmrm}")

    # 4. 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.learning_rate if hasattr(params, 'learning_rate') else 0.001,
        weight_decay=params.weight_decay if hasattr(params, 'weight_decay') else 0.0005
    )
    return model, optimizer


def load_best_model(model, checkpoint_dir, shot):
    """加载训练过程中保存的最佳模型"""
    best_model_path = os.path.join(checkpoint_dir, f'best_model_{shot}shot.pth')
    best_acc = 0.0
    best_epoch = -1

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc = checkpoint['best_val_acc']
        best_epoch = checkpoint['epoch'] + 1
        print(f"✅ 成功加载最佳模型 (Epoch {best_epoch}, 准确率: {best_acc:.4f})")
    else:
        print("⚠️  未找到最佳模型，使用当前模型权重")
    return model, best_acc, best_epoch


def standardize_batch(batch, target_size=(84, 84), device='cuda', verbose=False):
    """标准化batch数据"""
    x = None

    if isinstance(batch, (list, tuple)):
        if len(batch) >= 1 and torch.is_tensor(batch[0]):
            x = batch[0]
    elif torch.is_tensor(batch):
        x = batch

    if x is None or x.numel() == 0:
        return None

    # 调整维度
    if len(x.shape) == 5:
        B = x.shape[0] * x.shape[1]
        x = x.view(B, *x.shape[2:])
    elif len(x.shape) == 3:
        x = x.unsqueeze(0)

    # 统一通道数
    if x.shape[1] == 1:
        x = torch.cat([x, x, x], dim=1)
    elif x.shape[1] > 3:
        x = x[:, :3, :, :]

    x = x.to(device, dtype=torch.float32)

    # 统一尺寸和归一化
    if x.shape[2:] != target_size:
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    x = x / 255.0 if x.max() > 1.0 else x

    return x


# ===================== 特征提取 =====================
def extract_features(model, data_loader, device, max_batches=5, verbose=False):
    """提取特征用于TSNE/可视化"""
    model.eval()
    all_features = []
    all_labels = []
    batch_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_count >= max_batches:
                break

            try:
                x = standardize_batch(batch, target_size=(84, 84), device=device, verbose=verbose)
                if x is None:
                    batch_count += 1
                    continue

                features = model.feature(x)
                if len(features.shape) > 2:
                    features = features.view(features.shape[0], -1)
                features = features.cpu().numpy()

                # 处理标签
                y = np.zeros(features.shape[0])
                if isinstance(batch, (list, tuple)) and len(batch) >= 2 and torch.is_tensor(batch[1]):
                    try:
                        y = batch[1].cpu().numpy().flatten()
                    except:
                        pass

                # 标签映射
                y_np = []
                for idx in y:
                    try:
                        y_np.append(GLOBAL_CLASS_MAP.get(CLASS_NAMES[int(idx) % len(CLASS_NAMES)], int(idx) % 4))
                    except:
                        y_np.append(0)
                y_np = np.array(y_np)

                all_features.append(features)
                all_labels.append(y_np)
                batch_count += 1

            except Exception as e:
                batch_count += 1
                continue

    if len(all_features) == 0:
        raise Exception("❌ 所有批次特征提取失败！")

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0) if len(all_labels) > 0 else np.zeros(len(all_features))

    return all_features, all_labels


# ===================== T-SNE可视化 =====================
def plot_tsne(features, labels, epoch, save_path, shot, title_prefix="", max_samples_per_class=300):
    """绘制TSNE可视化"""
    # 采样
    sampled_features = []
    sampled_labels = []
    for class_idx in np.unique(labels):
        mask = labels == class_idx
        class_feats = features[mask]
        class_lbls = labels[mask]
        n_samples = min(max_samples_per_class, len(class_feats))
        if n_samples == 0:
            continue
        sample_idx = np.random.choice(len(class_feats), n_samples, replace=False)
        sampled_features.append(class_feats[sample_idx])
        sampled_labels.append(class_lbls[sample_idx])

    if len(sampled_features) == 0:
        print("⚠️  无足够样本绘制TSNE")
        return

    features = np.concatenate(sampled_features, axis=0)
    labels = np.concatenate(sampled_labels, axis=0)

    # TSNE降维
    perplexity = min(30, max(5, len(features) // 5))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=10,
        n_iter=1000,
        learning_rate='auto'
    )
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    tsne_results = tsne.fit_transform(features)

    # 绘图
    plt.figure(figsize=(10, 8))
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_idx
        if np.sum(mask) == 0:
            continue
        plt.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            c=CLASS_COLORS[class_idx],
            label=class_name,
            s=50,
            alpha=1,
            marker='o',
            edgecolors='none'
        )

    plt.title(f"{title_prefix} T-SNE ({shot}-shot, Epoch {epoch + 1})", fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.tight_layout()

    # 保存
    tsne_save_path = os.path.join(save_path, f"tsne_{shot}shot_{title_prefix.lower()}_epoch_{epoch + 1}.png")
    plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ TSNE可视化已保存: {tsne_save_path}")


# ===================== 训练分类器 =====================
def train_classifier(base_loader, val_loader, model, optimizer, params, shot, checkpoint_dir):
    """训练分类器"""
    # 设置默认训练轮数
    start_epoch = getattr(params, 'start_epoch', 0)
    stop_epoch = getattr(params, 'stop_epoch', 200)
    if stop_epoch <= start_epoch:
        stop_epoch = 200
        start_epoch = 0

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, verbose=True
    )

    # 初始化指标
    max_acc = 0.0
    best_epoch = -1
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []

    total_epochs = stop_epoch - start_epoch
    vis_frequency = 40

    print(f"\n{'=' * 80}")
    print(f"开始训练分类器 ({shot}-shot) | 总轮数: {total_epochs}")
    print(f"{'=' * 80}")
    print(f"损失函数：交叉熵损失 + ArcFace损失 (加权系数={model.arcface_weight})")

    for epoch in range(start_epoch, stop_epoch):
        # 训练
        model.train()
        train_loss, train_acc = model.train_loop(epoch, base_loader, optimizer)

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = model.test_loop(val_loader)

        # 保存指标
        train_loss_per_epoch.append(train_loss)
        train_acc_per_epoch.append(train_acc)
        val_loss_per_epoch.append(val_loss)
        val_acc_per_epoch.append(val_acc)

        # 打印日志
        print(f"[Epoch {epoch + 1:3d}/{total_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # TSNE可视化
        if (epoch + 1) % vis_frequency == 0 or epoch == stop_epoch - 1:
            try:
                train_features, train_labels = extract_features(
                    model, base_loader, next(model.parameters()).device, max_batches=5
                )
                plot_tsne(train_features, train_labels, epoch, checkpoint_dir, shot, "Train")

                val_features, val_labels = extract_features(
                    model, val_loader, next(model.parameters()).device, max_batches=5
                )
                plot_tsne(val_features, val_labels, epoch, checkpoint_dir, shot, "Val")
            except Exception as e:
                print(f"⚠️ TSNE可视化失败: {e}")

        # 保存最佳模型
        if val_acc > max_acc:
            max_acc = val_acc
            best_epoch = epoch
            save_dict = {
                'epoch': epoch,
                'n_shot': shot,
                'n_way': params.train_n_way,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': max_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            best_model_path = os.path.join(checkpoint_dir, f'best_model_{shot}shot.pth')
            torch.save(save_dict, best_model_path)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best_model_weights_{shot}shot.pth'))
            print(f"✅ 新最佳模型已保存 (Val Acc: {max_acc:.4f})")

        # 学习率调整
        scheduler.step(val_loss)

    # ========== 保存训练指标 ==========
    metrics_dict = {
        'n_shot': shot,
        'n_way': params.train_n_way,
        'train_loss': train_loss_per_epoch,
        'train_acc': train_acc_per_epoch,
        'val_loss': val_loss_per_epoch,
        'val_acc': val_acc_per_epoch,
        'best_val_acc': max_acc,
        'best_epoch': best_epoch + 1,
        'arcface_weight': model.arcface_weight,
        'val_acc_mean': float(np.mean(val_acc_per_epoch)) if val_acc_per_epoch else 0.0,
        'val_acc_std': float(np.std(val_acc_per_epoch)) if val_acc_per_epoch else 0.0,
        'val_acc_last_20': float(np.mean(val_acc_per_epoch[-20:])) if len(val_acc_per_epoch) >= 20 else 0.0
    }

    # 保存为NPY
    np.save(os.path.join(checkpoint_dir, f"{shot}shot_metrics.npy"), metrics_dict)

    # 保存为JSON
    with open(os.path.join(checkpoint_dir, f"{shot}shot_metrics.json"), 'w') as f:
        json.dump({
            'n_shot': int(shot),
            'n_way': int(params.train_n_way),
            'train_loss': [float(x) for x in train_loss_per_epoch],
            'train_acc': [float(x) for x in train_acc_per_epoch],
            'val_loss': [float(x) for x in val_loss_per_epoch],
            'val_acc': [float(x) for x in val_acc_per_epoch],
            'best_val_acc': float(max_acc),
            'best_epoch': int(best_epoch + 1),
            'arcface_weight': float(model.arcface_weight),
            'val_acc_mean': float(metrics_dict['val_acc_mean']),
            'val_acc_std': float(metrics_dict['val_acc_std']),
            'val_acc_last_20': float(metrics_dict['val_acc_last_20'])
        }, f, indent=4)

    # 保存训练曲线
    save_training_curves(train_loss_per_epoch, train_acc_per_epoch,
                         val_loss_per_epoch, val_acc_per_epoch,
                         metrics_dict['val_acc_mean'], checkpoint_dir, shot)

    print(f"\n{'=' * 80}")
    print(f"分类器训练完成 | {shot}-shot")
    print(f"最佳验证准确率: {max_acc:.4f} (Epoch {best_epoch + 1})")
    print(f"验证准确率均值: {metrics_dict['val_acc_mean']:.4f}, 标准差: {metrics_dict['val_acc_std']:.4f}")
    print(f"最后20轮验证准确率均值: {metrics_dict['val_acc_last_20']:.4f}")
    print(f"{'=' * 80}")

    return model, metrics_dict


# ===================== 保存训练曲线 =====================
def save_training_curves(train_loss, train_acc, val_loss, val_acc, val_acc_mean, save_dir, shot):
    """保存训练/验证曲线"""
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label=f'Train Loss ({shot}-shot)', linewidth=2, color='#1f77b4')
    plt.plot(val_loss, label=f'Val Loss ({shot}-shot)', linewidth=2, color='#ff7f0e')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (CrossEntropy + ArcFace)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title(f'Loss Curve ({shot}-shot)')

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label=f'Train Acc ({shot}-shot)', linewidth=2, color='#1f77b4')
    plt.plot(val_acc, label=f'Val Acc ({shot}-shot)', linewidth=2, color='#ff7f0e')
    plt.axhline(y=val_acc_mean, color='r', linestyle='--', label=f'Mean Val Acc: {val_acc_mean:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title(f'Accuracy Curve ({shot}-shot)')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"training_curves_{shot}shot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存数值文件
    np.save(os.path.join(save_dir, f"train_loss_{shot}shot.npy"), train_loss)
    np.save(os.path.join(save_dir, f"train_acc_{shot}shot.npy"), train_acc)
    np.save(os.path.join(save_dir, f"val_loss_{shot}shot.npy"), val_loss)
    np.save(os.path.join(save_dir, f"val_acc_{shot}shot.npy"), val_acc)

    print(f"✅ 训练曲线已保存: {save_path}")


# ===================== 训练可视化解码器 =====================
def train_visualization_decoder(model, train_loader, val_loader, checkpoint_dir, shot, epochs=1000):
    """训练可视化解码器（论文图6专用）"""
    print(f"\n{'=' * 80}")
    print(f"训练可视化解码器 ({shot}-shot)")
    print(f"{'=' * 80}")
    print(f"训练参数：Adam(lr=0.01), Batch Size=200, L1 Loss, {epochs} epochs")

    if not hasattr(model, 'visualization_decoder') or model.visualization_decoder is None:
        print("❌ 模型没有可视化解码器，跳过训练")
        return model

    device = next(model.parameters()).device

    # 冻结主模型参数
    for name, param in model.named_parameters():
        if "visualization_decoder" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 解码器优化器
    decoder_optimizer = torch.optim.Adam(
        model.visualization_decoder.parameters(),
        lr=0.01,
        weight_decay=0.0
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=epochs)
    criterion = torch.nn.L1Loss()

    # 准备训练数据
    print("准备解码器训练数据...")
    train_features = []
    train_images = []

    model.eval()
    batch_count = 0
    max_batches = 50

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_count >= max_batches:
                break

            try:
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    x_episode = batch[0].to(device)
                    n_way, n_samples, C, H, W = x_episode.shape
                    x_flat = x_episode.view(-1, C, H, W)

                    z = model.feature(x_flat)

                    train_features.append(z.cpu())
                    train_images.append(x_flat.cpu())
                    batch_count += 1
            except Exception as e:
                continue

    if len(train_features) == 0:
        print("❌ 无法准备解码器训练数据")
        return model

    train_features = torch.cat(train_features, dim=0)
    train_images = torch.cat(train_images, dim=0)

    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(train_features, train_images)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(200, len(dataset)),
        shuffle=True,
        num_workers=0
    )

    print(f"✅ 训练数据: {len(train_features)} 样本, 特征维度: {train_features.shape[1:]}")

    # 训练
    best_loss = float('inf')
    patience = 50
    patience_counter = 0

    for epoch in range(epochs):
        model.visualization_decoder.train()
        total_loss = 0.0
        batch_count = 0

        for batch_features, batch_images in dataloader:
            batch_features = batch_features.to(device)
            batch_images = batch_images.to(device)

            reconstructed = model.visualization_decoder(batch_features)

            if reconstructed.shape[-2:] != batch_images.shape[-2:]:
                reconstructed = F.interpolate(reconstructed, size=batch_images.shape[-2:],
                                              mode='bilinear', align_corners=False)

            loss = criterion(reconstructed, batch_images)

            decoder_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.visualization_decoder.parameters(), max_norm=1.0)
            decoder_optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(1, batch_count)
        scheduler.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"解码器 [{epoch + 1:4d}/{epochs}] | L1 Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(
                model.visualization_decoder.state_dict(),
                os.path.join(checkpoint_dir, f"decoder_best_{shot}shot.pth")
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于 epoch {epoch + 1}")
                break

    # 保存最终解码器
    torch.save(
        model.visualization_decoder.state_dict(),
        os.path.join(checkpoint_dir, f"decoder_pretrained_{shot}shot.pth")
    )
    print(f"✅ 解码器训练完成，最佳损失: {best_loss:.6f}")

    return model


# ===================== 主函数 =====================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset', help='dataset name')
    parser.add_argument('--backbone', type=str, default='resnet12', help='backbone network')
    parser.add_argument('--method', type=str, default='OurNet', help='method name')
    parser.add_argument('--train_n_way', type=int, default=4, help='number of classes for training')
    parser.add_argument('--test_n_way', type=int, default=4, help='number of classes for testing')
    parser.add_argument('--n_shot', type=int, required=True, help='number of support samples per class (1 or 5)')
    parser.add_argument('--train_aug', action='store_true', help='perform data augmentation or not')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--stop_epoch', type=int, default=200, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--train_decoder', action='store_true', help='train visualization decoder after classifier')

    args = parser.parse_args()

    # 验证shot参数
    if args.n_shot not in [1, 5]:
        raise ValueError("n_shot must be 1 or 5")

    print(f"\n{'=' * 80}")
    print(f"开始训练 {args.n_shot}-shot 模型")
    print(f"{'=' * 80}")
    print(f"参数配置:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Backbone: {args.backbone}")
    print(f"  - Method: {args.method}")
    print(f"  - n_way: {args.train_n_way}")
    print(f"  - n_shot: {args.n_shot}")
    print(f"  - GPU: {args.gpu}")
    print(f"  - Train Aug: {args.train_aug}")
    print(f"  - Stop Epoch: {args.stop_epoch}")
    print(f"  - Train Decoder: {args.train_decoder}")

    # 设置随机种子
    set_seed(10)

    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")
    print(f"✅ 使用设备: {device}")

    # 创建检查点目录
    checkpoint_dir = setup_checkpoint_dir(args, args.n_shot)

    # 保存配置
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 加载数据
    base_loader, val_loader = load_data(args, args.n_shot)

    # 初始化模型
    model, optimizer = init_model(args, args.n_shot, device, checkpoint_dir)
    print(f"✅ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练分类器
    model, metrics = train_classifier(base_loader, val_loader, model, optimizer, args, args.n_shot, checkpoint_dir)

    # 训练可视化解码器（如果需要）
    if args.train_decoder:
        model = train_visualization_decoder(model, base_loader, val_loader, checkpoint_dir, args.n_shot)

    # 加载最佳模型（用于后续评估）
    model, best_acc, best_epoch = load_best_model(model, checkpoint_dir, args.n_shot)

    # 保存最终结果摘要
    summary = {
        'n_shot': args.n_shot,
        'n_way': args.train_n_way,
        'best_val_acc': float(best_acc),
        'best_epoch': int(best_epoch),
        'val_acc_mean': float(metrics['val_acc_mean']),
        'val_acc_std': float(metrics['val_acc_std']),
        'checkpoint_dir': checkpoint_dir,
        'model_path': os.path.join(checkpoint_dir, f'best_model_{args.n_shot}shot.pth'),
        'metrics_path': os.path.join(checkpoint_dir, f'{args.n_shot}shot_metrics.json')
    }

    with open(os.path.join(checkpoint_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n🎉 {args.n_shot}-shot 训练完成！")
    print(f"📁 结果保存目录: {checkpoint_dir}")
    print(f"📊 最佳验证准确率: {best_acc:.4f}")
    print(f"📈 平均验证准确率: {metrics['val_acc_mean']:.4f} ± {metrics['val_acc_std']:.4f}")


if __name__ == '__main__':
    main()
