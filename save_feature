import numpy as np
import torch
import os
import h5py
import sys
import traceback
import logging
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# 强制使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('save_features_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# 保留原有依赖
import configs
import backbone
from io_utils import parse_args, get_best_file


# ===================== 自定义数据集（适配特殊JSON格式） =====================
class CustomNovelDataset(Dataset):
    def __init__(self, json_file, dataset_root, image_size, transform=None):
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.transform = transform

        # 解析特殊格式novel.json
        with open(json_file, 'r', encoding='utf-8') as f:
            self.json_data = json.load(f)

        # 提取核心信息
        self.class_names = self.json_data["label_names"]
        self.image_paths = self.json_data["image_names"]
        self.image_labels = self.json_data["image_labels"]

        # 拼接完整图片路径
        self.full_image_paths = [
            os.path.join(self.dataset_root, img_path)
            for img_path in self.image_paths
        ]

        # 验证图片有效性，过滤不存在的图片
        self.valid_indices = []
        for idx, img_path in enumerate(self.full_image_paths):
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
            else:
                logger.warning(f"图片不存在，自动跳过：{img_path}")

        logger.info(f"有效图片数量：{len(self.valid_indices)} / 原始图片数量：{len(self.full_image_paths)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 获取有效索引对应的真实数据
        real_idx = self.valid_indices[idx]
        img_path = self.full_image_paths[real_idx]
        label = self.image_labels[real_idx]

        try:
            # 加载图片并转换为RGB格式
            img = Image.open(img_path).convert('RGB')
            # 应用图片变换
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            logger.error(f"加载图片失败：{img_path}，错误信息：{str(e)}")
            # 返回占位符，避免程序中断
            return torch.zeros(3, self.image_size, self.image_size), label


# ===================== 保存特征（支持分块存储+类别信息） =====================
def save_features(model, data_loader, class_names, outfile):
    """
    保存模型特征到HDF5文件
    """
    logger.info(f"开始特征提取，输出文件: {outfile}")

    try:
        # 创建输出目录
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        with h5py.File(outfile, 'w') as f:
            max_count = len(data_loader) * data_loader.batch_size
            logger.info(f"数据加载器共有 {len(data_loader)} 个批次，批次大小 {data_loader.batch_size}")
            logger.info(f"预计最大样本数: {max_count}")

            # 创建分块数据集（支持resize）
            chunk_size = min(16, max_count)
            all_labels = f.create_dataset(
                'all_labels',
                shape=(max_count,),
                dtype='i',
                chunks=(chunk_size,),
                maxshape=(None,)
            )
            all_feats = None
            count = 0
            device = torch.device("cpu")

            for i, (x, y) in enumerate(data_loader):
                logger.info(f"处理批次 {i + 1}/{len(data_loader)}")
                logger.info(f"输入数据形状: {x.shape}, 数据类型: {x.dtype}")

                try:
                    # 确保数据在CPU上
                    x = x.to(device)

                    # 无梯度计算
                    with torch.no_grad():
                        feats = model(x)

                    logger.info(f"特征输出形状: {feats.shape}")

                    if all_feats is None:
                        # 创建特征数据集
                        feat_shape = [max_count] + list(feats.size()[1:])
                        all_feats = f.create_dataset(
                            'all_feats',
                            shape=feat_shape,
                            dtype='f',
                            chunks=(chunk_size,) + tuple(feats.size()[1:]),
                            maxshape=(None,) + tuple(feats.size()[1:])
                        )
                        logger.info(f"创建特征数据集，形状: {all_feats.shape}")

                    # 保存当前批次数据
                    batch_size = feats.size(0)
                    all_feats[count:count + batch_size] = feats.cpu().numpy()
                    all_labels[count:count + batch_size] = y.numpy()
                    count += batch_size

                    logger.info(f"成功保存批次 {i + 1}，累计处理样本: {count}")

                except Exception as e:
                    logger.error(f"处理批次 {i + 1} 时发生错误: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            # 调整数据集大小到实际样本数
            all_feats.resize((count,) + all_feats.shape[1:])
            all_labels.resize((count,))

            # 保存元信息
            count_var = f.create_dataset('count', (1,), dtype='i')
            count_var[0] = count

            # 保存类别信息
            f.create_dataset('class_names', data=np.array(class_names, dtype='S'))
            label_to_class = {i: name for i, name in enumerate(class_names)}
            f.create_dataset('label_to_class', data=str(label_to_class).encode('utf-8'))

            logger.info(f"特征提取完成! 实际处理样本数: {count}")
            logger.info(f"保存的类别数量: {len(class_names)}")
            logger.info(f"保存的类别名称: {class_names}")

    except Exception as e:
        logger.error(f"特征保存过程发生严重错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# ===================== 验证模型和数据 =====================
def validate_model_and_data(model, data_loader):
    """
    验证模型和数据加载器是否正常工作
    """
    logger.info("=== 开始模型和数据验证 ===")

    # 测试数据加载
    try:
        test_batch = next(iter(data_loader))
        if isinstance(test_batch, (list, tuple)) and len(test_batch) == 2:
            x, y = test_batch
            logger.info(f"数据加载测试通过 - 数据形状: {x.shape}, 标签形状: {y.shape}")
        else:
            logger.warning(f"数据加载器返回非常规格式: {type(test_batch)}")
    except StopIteration:
        logger.error("数据加载器为空，没有数据可加载!")
        return False
    except Exception as e:
        logger.error(f"数据加载测试失败: {e}")
        return False

    # 测试模型前向传播
    try:
        dummy_input = torch.randn(1, 3, 84, 84)
        with torch.no_grad():
            dummy_output = model(dummy_input)
        logger.info(f"模型前向传播测试通过 - 输出形状: {dummy_output.shape}")
        return True
    except Exception as e:
        logger.error(f"模型前向传播测试失败: {e}")
        return False


# ===================== 主函数 =====================
if __name__ == '__main__':
    try:
        logger.info("=== 特征提取程序开始 ===")

        params = parse_args('save_features')
        device = torch.device("cpu")
        logger.info(f"使用设备: {device}")

        # 记录参数
        logger.info(f"参数: dataset={params.dataset}, backbone={params.backbone}, "
                    f"method={params.method}, n_shot={params.n_shot}, "
                    f"train_n_way={params.train_n_way}")

        image_size = 84
        split = params.split
        logger.info(f"数据集划分: {split}")

        # 1. 检查数据文件
        loadfile = os.path.join(configs.data_dir[params.dataset], split + '.json')
        logger.info(f"数据文件路径: {loadfile}")

        if not os.path.exists(loadfile):
            logger.error(f"数据文件不存在: {loadfile}")
            parent_dir = os.path.dirname(loadfile)
            if os.path.exists(parent_dir):
                logger.info(f"目录 {parent_dir} 中的文件:")
                for f in os.listdir(parent_dir):
                    logger.info(f"  - {f}")
            raise FileNotFoundError(f"数据文件 {loadfile} 未找到")

        # 2. 检查模型文件
        checkpoint_dir = "/home/xpu/文档/Python/fhoq/实验2/checkpoints/dataset/resnet12_OurNet_20260305_154239_4way_5shot"
        logger.info(f"手动指定的检查点目录: {checkpoint_dir}")

        if not os.path.exists(checkpoint_dir):
            logger.error(f"手动指定的检查点目录不存在: {checkpoint_dir}")
            raise FileNotFoundError(f"检查点目录 {checkpoint_dir} 未找到")

        # 生成时间戳
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"生成时间戳: {time_str}")

        # 获取模型文件
        modelfile = get_best_file(checkpoint_dir)
        logger.info(f"模型文件: {modelfile}")

        if modelfile is None:
            logger.error(f"在检查点目录 {checkpoint_dir} 中未找到有效模型文件")
            logger.info(f"检查点目录内文件列表：")
            for file_name in os.listdir(checkpoint_dir):
                logger.info(f"  - {file_name}")
            raise FileNotFoundError(f"未在 {checkpoint_dir} 中找到最佳模型文件")

        if not os.path.exists(modelfile):
            logger.error(f"模型文件不存在: {modelfile}")
            checkpoint_files = os.listdir(checkpoint_dir)
            logger.info(f"检查点目录中的文件: {checkpoint_files}")
            raise FileNotFoundError(f"模型文件 {modelfile} 未找到")

        # 3. 设置输出路径
        feat_root_dir = checkpoint_dir.replace("checkpoints", "features")
        feat_time_dir = os.path.join(feat_root_dir, time_str)

        if params.save_iter != -1:
            outfile = os.path.join(feat_time_dir, split + "_" + str(params.save_iter) + ".hdf5")
        else:
            outfile = os.path.join(feat_time_dir, split + ".hdf5")

        logger.info(f"输出特征文件: {outfile}")
        logger.info(f"特征根目录: {feat_root_dir}")
        logger.info(f"带时间戳的特征子目录: {feat_time_dir}")

        # 4. 创建自定义数据加载器
        logger.info("创建自定义数据加载器...")
        dataset_root = "/home/xpu/文档/Python/fhoq/实验2/datasets"

        # 图片变换
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 创建数据集和加载器
        custom_dataset = CustomNovelDataset(loadfile, dataset_root, image_size, transform)
        class_names = custom_dataset.class_names
        data_loader = DataLoader(
            custom_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )
        logger.info(f"自定义数据加载器创建成功")
        logger.info(f"真实类别数量: {len(class_names)}")
        logger.info(f"真实类别名称: {class_names}")
        logger.info(f"总有效图片数量: {len(custom_dataset)}")

        # 5. 加载模型（核心修复：KeyError 'state'）
        logger.info(f"加载骨干网络: {params.backbone}")

        # 加载骨干网络
        if params.backbone == 'conv4':
            model = backbone.Conv4NP()
        elif params.backbone == 'resnet12':
            model = backbone.resnet12()  # 注意：这里要确保resnet12()是正确的函数名
        else:
            raise ValueError(f"不支持的骨干网络: {params.backbone}")

        model = model.to(device)

        # ========== 核心修复：读取正确的键名 state_dict ==========
        logger.info(f"加载模型权重: {modelfile}")
        tmp = torch.load(modelfile, map_location='cpu')

        # 打印模型文件中的所有键，方便调试
        logger.info(f"模型文件中的键: {list(tmp.keys())}")

        # 读取正确的键名（训练代码中保存的是 state_dict）
        if 'state_dict' in tmp:
            state = tmp['state_dict']  # 修复KeyError的核心行
            logger.info("成功读取 state_dict 键")
        elif 'state' in tmp:
            state = tmp['state']
            logger.info("成功读取 state 键")
        else:
            raise KeyError(f"模型文件中没有找到 'state_dict' 或 'state' 键，可用键: {list(tmp.keys())}")

        # 处理状态字典键名（适配backbone特征提取）
        state_keys = list(state.keys())
        new_state = {}
        for key in state_keys:
            if "feature." in key:
                # 移除 feature. 前缀
                newkey = key.replace("feature.", "")
                new_state[newkey] = state[key]
                logger.info(f"重命名键: {key} -> {newkey}")
            elif not key.startswith('classifier') and not key.startswith('relation'):
                # 保留backbone的参数
                new_state[key] = state[key]
                logger.info(f"保留键: {key}")

        # 加载模型参数
        try:
            model.load_state_dict(new_state, strict=True)
            logger.info("模型参数加载成功 (严格模式)")
        except Exception as e:
            logger.warning(f"严格模式加载失败: {e}")
            logger.info("尝试宽松模式加载...")
            model.load_state_dict(new_state, strict=False)
            logger.info("模型参数加载成功 (宽松模式)")

        model.eval()
        logger.info("模型设置为评估模式")

        # 6. 验证模型和数据
        if not validate_model_and_data(model, data_loader):
            logger.error("模型和数据验证失败，终止执行")
            sys.exit(1)

        # 7. 执行特征提取
        logger.info("=== 开始特征提取过程 ===")
        save_features(model, data_loader, class_names, outfile)
        logger.info("=== 特征提取程序完成 ===")

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
