import h5py
import numpy as np


def init_loader(filename):
    """
    加载特征文件，构建分类特征字典（彻底规避标量数据问题）
    :param filename: .hdf5 特征文件路径
    :return: cl_data_file: 键为类别标签，值为该类别的特征数组
    """
    cl_data_file = {}

    with h5py.File(filename, 'r') as f:
        # 1. 直接读取核心数据（不遍历所有键，避免标量问题）
        try:
            # 读取特征和标签（核心数据，必选）
            all_feats = f['all_feats'][:]  # 样本特征数组
            all_labels = f['all_labels'][:]  # 样本标签数组
            print(f"成功读取核心数据：")
            print(f"  - all_feats: 形状 {all_feats.shape}，数据类型 {all_feats.dtype}")
            print(f"  - all_labels: 形状 {all_labels.shape}，数据类型 {all_labels.dtype}")
        except KeyError as e:
            raise KeyError(f"特征文件缺少核心数据键：{e}\n请确保特征文件包含 all_feats 和 all_labels")
        except ValueError as e:
            raise ValueError(f"读取核心数据失败：{e}\n可能是数据格式异常，请重新生成特征文件")

        # 2. 读取可选额外信息（跳过标量错误，不影响核心逻辑）
        try:
            class_names = f['class_names'][:]
            # 转换字节型字符串为普通字符串
            if class_names.dtype == 'S':
                class_names = [name.decode('utf-8') for name in class_names]
            print(f"成功读取类别名称：{class_names}")
        except (KeyError, ValueError):
            print("未读取到类别名称（可选信息，不影响测试）")
            class_names = None

        try:
            label_to_class = f['label_to_class'][:]
            label_to_class = label_to_class.decode('utf-8')
            print(f"成功读取标签-类别映射：{label_to_class}")
        except (KeyError, ValueError):
            print("未读取到标签-类别映射（可选信息，不影响测试）")
            label_to_class = None

        try:
            count = f['count'][:]
            print(f"成功读取样本数量：{count[0]}")
        except (KeyError, ValueError):
            print("未读取到样本数量（可选信息，不影响测试）")
            count = None

        # 3. 构建分类特征字典（按标签分组）
        unique_labels = np.unique(all_labels)
        for label in unique_labels:
            # 提取该标签对应的所有特征
            label_feats = all_feats[all_labels == label]
            # 用字符串作为键（兼容不同标签类型）
            cl_key = str(int(label)) if isinstance(label, (np.float64, np.float32)) else str(label)
            cl_data_file[cl_key] = label_feats

        print(f"成功构建分类特征字典：共 {len(unique_labels)} 个类别")
        for cls_key, cls_feats in cl_data_file.items():
            print(f"  - 类别 {cls_key}：{cls_feats.shape[0]} 个样本，特征维度 {cls_feats.shape[1:]}")

    return cl_data_file


