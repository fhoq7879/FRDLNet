
import torch
import numpy as np
import torch.optim
import torch.utils.data.sampler
import os
import random
import time

import configs
import backbone
import data.feature_loader as feat_loader
from io_utils import parse_args, get_best_file


# 按需导入模型（避免冗余，修复Classifier_cosine未定义问题）
def import_model(method):
    if method == 'relationnet':
        from methods.relationnet import RelationNet, RelationModule
        return RelationNet, RelationModule
    elif method == 'CosineBatch':
        from methods.cosine_batch import CosineBatch, Classifier_cosine  # 同时导入
        return CosineBatch, Classifier_cosine
    elif method == 'OurNet':
        from methods.ournet import OurNet
        return OurNet, None
    else:
        raise ValueError(f"不支持的模型方法：{method}")


def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15):
    class_list = list(cl_data_file.keys())
    total_classes = len(class_list)

    # 确保n_way不超过总类别数
    actual_n_way = min(n_way, total_classes)
    if actual_n_way != n_way:
        print(f"警告: n_way ({n_way}) 大于总类别数 ({total_classes})，已自动调整为 {actual_n_way}。")
        n_way = actual_n_way

    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])

    z_all = torch.from_numpy(np.array(z_all))

    # 关键：获取模型设备并将数据移到相同设备
    device = next(model.parameters()).device
    z_all = z_all.to(device)

    model.n_query = n_query

    # 模型分支判断（重构逻辑，适配不同返回值）
    if hasattr(model, 'is_ournet') or model.__class__.__name__ == 'OurNet':
        # OurNet返回3个值：final_scores, relations, cosine → 直接用final_scores
        final_scores, _, _ = model.set_forward(z_all, is_feature=True)
        scores = final_scores  # 无需平均，直接用最终融合得分
    else:
        # 其他模型（relationnet/CosineBatch）返回1个值
        scores = model.set_forward(z_all, is_feature=True)

    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc


if __name__ == '__main__':
    params = parse_args('test')

    # 设备设置
    device = torch.device('cpu')  # 强制使用CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用GPU

    acc_all = []
    iter_num = 600

    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    loss_type = 'mse'

    # 1. 动态选择backbone
    if params.backbone == 'conv4':
        feature_model = backbone.Conv4NP
    elif params.backbone == 'resnet12':
        feature_model = backbone.ResNet12
    elif params.backbone == 'resnet18':
        feature_model = backbone.ResNet18
    else:
        raise ValueError(f"不支持的backbone类型：{params.backbone}")

    # 2. 动态导入并创建模型（ExtraModule对CosineBatch是Classifier_cosine）
    ModelClass, ExtraModule = import_model(params.method)
    model = ModelClass(feature_model, loss_type=loss_type, **few_shot_params)
    model = model.to(device)  # 移到CPU

    # 3. 构建checkpoint路径
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
        configs.save_dir, params.dataset, params.backbone, params.method
    )
    if params.train_aug:
        checkpoint_dir += '_aug'
    checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    # ===================== 核心修改：硬编码指定特征文件（无需命令行参数） =====================
    # 替换为你的实际特征文件完整路径（.hdf5文件）
    novel_file = "/home/xpu/文档/Python/fhoq/实验2/features/dataset/resnet12_OurNet_20260305_154239_4way_5shot/20260305_193134/novel.hdf5"
    print(f"使用硬编码指定的特征文件：{novel_file}")

    # 验证特征文件是否存在
    if not os.path.exists(novel_file):
        raise FileNotFoundError(
            f"特征文件不存在：{novel_file}\n"
            f"请核对路径是否正确，或重新执行特征提取生成该文件。"
        )

    # 4. 加载特征文件
    cl_data_file = feat_loader.init_loader(novel_file)
    print(f"✅ 成功加载特征文件：{novel_file}")

    # 5. 加载模型权重
    modelfile = get_best_file(checkpoint_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile, map_location=device)
        state_dict = tmp['state']

        # 按模型类型处理额外初始化
        if params.method == 'relationnet':
            # 加载/手动指定feat_dim
            if 'feat_dim' in tmp:
                model.feat_dim = tmp['feat_dim']
                while len(model.feat_dim) < 3:
                    model.feat_dim.append(1)
                print(f"从checkpoint加载并补全feat_dim：{model.feat_dim}")
            else:
                model.feat_dim = [512, 1, 1]
                print("未找到checkpoint中的feat_dim，手动指定：", model.feat_dim)

            # 初始化relation_module
            model.relation_module = ExtraModule(model.feat_dim, 8, model.loss_type)
            model.relation_module = model.relation_module.to(device)
        else:
            # CosineBatch/OurNet：加载并补全feat_dim
            print(f"当前模型为{params.method}，无需初始化relation_module")

            # 必须修改：OurNet过滤无关权重
            if params.method == 'OurNet':
                state_dict = {k: v for k, v in state_dict.items() if
                              'classifier_cosine' not in k and 'relation_module' not in k}
                print("已过滤OurNet不需要的classifier_cosine和relation_module权重")

            if 'feat_dim' in tmp:
                model.feat_dim = tmp['feat_dim']
                # 强制补全至3维，避免索引越界
                while len(model.feat_dim) < 3:
                    model.feat_dim.append(1)
                print(f"从checkpoint加载并补全feat_dim：{model.feat_dim}")
            # 对CosineBatch，若未加载feat_dim则手动指定3维
            if params.method == 'CosineBatch' and model.feat_dim is None:
                model.feat_dim = [512, 1, 1]  # 固定3维[C, H, W]
                print(f"手动指定CosineBatch的feat_dim：{model.feat_dim}")
            # 必须修改：OurNet feat_dim兜底
            if params.method == 'OurNet' and model.feat_dim is None:
                model.feat_dim = [512, 1, 1]  # 与ResNet12输出匹配
                print(f"手动指定OurNet的feat_dim：{model.feat_dim}")

            # 核心修复：使用ExtraModule（即Classifier_cosine）初始化（仅CosineBatch需要）
            if params.method == 'CosineBatch':
                model.classifier_cosine = ExtraModule(
                    n_way=model.n_way,
                    n_support=model.n_support,
                    feat_dim=model.feat_dim
                )
                model.classifier_cosine = model.classifier_cosine.to(device)
                print("已手动初始化CosineBatch的classifier_cosine")

        # 权重名映射（解决命名不匹配问题，按需添加）
        weight_name_mapping = {}
        # 示例：weight_name_mapping['old_name'] = 'new_name'

        # 构建映射后的state_dict
        mapped_state_dict = {}
        for k, v in state_dict.items():
            mapped_k = weight_name_mapping.get(k, k)
            mapped_state_dict[mapped_k] = v

        # 过滤形状匹配的参数，确保权重正常加载
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        for k, v in mapped_state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
                # 打印classifier_cosine权重加载情况（可选）
                if 'classifier_cosine' in k:
                    print(f"成功加载权重：{k}")

        # 加载权重
        model.load_state_dict(filtered_state_dict, strict=False)
        model = model.to(device)

        # 打印权重加载统计
        print(f"\n权重加载完成：成功加载 {len(filtered_state_dict)} 个参数")
        unloaded_keys = [k for k in state_dict.keys() if k not in filtered_state_dict.keys()]
        missing_keys = [k for k in model_state_dict.keys() if k not in filtered_state_dict.keys()]
        if unloaded_keys:
            print(f"警告：{len(unloaded_keys)}个参数未加载（示例：{unloaded_keys[:4]}...）")
        if missing_keys:
            print(f"警告：{len(missing_keys)}个参数缺失（示例：{missing_keys[:4]}...）")
    else:
        print("未找到checkpoint文件，使用随机初始化模型测试")
        # 若未找到checkpoint，对CosineBatch/OurNet手动初始化feat_dim
        if params.method == 'CosineBatch' and model.feat_dim is None:
            model.feat_dim = [512, 1, 1]
            model.classifier_cosine = ExtraModule(
                n_way=model.n_way,
                n_support=model.n_support,
                feat_dim=model.feat_dim
            )
            model.classifier_cosine = model.classifier_cosine.to(device)
            print("已手动初始化CosineBatch的classifier_cosine（随机权重）")
        # 可选修改：OurNet兜底
        if params.method == 'OurNet' and model.feat_dim is None:
            model.feat_dim = [512, 1, 1]
            print("已手动指定OurNet的feat_dim（随机权重）")

    # 6. 执行测试
    print("\n开始执行测试，共{}次迭代...".format(iter_num))
    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, model, n_query=15, **few_shot_params)
        acc_all.append(acc)
        # 每100次迭代打印进度
        if (i + 1) % 100 == 0:
            print(f"迭代 {i + 1}/{iter_num}，当前平均准确率：{np.mean(acc_all[:i + 1]):.2f}%")

    # 7. 输出最终结果
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    confidence_interval = 1.96 * acc_std / np.sqrt(iter_num)
    print(f"\n测试完成！")
    print(f"总迭代数：{iter_num}")
    print(f"平均准确率：{acc_mean:.2f}%")
    print(f"准确率标准差：{acc_std:.2f}%")
    print(f"95%置信区间：±{confidence_interval:.2f}%")
    print(f"最终结果：{acc_mean:.2f}% ± {confidence_interval:.2f}%")

    # 8. 保存结果到文件
    record_dir = './record'
    os.makedirs(record_dir, exist_ok=True)
    with open(os.path.join(record_dir, "results.txt"), 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        aug_str = '-aug' if params.train_aug else ''
        exp_setting = (
            f"{params.dataset}-{os.path.basename(novel_file).replace('.hdf5', '')}-{params.backbone}-{params.method}{aug_str} "
            f"{params.n_shot}shot {params.train_n_way}way_train {params.test_n_way}way_test"
        )
        acc_str = f"{iter_num} Test Acc = {acc_mean:.2f}% ± {confidence_interval:.2f}%"
        f.write(f"Time: {timestamp} | Setting: {exp_setting} | {acc_str}\n")

    print(f"\n结果已保存到 {os.path.join(record_dir, 'results.txt')}")
