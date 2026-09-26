import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import utils
import math
from collections import OrderedDict


# ===================== 1. ArcFace损失类（适配小样本） =====================
class ArcFaceLoss(nn.Module):
    """适配小样本的ArcFace损失（基于动态原型）"""

    def __init__(self, margin=0.3, scale=30.0, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.margin = margin  # 基础角度边际
        self.scale = scale  # 尺度因子
        self.easy_margin = easy_margin  # 简化边际计算，避免训练初期梯度爆炸

        # 预计算基础边际的余弦/正弦值
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)  # 阈值
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, cosine_scores, labels, n_support=None):
        """
        输入：
            cosine_scores: [batch_size, n_way] 原始余弦相似度得分（范围[-1,1]）
            labels: [batch_size] 真实标签（0~n_way-1）
            n_support: 支持集样本数（用于动态调整margin）
        输出：
            arcface_loss: ArcFace损失值
        """
        # 1. 动态调整margin：小样本支持集越少，margin越小，避免过拟合
        if n_support is not None and n_support < 5:
            dynamic_m = self.margin * (n_support / 5)  # 按支持集数量比例缩放
            cos_m = math.cos(dynamic_m)
            sin_m = math.sin(dynamic_m)
            th = math.cos(math.pi - dynamic_m)
            mm = math.sin(math.pi - dynamic_m) * dynamic_m
        else:
            cos_m, sin_m, th, mm = self.cos_m, self.sin_m, self.th, self.mm

        # 2. 提取对应类别的余弦值
        cos_theta = cosine_scores
        cos_theta_target = cos_theta.gather(1, labels.view(-1, 1))  # [batch_size, 1]

        # 3. 计算cos(theta + m) = cosθcosm - sinθsinm
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta_target, 2) + 1e-8)  # 加小值避免梯度NaN
        cos_theta_m = cos_theta_target * cos_m - sin_theta * sin_m

        # 4. 易边际处理：避免训练初期梯度爆炸
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta_target > 0, cos_theta_m, cos_theta_target)
        else:
            cos_theta_m = torch.where(cos_theta_target > th, cos_theta_m, cos_theta_target - mm)

        # 5. 重构得分矩阵：仅对目标类别应用边际
        output = cos_theta.scatter(1, labels.view(-1, 1), cos_theta_m)
        # 6. 缩放得分（增强判别性）
        output *= self.scale

        # 7. 计算带角度边际的交叉熵损失
        loss = F.cross_entropy(output, labels)
        return loss


# ===================== 2. Feature Self-Reconstruction Module (FSRM) =====================
class FeatureSelfReconstructionModule(nn.Module):
    """特征自重构模块 (FSRM) 

    def __init__(self, feat_dim, num_heads=8, dropout=0.1, use_position_embedding=True):
        """
        Args:
            feat_dim: [C, H, W] 特征维度
            num_heads: 注意力头数
            dropout: dropout比率
            use_position_embedding: 是否使用位置编码
        """
        super(FeatureSelfReconstructionModule, self).__init__()

        # 特征维度
        self.C = feat_dim[0]  # 通道数
        self.H = feat_dim[1]  # 高度
        self.W = feat_dim[2]  # 宽度
        self.num_patches = self.H * self.W  # r = h × w
        self.num_heads = num_heads
        self.head_dim = self.C // num_heads
        self.use_position_embedding = use_position_embedding

        # 位置编码 E_pos ∈ R^(r×d)
        if use_position_embedding:
            self.position_embedding = nn.Parameter(
                torch.randn(1, self.num_patches, self.C) * 0.02
            )

        # 自注意力的Q、K、V投影矩阵
        self.W_Q = nn.Linear(self.C, self.C, bias=False)
        self.W_K = nn.Linear(self.C, self.C, bias=False)
        self.W_V = nn.Linear(self.C, self.C, bias=False)

        # Layer Normalization
        self.ln1 = nn.LayerNorm(self.C)
        self.ln2 = nn.LayerNorm(self.C)

        # MLP 
        self.mlp = nn.Sequential(
            nn.Linear(self.C, self.C * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.C * 4, self.C),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def split_heads(self, x):
        """将输入拆分为多头"""
        batch_size, num_patches, _ = x.shape
        x = x.view(batch_size, num_patches, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, num_heads, num_patches, head_dim]

    def forward(self, x):
        """
        输入: x - [B, C, H, W] 特征图
        输出: reconstructed - [B, r, C] 其中 r = H*W
        """
        # 保存原始形状
        B, C, H, W = x.shape

        # Reshape特征为序列: [B, C, H, W] -> [B, H*W, C] (对应论文中的 [x_i^1, x_i^2, ..., x_i^r])
        x_flat = x.flatten(2).transpose(1, 2)  # [B, r, C]

        # 添加位置编码: z_i = [x_i^1, x_i^2, ..., x_i^r] + E_pos
        if self.use_position_embedding:
            z = x_flat + self.position_embedding
        else:
            z = x_flat

        # 计算自注意力
        Q = self.split_heads(self.W_Q(z))  # [B, num_heads, r, head_dim]
        K = self.split_heads(self.W_K(z))  # [B, num_heads, r, head_dim]
        V = self.split_heads(self.W_V(z))  # [B, num_heads, r, head_dim]

        # 注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力
        attended = torch.matmul(attention_weights, V)  # [B, num_heads, r, head_dim]
        attended = attended.transpose(1, 2).contiguous().view(B, self.num_patches, self.C)  # [B, r, C]

        # 残差连接 + LayerNorm 
        z_attended = z + attended
        z_norm = self.ln1(z_attended)

        # MLP处理
        mlp_out = self.mlp(z_norm)
        z_out = z_norm + mlp_out
        z_out = self.ln2(z_out)

        return z_out  # [B, r, C]


# ===================== 3. Feature Mutual Reconstruction Module (FMRM) =====================
class FeatureMutualReconstructionModule(nn.Module):
    """特征互重构模块 (FMRM) 

    实现双向重构:
    1. 支持集 -> 查询集重构: S_c -> Q_i (增加类间差异)
    2. 查询集 -> 支持集重构: Q_i -> S_c (减少类内差异)
    """

    def __init__(self, feat_dim, num_heads=8, dropout=0.1):
        """
        Args:
            feat_dim: [C, H, W] 特征维度
            num_heads: 注意力头数
            dropout: dropout比率
        """
        super(FeatureMutualReconstructionModule, self).__init__()

        self.C = feat_dim[0]  # 通道数
        self.H = feat_dim[1]  # 高度
        self.W = feat_dim[2]  # 宽度
        self.num_patches = self.H * self.W  # r = h × w
        self.num_heads = num_heads
        self.head_dim = self.C // num_heads

        # 共享的Q、K、V投影矩阵
        self.W_Q = nn.Linear(self.C, self.C, bias=False)
        self.W_K = nn.Linear(self.C, self.C, bias=False)
        self.W_V = nn.Linear(self.C, self.C, bias=False)

        self.dropout = nn.Dropout(dropout)

        # 可学习的权重参数 λ1 和 λ2 (初始化为0.5)
        self.lambda1 = nn.Parameter(torch.tensor(0.5))
        self.lambda2 = nn.Parameter(torch.tensor(0.5))

        # 可学习的温度因子 τ - 初始化为较小值
        self.tau = nn.Parameter(torch.tensor(0.01))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, num_heads, seq_len, head_dim]

    def forward(self, support_features, query_features):
        """
            support_features: [n_way, k, r, C] - 支持集特征 (S_c)
            query_features: [n_total_query, r, C] - 查询集特征 (Q_i)

        输出:
            reconstructed_query: [n_way, n_total_query, r, C] - 重构的查询特征 (Q̂_(c,i))
            reconstructed_support: [n_way, n_total_query, k, r, C] - 重构的支持特征 (Ŝ_(i,c))
            distances_q2s: [n_total_query, n_way] - 查询到支持的距离 d_Q_i→S_c (已归一化)
            distances_s2q: [n_total_query, n_way] - 支持到查询的距离 d_S_c→Q_i (已归一化)
            total_distances: [n_total_query, n_way] - 加权总距离
        """
        n_way, k, r, C = support_features.shape
        n_total_query = query_features.shape[0]

        # 展平支持特征用于注意力计算
        S_c_flat = support_features.view(n_way, k * r, C)  # [n_way, k*r, C]

        # 计算Q、K、V (使用共享投影矩阵)
        # 对支持特征
        S_Q = self.W_Q(S_c_flat)  # [n_way, k*r, C]
        S_K = self.W_K(S_c_flat)  # [n_way, k*r, C]
        S_V = self.W_V(S_c_flat)  # [n_way, k*r, C]

        # 对查询特征
        Q_Q = self.W_Q(query_features)  # [n_total_query, r, C]
        Q_K = self.W_K(query_features)  # [n_total_query, r, C]
        Q_V = self.W_V(query_features)  # [n_total_query, r, C]

        # ========== (1) 支持集 -> 查询集重构 (对应论文公式4) ==========
        # Q̂_(c,i) = Attention(Q_i^Q, S_c^K, S_c^V)

        # 准备多头注意力
        Q_Q_heads = self.split_heads(Q_Q)  # [n_total_query, num_heads, r, head_dim]

        # 为每个类计算重构
        reconstructed_query = []

        for c in range(n_way):
            # S_c^K, S_c^V
            S_K_heads = self.split_heads(S_K[c:c + 1])  # [1, num_heads, k*r, head_dim]
            S_V_heads = self.split_heads(S_V[c:c + 1])  # [1, num_heads, k*r, head_dim]

            # 计算注意力: Q_i^Q @ (S_c^K)^T / sqrt(d_k)
            attention_scores = torch.matmul(Q_Q_heads, S_K_heads.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attention_weights = F.softmax(attention_scores, dim=-1)  # [n_total_query, num_heads, r, k*r]
            attention_weights = self.dropout(attention_weights)

            # 应用注意力到V: attention_weights @ S_c^V
            Q_recon_heads = torch.matmul(attention_weights, S_V_heads)  # [n_total_query, num_heads, r, head_dim]
            Q_recon = Q_recon_heads.transpose(1, 2).contiguous().view(n_total_query, r, C)  # [n_total_query, r, C]

            reconstructed_query.append(Q_recon)

        # 堆叠所有类: [n_way, n_total_query, r, C]
        reconstructed_query = torch.stack(reconstructed_query, dim=0)

        # ========== (2) 查询集 -> 支持集重构 ==========
        # Ŝ_(i,c) = Attention(S_c^Q, Q_i^K, Q_i^V)

        reconstructed_support = []

        for c in range(n_way):
            # S_c^Q
            S_Q_heads = self.split_heads(S_Q[c:c + 1])  # [1, num_heads, k*r, head_dim]

            # 对每个查询计算
            c_recon_support = []
            for i in range(n_total_query):
                # Q_i^K, Q_i^V
                Q_K_heads = self.split_heads(Q_K[i:i + 1])  # [1, num_heads, r, head_dim]
                Q_V_heads = self.split_heads(Q_V[i:i + 1])  # [1, num_heads, r, head_dim]

                # 计算注意力: S_c^Q @ (Q_i^K)^T / sqrt(d_k)
                attention_scores = torch.matmul(S_Q_heads, Q_K_heads.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attention_weights = F.softmax(attention_scores, dim=-1)  # [1, num_heads, k*r, r]
                attention_weights = self.dropout(attention_weights)

                # 应用注意力到V: attention_weights @ Q_i^V
                S_recon_heads = torch.matmul(attention_weights, Q_V_heads)  # [1, num_heads, k*r, head_dim]
                S_recon = S_recon_heads.transpose(1, 2).contiguous().view(1, k, r, C)  # [1, k, r, C]

                c_recon_support.append(S_recon)

            # 堆叠所有查询: [n_total_query, k, r, C]
            c_recon_support = torch.cat(c_recon_support, dim=0)  # [n_total_query, k, r, C]
            reconstructed_support.append(c_recon_support)

        # 堆叠所有类: [n_way, n_total_query, k, r, C]
        reconstructed_support = torch.stack(reconstructed_support, dim=0)

        # ========== (3) 计算距离 ==========
        # d_Q_i→S_c = ||Q_i^V - Q̂_(c,i)||^2
        # 使用欧氏距离
        Q_V_expanded = Q_V.unsqueeze(0).expand(n_way, -1, -1, -1)  # [n_way, n_total_query, r, C]
        distances_q2s = torch.sum((Q_V_expanded - reconstructed_query) ** 2, dim=[2, 3])  # [n_way, n_total_query]
        distances_q2s = distances_q2s.transpose(0, 1)  # [n_total_query, n_way]

        # d_S_c→Q_i = ||S_c^V - Ŝ_(i,c)||^2
        # 平均所有支持样本的距离
        S_V_expanded = S_V.unsqueeze(1).expand(-1, n_total_query, -1, -1)  # [n_way, n_total_query, k*r, C]
        S_V_expanded = S_V_expanded.view(n_way, n_total_query, k, r, C)

        distances_s2q_per_sample = torch.sum((S_V_expanded - reconstructed_support) ** 2,
                                             dim=[3, 4])  # [n_way, n_total_query, k]
        distances_s2q = distances_s2q_per_sample.mean(dim=2)  # [n_way, n_total_query]
        distances_s2q = distances_s2q.transpose(0, 1)  # [n_total_query, n_way]

        # ========== 关键修改：距离归一化 ==========
        # 除以特征维度，使距离在合理范围（0-1左右）
        feature_dim = r * C
        distances_q2s = distances_q2s / feature_dim
        distances_s2q = distances_s2q / feature_dim

        # ========== (4) 计算总距离 ==========
        # d_i^c = τ(λ1 * d_Q_i→S_c + λ2 * d_S_c→Q_i)
        # 使用softmax确保λ1+λ2=1
        lambda_weights = F.softmax(torch.stack([self.lambda1, self.lambda2]), dim=0)
        total_distances = self.tau * (lambda_weights[0] * distances_q2s + lambda_weights[1] * distances_s2q)

        return reconstructed_query, reconstructed_support, distances_q2s, distances_s2q, total_distances



# ===================== 4. 分类器模块（保持不变） =====================
class Classifier_cosine(nn.Module):
    def __init__(self, n_way, n_support, feat_dim):
        super(Classifier_cosine, self).__init__()
        self.n_way = n_way
        self.feat_dim = feat_dim
        self.n_support = n_support

        # 补全feat_dim至3维
        while len(self.feat_dim) < 3:
            self.feat_dim.append(1)

        self.feat_h, self.feat_w = feat_dim[1], feat_dim[2]
        self.use_pool = self.feat_h > 4 and self.feat_w > 4
        self.use_bn = self.feat_h > 1 and self.feat_w > 1

        # 构建layer1
        layers = [nn.Conv2d(feat_dim[0], feat_dim[0], kernel_size=3, padding=1)]
        if self.use_bn:
            layers.append(nn.BatchNorm2d(feat_dim[0], momentum=1, affine=True))
        layers.append(nn.ReLU())
        if self.use_pool:
            layers.append(nn.MaxPool2d(2))
        self.layer1 = nn.Sequential(*layers)

        # 构建layer2
        layers2 = [nn.Conv2d(feat_dim[0], feat_dim[0], kernel_size=3, padding=1)]
        if self.use_bn:
            layers2.append(nn.BatchNorm2d(feat_dim[0], momentum=1, affine=True))
        layers2.append(nn.ReLU())
        if self.use_pool:
            layers2.append(nn.AvgPool2d(2))
        else:
            layers2.append(nn.AdaptiveAvgPool2d(1))
        self.layer2 = nn.Sequential(*layers2)

        # 模拟前向传播获取展平维度
        self.eval()
        with torch.no_grad():
            dummy = torch.randn(2, *feat_dim)
            out1 = self.layer1(dummy)
            out2 = self.layer2(out1)
            self.flat_dim = out2.size(1) * out2.size(2) * out2.size(3)
        self.train()

    def forward(self, z_support, z_query):
        self.n_query = z_query.size(1)
        extend_final_feat_dim = self.feat_dim.copy()

        z_support = z_support.mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, *self.feat_dim)

        # 扩展维度
        z_support_ext = z_support.unsqueeze(0).repeat(self.n_query * self.n_way, 1, 1, 1, 1)
        z_query_ext = z_query.unsqueeze(0).repeat(self.n_way, 1, 1, 1, 1)
        z_query_ext = torch.transpose(z_query_ext, 0, 1)

        # 展平为卷积层输入
        z_support_ext = z_support_ext.view(-1, *extend_final_feat_dim)
        z_query_ext = z_query_ext.contiguous().view(-1, *extend_final_feat_dim)

        # 卷积层前向
        x_support = self.layer1(z_support_ext)
        x_query = self.layer1(z_query_ext)
        x_support = self.layer2(x_support)
        x_query = self.layer2(x_query)

        # 展平为向量
        batch_size = x_support.size(0)
        x_support_flat = x_support.view(batch_size, -1)
        x_query_flat = x_query.view(batch_size, -1)

        # 计算余弦相似度
        cosine = F.cosine_similarity(x_support_flat, x_query_flat, dim=1)

        return cosine


# ===================== 5. 关系网络模块 =====================
class RelationConvBlock(nn.Module):
    def __init__(self, indim, outdim, padding=0, use_pool=True, pool_type='max'):
        super(RelationConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.use_pool = use_pool
        self.pool_type = pool_type

        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim, momentum=1, affine=True)
        self.relu = nn.ReLU()

        self.parametrized_layers = [self.C, self.BN, self.relu]

        if self.use_pool:
            if self.pool_type == 'max':
                self.pool = nn.MaxPool2d(2)
            elif self.pool_type == 'adaptive':
                self.pool = nn.AdaptiveAvgPool2d(1)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            backbone.init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class RelationModule(nn.Module):
    def __init__(self, input_size, hidden_size, loss_type='softmax'):
        super(RelationModule, self).__init__()
        self.loss_type = loss_type

        while len(input_size) < 3:
            input_size.append(1)

        padding = 1 if (input_size[1] <= 10) and (input_size[2] <= 10) else 0

        def get_conv_size(s, padding):
            return (s - 3 + 2 * padding) + 1

        conv1_size = get_conv_size(input_size[1], padding)
        use_pool1 = conv1_size > 2

        self.layer1 = RelationConvBlock(
            input_size[0] * 2, input_size[0],
            padding=padding,
            use_pool=use_pool1,
            pool_type='max' if use_pool1 else 'adaptive'
        )
        self.layer2 = RelationConvBlock(
            input_size[0], input_size[0],
            padding=padding,
            use_pool=False,
            pool_type='adaptive'
        )

        # 模拟前向传播获取全连接层输入维度
        self.eval()
        with torch.no_grad():
            dummy_input = torch.randn(2, input_size[0] * 2, input_size[1], input_size[2])
            out1 = self.layer1(dummy_input)
            out2 = self.layer2(out1)
            self.fc_in_dim = out2.size(1) * out2.size(2) * out2.size(3)
        self.train()

        # 全连接层
        self.fc1 = nn.Linear(self.fc_in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        backbone.init_layer(self.fc1)
        backbone.init_layer(self.fc2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.loss_type == 'mse':
            out = torch.sigmoid(self.fc2(out))
        elif self.loss_type == 'softmax':
            out = self.fc2(out)
        return out


# ===================== 6. 改进的OurNet主类 =====================
class OurNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, loss_type=None,
                 use_fsrm=True,  # 使用特征自重构模块 FSRM
                 use_fmrm=True,  # 使用特征互重构模块 FMRM
                 attention_heads=8,
                 attention_dropout=0.1,
                 use_position_embedding=True,
                 # ArcFace相关参数
                 use_arcface=True,
                 arcface_margin=0.2,
                 arcface_scale=30.0,
                 arcface_weight=0.5,
                 # 可视化相关
                 enable_visualization=False):  # 是否启用可视化功能
        super(OurNet, self).__init__(model_func, n_way, n_support)

        # 兼容旧的 loss_type 参数
        if loss_type is not None:
            print(f"警告：OurNet 已使用交叉熵+ArcFace损失（传入值：{loss_type}）")

        # 基础交叉熵损失
        self.loss_type = 'softmax'
        self.ce_loss_fn = nn.CrossEntropyLoss()

        # ArcFace损失配置
        self.use_arcface = use_arcface
        self.arcface_weight = arcface_weight
        self.arcface_margin = arcface_margin
        self.arcface_scale = arcface_scale
        if self.use_arcface:
            self.arcface_loss_fn = ArcFaceLoss(
                margin=arcface_margin,
                scale=arcface_scale
            )

        # 论文核心模块配置
        self.use_fsrm = use_fsrm  # FSRM (特征自重构)
        self.use_fmrm = use_fmrm  # FMRM (特征互重构)
        self.enable_visualization = enable_visualization

        # 延迟初始化
        self.classifier_cosine = None
        self.relation_module = None
        self.fsrm = None  # 特征自重构模块
        self.fmrm = None  # 特征互重构模块
        self.visualization_decoder = None  # 可视化解码器
        self.feat_dim = None
        self.device = None

        # 超参数
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.use_position_embedding = use_position_embedding

        # 自适应融合权重
        self.fusion_weights = nn.Parameter(torch.ones(2))

        # 添加距离缩放因子（可学习）
        self.distance_scale = nn.Parameter(torch.tensor(1.0))

    def init_modules(self):
        """初始化所有子模块"""
        if self.feat_dim is None:
            raise ValueError("请先设置feat_dim，再调用init_modules")

        # 1. 初始化基础模块
        self.classifier_cosine = Classifier_cosine(
            n_way=self.n_way,
            n_support=self.n_support,
            feat_dim=self.feat_dim
        )

        self.relation_module = RelationModule(
            input_size=self.feat_dim,
            hidden_size=8,
            loss_type=self.loss_type
        )

        # 2. 初始化论文核心模块
        if self.use_fsrm:
            self.fsrm = FeatureSelfReconstructionModule(
                feat_dim=self.feat_dim,
                num_heads=self.attention_heads,
                dropout=self.attention_dropout,
                use_position_embedding=self.use_position_embedding
            )

        if self.use_fmrm:
            self.fmrm = FeatureMutualReconstructionModule(
                feat_dim=self.feat_dim,
                num_heads=self.attention_heads,
                dropout=self.attention_dropout
            )

       
        # 3. 移动到设备
        self.device = next(self.feature.parameters()).device
        self.classifier_cosine = self.classifier_cosine.to(self.device)
        self.relation_module = self.relation_module.to(self.device)

        if self.fsrm is not None:
            self.fsrm = self.fsrm.to(self.device)

        if self.fmrm is not None:
            self.fmrm = self.fmrm.to(self.device)

        if self.visualization_decoder is not None:
            self.visualization_decoder = self.visualization_decoder.to(self.device)

        # 移动ArcFace损失到设备
        if self.use_arcface and self.arcface_loss_fn is not None:
            self.arcface_loss_fn = self.arcface_loss_fn.to(self.device)

        print(f"OurNet初始化完成:")
        print(f"  - feat_dim={self.feat_dim}")
        print(f"  - 设备={self.device}")
        print(f"  - 使用FSRM（特征自重构）={self.use_fsrm}")
        print(f"  - 使用FMRM（特征互重构）={self.use_fmrm}")
        print(f"  - 使用ArcFace损失={self.use_arcface} (margin={self.arcface_margin}, scale={self.arcface_scale})")
       

    def parameters(self, recurse=True):
        """重写parameters方法，包含所有子模块参数"""
        base_params = list(super().parameters(recurse=recurse))

        if self.classifier_cosine is not None:
            base_params += list(self.classifier_cosine.parameters(recurse=recurse))

        if self.relation_module is not None:
            base_params += list(self.relation_module.parameters(recurse=recurse))

        if self.fsrm is not None:
            base_params += list(self.fsrm.parameters(recurse=recurse))

        if self.fmrm is not None:
            base_params += list(self.fmrm.parameters(recurse=recurse))

        if self.visualization_decoder is not None:
            base_params += list(self.visualization_decoder.parameters(recurse=recurse))

        base_params.append(self.fusion_weights)
        base_params.append(self.distance_scale)

        return iter(base_params)

    def parse_feature(self, x, is_feature=False):
        """特征提取和模块初始化"""
        if is_feature:
            z_all = x
            self.feat_dim = list(z_all.size()[2:])
            while len(self.feat_dim) < 3:
                self.feat_dim.append(1)

            if self.classifier_cosine is None or self.relation_module is None:
                self.init_modules()
        else:
            n_way = x.size(0)
            n_samples = x.size(1)
            x_flat = x.view(-1, *x.size()[2:])

            if hasattr(self, 'device') and self.device is not None:
                x_flat = x_flat.to(self.device)
            else:
                self.device = next(self.feature.parameters()).device
                x_flat = x_flat.to(self.device)

            z_all = self.feature(x_flat)
            self.feat_dim = list(z_all.size()[1:])

            while len(self.feat_dim) < 3:
                self.feat_dim.append(1)

            if self.classifier_cosine is None or self.relation_module is None:
                self.init_modules()

            z_all = z_all.view(n_way, n_samples, *self.feat_dim)

        z_support = z_all[:, :self.n_support, ...]
        z_query = z_all[:, self.n_support:, ...]

        return z_support, z_query

    def set_forward(self, x, is_feature=False, return_reconstruction=False):
        # 1. 特征提取
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support_save = z_support.clone()
        z_query_save = z_query.clone()

        z_support = z_support.contiguous()
        self.n_query = z_query.size(1)
        n_total_query = self.n_way * self.n_query

        # 2. 展平特征用于FSRM和FMRM
        # [n_way, n_support, C, H, W] -> [n_way * n_support, C, H, W]
        support_flat = z_support.view(-1, *self.feat_dim)
        # [n_way, n_query, C, H, W] -> [n_total_query, C, H, W]
        query_flat = z_query.contiguous().view(n_total_query, *self.feat_dim)

        # 3. FSRM: 特征自重构
        if self.use_fsrm and self.fsrm is not None:
            # 对支持集进行自重构
            support_fsrm_out = self.fsrm(support_flat)  # [n_way*n_support, r, C]
            # 对查询集进行自重构
            query_fsrm_out = self.fsrm(query_flat)  # [n_total_query, r, C]

            # 重塑回原始格式
            # 支持集: [n_way*n_support, r, C] -> [n_way, n_support, r, C]
            support_fsrm_out = support_fsrm_out.view(self.n_way, self.n_support, -1, self.feat_dim[0])
            # 查询集: [n_total_query, r, C] -> [n_way, n_query, r, C]
            query_fsrm_out = query_fsrm_out.view(self.n_way, self.n_query, -1, self.feat_dim[0])
        else:
            # 如果不使用FSRM，将特征转换为序列格式
            support_fsrm_out = support_flat.flatten(2).transpose(1, 2)  # [n_way*n_support, r, C]
            support_fsrm_out = support_fsrm_out.view(self.n_way, self.n_support, -1, self.feat_dim[0])

            query_fsrm_out = query_flat.flatten(2).transpose(1, 2)  # [n_total_query, r, C]
            query_fsrm_out = query_fsrm_out.view(self.n_way, self.n_query, -1, self.feat_dim[0])

        # 4. FMRM: 特征互重构
        if self.use_fmrm and self.fmrm is not None:
            # 展平查询特征用于FMRM
            query_for_fmrm = query_fsrm_out.view(n_total_query, -1, self.feat_dim[0])  # [n_total_query, r, C]

            # 双向重构
            reconstructed_query, reconstructed_support, distances_q2s, distances_s2q, total_distances = self.fmrm(
                support_fsrm_out,  # [n_way, n_support, r, C]
                query_for_fmrm  # [n_total_query, r, C]
            )

           
            scaled_distances = total_distances * self.distance_scale * 5
            final_scores = -scaled_distances  # 距离越小，相似度越高
        else:
            # 如果不使用FMRM，回退到传统方法
            final_scores = None
            reconstructed_query = None
            reconstructed_support = None

        # 5. 计算关系分数
        z_proto = support_fsrm_out.mean(dim=1)  # [n_way, r, C]

        # 将序列转换回特征图用于关系网络
        if self.use_fsrm:
            # 将FSRM输出转回特征图
            r, C = z_proto.shape[1], z_proto.shape[2]
            H = W = int(math.sqrt(r))
            if H * W == r:
                z_proto_img = z_proto.transpose(1, 2).view(self.n_way, C, H, W)
            else:
                # 如果不是正方形，使用插值
                z_proto_img = z_proto.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        else:
            z_proto_img = z_proto.mean(dim=1).unsqueeze(-1).unsqueeze(-1)

        # 展平查询特征用于关系网络
        if self.use_fsrm:
            query_for_rel = query_fsrm_out.view(n_total_query, -1, self.feat_dim[0])
            if H * W == query_for_rel.shape[1]:
                query_img = query_for_rel.transpose(1, 2).view(n_total_query, C, H, W)
            else:
                query_img = query_for_rel.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        else:
            query_img = query_flat

        # 扩展维度以计算关系
        z_proto_expand = z_proto_img.unsqueeze(0).repeat(n_total_query, 1, 1, 1, 1)
        query_expand = query_img.unsqueeze(1).repeat(1, self.n_way, 1, 1, 1)

        # 拼接为关系对
        relation_pairs = torch.cat([z_proto_expand, query_expand], dim=2)
        relation_pairs = relation_pairs.view(-1, 2 * self.feat_dim[0], self.feat_dim[1], self.feat_dim[2])

        # 关系网络计算
        relations = self.relation_module(relation_pairs)
        relations = relations.view(n_total_query, self.n_way)

        # 6. Cosine分类器（使用原始特征）
        cosine = self.classifier_cosine(z_support_save, z_query_save).view(-1, self.n_way)

        # 7. 自适应融合
        if final_scores is None:
            # 如果没有FMRM，只使用关系和余弦
            fusion_weights = F.softmax(self.fusion_weights, dim=0)
            final_scores = fusion_weights[0] * relations + fusion_weights[1] * cosine
        else:
            # 融合FMRM、关系和余弦
            fusion_weights = F.softmax(self.fusion_weights, dim=0)
            final_scores = (0.4 * final_scores + 0.3 * relations + 0.3 * cosine)

        # 8. 如果需要返回重构特征（用于可视化）
        if return_reconstruction:
            return final_scores, relations, cosine, {
                'support_original': support_fsrm_out,  # 原始支持特征
                'query_original': query_fsrm_out,  # 原始查询特征
                'reconstructed_query': reconstructed_query,  # 重构的查询特征 Q̂_(c,i)
                'reconstructed_support': reconstructed_support,  # 重构的支持特征 Ŝ_(i,c)
                'distances_q2s': distances_q2s,  # 查询->支持距离
                'distances_s2q': distances_s2q,  # 支持->查询距离
                'support_flat': support_flat,  # 原始支持特征图
                'query_flat': query_flat  # 原始查询特征图
            }
        else:
            return final_scores, relations, cosine
    def correct(self, x):
        """重写correct方法"""
        scores, _, _ = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()

        top1_correct = np.sum(topk_ind[:, 0] == y_query)

        return float(top1_correct), len(y_query)

    def set_forward_loss(self, x):
        """计算融合了ArcFace的损失"""
        self.n_query = x.size(1) - self.n_support

        if self.n_query <= 0:
            raise ValueError(
                f"Invalid episode setting: "
                f"x.size(1)={x.size(1)}, "
                f"n_support={self.n_support}, "
                f"n_query={self.n_query}"
            )

        y = torch.from_numpy(
            np.repeat(range(self.n_way), self.n_query)
        ).long()

        final_scores, _, cosine_scores = self.set_forward(x)

        device = final_scores.device
        y = y.to(device)

        ce_loss = self.ce_loss_fn(final_scores, y)

        if self.use_arcface and self.arcface_loss_fn is not None:
            cosine_normalized = F.normalize(cosine_scores, dim=1)
            arcface_loss = self.arcface_loss_fn(cosine_normalized, y, n_support=self.n_support)
            total_loss = (1 - self.arcface_weight) * ce_loss + self.arcface_weight * arcface_loss
        else:
            total_loss = ce_loss

        _, preds = torch.max(final_scores, dim=1)
        acc = (preds == y).float().mean().item()

        return total_loss, acc

    def train_loop(self, epoch, train_loader, optimizer):
        self.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for i, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()

            loss, acc = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += acc * x.size(0)
            total_count += x.size(0)

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_count

        return avg_loss, avg_acc

    def test_loop(self, test_loader):
        self.eval()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):
                loss, acc = self.set_forward_loss(x)

                total_loss += loss.item()
                total_correct += acc * x.size(0)
                total_count += x.size(0)

        avg_loss = total_loss / len(test_loader)
        avg_acc = total_correct / total_count

        return avg_loss, avg_acc


# ===================== 8. 辅助函数 =====================
def create_ournet_model(backbone_name, n_way, n_support, **kwargs):
    """创建OurNet模型的工厂函数"""
    from methods.ournet import OurNet

    # 根据backbone名称获取模型函数
    if backbone_name == 'conv4np':
        model_func = backbone.Conv4NP()
    elif backbone_name == 'resnet12':
        model_func = backbone.ResNet12()
    else:
        raise ValueError(f"不支持的骨干网络: {backbone_name}")

    # 过滤无关参数
    filtered_kwargs = {k: v for k, v in kwargs.items() if
                       not (k.startswith('triplet_') or k in ['ce_weight', 'triplet_weight'])}

    # 创建OurNet实例
    model = OurNet(
        model_func=model_func,
        n_way=n_way,
        n_support=n_support,
        **filtered_kwargs
    )

    return model


# ===================== 9. 配置类 =====================
class OurNetConfig:
    """OurNet配置类"""

    def __init__(self):
        # 基础配置
        self.loss_type = 'softmax'

        # 论文核心模块配置
        self.use_fsrm = True  # 使用FSRM（特征自重构）
        self.use_fmrm = True  # 使用FMRM（特征互重构）

        # 注意力配置
        self.attention_heads = 8
        self.attention_dropout = 0.1
        self.use_position_embedding = True

        # ArcFace配置
        self.use_arcface = True
        self.arcface_margin = 0.2
        self.arcface_scale = 30.0
        self.arcface_weight = 0.25

        # 可视化配置
        self.enable_visualization = False  # 训练时关闭，测试时可按需开启

        # 训练配置
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.optimizer = 'Adam'

    def to_dict(self):
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ===================== 10. 可视化测试函数 =====================
def visualize_reconstruction_example(model, test_loader, save_path='reconstruction_viz.png'):
    """
    可视化重构特征的示例函数

    Args:
        model: 训练好的OurNet模型
        test_loader: 测试数据加载器
        save_path: 保存路径
    """
    model.eval()

    # 获取一个batch数据
    for x, _ in test_loader:
        x = x.to(next(model.parameters()).device)

        # 调用可视化函数
        vis_data = model.visualize_reconstruction(x, save_path=save_path)

        if vis_data:
            print("可视化完成！")
            print(f"距离矩阵 d_Q_i→S_c 形状: {vis_data['distances_q2s'].shape}")
            print(f"距离矩阵 d_S_c→Q_i 形状: {vis_data['distances_s2q'].shape}")
        break
    print("4. ArcFaceLoss - 适配小样本的ArcFace损失")
    print("5. OurNet - 主网络类（完整实现Bi-FRN）")
    print("\n主要改进:")
    print("- 支持双向重构: S_c -> Q_i (增加类间差异) 和 Q_i -> S_c (减少类内差异)")
    print("- 支持图6的可视化效果")
    print("- 可学习的λ1、λ2权重和温度因子τ")
    print("- 兼容ArcFace损失")
