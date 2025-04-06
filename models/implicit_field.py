# models/implicit_field.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, feature_dims=[256, 512, 1024, 2048], output_dim=256):
        super().__init__()

        # 特征转换层
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, output_dim, 1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=False)
            ) for dim in feature_dims
        ])

        # 特征权重预测
        self.weight_predictors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(output_dim, 1, 1),
                nn.Sigmoid()
            ) for _ in feature_dims
        ])

        self.output_dim = output_dim

    def forward(self, features_list):
        """
        融合多尺度特征

        Args:
            features_list: 不同尺度的特征列表 [B, C_i, H_i, W_i]

        Returns:
            fused_features: 融合后的特征 [B, output_dim, H_0, W_0]
            weights: 各尺度特征的权重 [B, len(features_list)]
        """
        B = features_list[0].shape[0]
        target_size = features_list[0].shape[2:]  # 使用最高分辨率作为目标尺寸

        # 转换并上采样所有特征
        transformed_features = []
        for i, (features, transform) in enumerate(zip(features_list, self.transforms)):
            # 转换特征维度
            transformed = transform(features)

            # 如果不是最高分辨率，进行上采样
            if i > 0:
                transformed = F.interpolate(
                    transformed,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )

            transformed_features.append(transformed)

        # 预测每个尺度的权重
        weights = []
        for i, (features, predictor) in enumerate(zip(transformed_features, self.weight_predictors)):
            weight = predictor(features)  # [B, 1, 1, 1]
            weights.append(weight.view(B, 1, 1, 1))

        # 归一化权重
        weights_tensor = torch.cat(weights, dim=1)  # [B, N, 1, 1]
        weights_normalized = F.softmax(weights_tensor, dim=1)

        # 加权融合特征
        fused_features = torch.zeros_like(transformed_features[0])
        for i, features in enumerate(transformed_features):
            fused_features += features * weights_normalized[:, i:i + 1]

        # 返回融合特征和权重（用于可视化）
        return fused_features, weights_normalized.squeeze(-1).squeeze(-1)


class ImplicitField(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=1, num_layers=3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 特征金字塔融合
        self.feature_pyramid = FeaturePyramidFusion(
            feature_dims=[512, 1024, 2048],  # ResNet的C3, C4, C5特征
            output_dim=input_dim
        )

        # 位置编码 - 增强坐标表示
        self.num_encoding_functions = 6
        self.pos_enc_dim = 3 * (1 + 2 * self.num_encoding_functions)

        # 坐标编码网络 - 处理编码后的3D点坐标
        self.coordinate_net = nn.Sequential(
            nn.Linear(self.pos_enc_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False)
        )

        # 特征处理网络 - 处理图像特征
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False)
        )

        # 融合网络 - 结合坐标和特征信息
        fusion_layers = []
        fusion_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        fusion_layers.append(nn.ReLU(inplace=False))

        for _ in range(num_layers - 2):
            fusion_layers.append(nn.Linear(hidden_dim, hidden_dim))
            fusion_layers.append(nn.ReLU(inplace=False))

        fusion_layers.append(nn.Linear(hidden_dim, output_dim))

        self.fusion_net = nn.Sequential(*fusion_layers)

        # 网格配置
        self.grid_size = 32
        self.chunk_size = 8192

    def positional_encoding(self, x):
        """
        对输入坐标应用位置编码

        Args:
            x: 输入坐标 [B, N, 3]

        Returns:
            encoded: 编码后的坐标 [B, N, pos_enc_dim]
        """
        funcs = [torch.sin, torch.cos]

        # 原始坐标
        out = [x]

        # 添加位置编码项
        for i in range(self.num_encoding_functions):
            freq = 2.0 ** i
            for func in funcs:
                out.append(func(freq * x))

        return torch.cat(out, dim=-1)

    def forward(self, features_list, points=None):
        """
        前向传播函数

        Args:
            features_list: 特征金字塔列表 [C3, C4, C5]，每个元素形状为[B, C_i, H_i, W_i]
            points: 可选的点云坐标 [B, N, 3]，如果为None则创建网格点

        Returns:
            points: 点云坐标 [B, N, 3]
            densities: 密度值 [B, N, 1]
        """
        # 融合多尺度特征
        fused_features, feature_weights = self.feature_pyramid(features_list)
        B = fused_features.shape[0]

        # 如果没有提供点云，创建网格点
        if points is None:
            points = self._create_grid_points(B, fused_features.device)
            points.requires_grad_(True)
        elif not points.requires_grad:
            # 确保点云需要梯度
            points = points.detach().clone().requires_grad_(True)

        # 分批处理点云以节省内存
        all_densities = []
        for i in range(0, points.shape[1], self.chunk_size):
            chunk_points = points[:, i:i + self.chunk_size]
            chunk_densities = self.compute_density(chunk_points, fused_features)
            all_densities.append(chunk_densities)

        # 合并所有密度预测
        densities = torch.cat(all_densities, dim=1)

        return points, densities

    def compute_density(self, points, features):
        """
        计算点云密度

        Args:
            points: 点云坐标 [B, N, 3]
            features: 融合的特征图 [B, C, H, W]

        Returns:
            densities: 密度值 [B, N, 1]
        """
        B, N = points.shape[:2]

        # 确保点云需要梯度
        if not points.requires_grad:
            points = points.detach().clone().requires_grad_(True)

        # 对点坐标应用位置编码
        encoded_points = self.positional_encoding(points)

        # 处理编码后的3D坐标
        coord_features = self.coordinate_net(encoded_points)

        # 提取点特征 - 使用全局特征
        global_features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        global_features = global_features.unsqueeze(1).expand(B, N, -1)
        feat_features = self.feature_net(global_features)

        # 融合坐标和特征信息
        combined_features = torch.cat([coord_features, feat_features], dim=-1)
        densities = self.fusion_net(combined_features)

        return densities

    def _create_grid_points(self, batch_size, device):
        """
        创建均匀网格点

        Args:
            batch_size: 批次大小
            device: 设备

        Returns:
            points: 网格点坐标 [B, N, 3]
        """
        # 生成网格点
        x = torch.linspace(-1, 1, self.grid_size, device=device)
        y = torch.linspace(-1, 1, self.grid_size, device=device)
        z = torch.linspace(-1, 1, self.grid_size, device=device)

        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
        points = points.unsqueeze(0).expand(batch_size, -1, -1)

        return points
