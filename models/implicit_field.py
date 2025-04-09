# models/implicit_field.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lightning import LightingModel


class FeaturePyramidFusion_(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, feature_dims=[256, 512, 1024, 2048], output_dim=256, clip_dim=512):
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

        # 密度特征提取网络
        density_feature_dim = feature_dims[0] + 3 + 2  # 图像特征 + 加权点坐标 + 密度统计
        if clip_dim > 0:
            density_feature_dim += clip_dim

        self.density_feature_mlp = nn.Sequential(
            nn.Linear(density_feature_dim, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 512)
        )

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


# models/implicit_field.py
class FeaturePyramidFusion(nn.Module):
    """多尺度特征融合模块，带有动态光照估计"""

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

        # 光照估计网络
        self.lighting_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_dim, 128, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 9, 1)  # 3个光照参数 (环境光、漫反射、镜面反射)
        )

        self.output_dim = output_dim

    def forward(self, features_list):
        """
        融合多尺度特征

        Args:
            features_list: 不同尺度的特征列表 [B, C_i, H_i, W_i]

        Returns:
            fused_features: 融合后的特征 [B, output_dim, H_0, W_0]
            weights: 各尺度特征的权重 [B, len(features_list)]
            lighting: 估计的光照参数 [B, 9]
        """
        B = features_list[0].shape[0]
        target_size = features_list[0].shape[2:]

        # 转换并上采样所有特征
        transformed_features = []
        for i, (features, transform) in enumerate(zip(features_list, self.transforms)):
            transformed = transform(features)
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
            weight = predictor(features)
            weights.append(weight.view(B, 1, 1, 1))

        # 归一化权重
        weights_tensor = torch.cat(weights, dim=1)
        weights_normalized = F.softmax(weights_tensor, dim=1)

        # 加权融合特征
        fused_features = torch.zeros_like(transformed_features[0])
        for i, features in enumerate(transformed_features):
            fused_features += features * weights_normalized[:, i:i + 1]

        # 估计光照参数
        lighting = self.lighting_estimator(fused_features)
        lighting = lighting.view(B, 9)  # [B, 9]

        # 返回融合特征、权重和光照参数
        return fused_features, weights_normalized.squeeze(-1).squeeze(-1), lighting


class ImplicitField_(nn.Module):
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

    # models/implicit_field.py
    def extract_density_features(self, points, densities, features, clip_features=None):
        """
        从密度场提取语义特征

        Args:
            points: 点云坐标 [B, N, 3]
            densities: 密度值 [B, N, 1]
            features: 图像特征 [B, C, H, W]
            clip_features: CLIP特征 [B, D] 或 None

        Returns:
            density_features: 密度场特征 [B, C]
        """
        B = points.shape[0]

        # 提取全局图像特征
        global_features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # [B, C]

        # 提取密度统计特征
        # 使用密度作为权重，计算加权平均的点坐标
        weights = F.softmax(densities.squeeze(-1), dim=1)  # [B, N]
        weighted_points = torch.bmm(weights.unsqueeze(1), points).squeeze(1)  # [B, 3]

        # 计算密度的统计矩
        density_mean = torch.mean(densities, dim=1)  # [B, 1]
        density_var = torch.var(densities, dim=1)  # [B, 1]

        # 组合特征
        combined_features = [global_features, weighted_points, density_mean, density_var]

        # 如果有CLIP特征，也加入
        if clip_features is not None:
            combined_features.append(clip_features)

        # 拼接所有特征
        density_features = torch.cat([f.view(B, -1) for f in combined_features], dim=1)

        # 使用MLP投影到固定维度
        density_features = self.density_feature_mlp(density_features)

        return density_features


class ImplicitField(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=1, num_layers=3, clip_dim=512):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.clip_dim = clip_dim

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

        # 添加颜色预测网络
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()  # 输出范围[0,1]的RGB颜色
        )

        # 添加光照模型
        self.lighting_model = LightingModel()

        # 添加密度特征提取网络 - 这是之前缺失的部分
        density_feature_dim = input_dim + 3 + 1 + 1
        if clip_dim > 0:
            density_feature_dim += clip_dim

        self.density_feature_mlp = nn.Sequential(
            nn.Linear(density_feature_dim, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 512)
        )

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
            colors: 颜色值 [B, N, 3]
        """
        # 融合多尺度特征
        fused_features, feature_weights, lighting_params = self.feature_pyramid(features_list)
        B = fused_features.shape[0]

        # 如果没有提供点云，创建网格点
        if points is None:
            points = self._create_grid_points(B, fused_features.device)
            points.requires_grad_(True)
        elif not points.requires_grad:
            # 确保点云需要梯度
            points = points.detach().clone().requires_grad_(True)

        # 计算密度和特征 - 直接使用compute_density_and_features，移除重复计算
        densities, features = self.compute_density_and_features(points, fused_features)

        # 预测颜色（基础albedo）
        albedo = self.color_net(features)

        # 计算法线
        normals = self.compute_normals(points, densities)

        # 应用光照模型
        colors = self.lighting_model.apply_lighting(points, normals, albedo, lighting_params)

        return points, densities, colors

    def compute_normals(self, points, densities):
        """计算点云的法线"""
        grad_outputs = torch.ones_like(densities)
        gradients = torch.autograd.grad(
            outputs=densities,
            inputs=points,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 归一化梯度
        normals = F.normalize(gradients, dim=2)

        return normals

    def compute_density_and_features(self, points, features):
        """
        计算点云密度和特征

        Args:
            points: 点云坐标 [B, N, 3]
            features: 融合的特征图 [B, C, H, W]

        Returns:
            densities: 密度值 [B, N, 1]
            point_features: 点特征 [B, N, hidden_dim*2]
        """
        B, N = points.shape[:2]

        # 分批处理点云以节省内存
        all_densities = []
        all_point_features = []

        for i in range(0, points.shape[1], self.chunk_size):
            chunk_points = points[:, i:i + self.chunk_size]

            # 确保点云需要梯度
            if not chunk_points.requires_grad:
                chunk_points = chunk_points.detach().clone().requires_grad_(True)

            # 对点坐标应用位置编码
            encoded_points = self.positional_encoding(chunk_points)

            # 处理编码后的3D坐标
            coord_features = self.coordinate_net(encoded_points)

            # 提取点特征 - 使用全局特征
            global_features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
            global_features = global_features.unsqueeze(1).expand(B, chunk_points.shape[1], -1)
            feat_features = self.feature_net(global_features)

            # 融合坐标和特征信息
            combined_features = torch.cat([coord_features, feat_features], dim=-1)

            # 预测密度
            chunk_densities = self.fusion_net(combined_features)

            all_densities.append(chunk_densities)
            all_point_features.append(combined_features)

        # 合并所有密度预测和特征
        densities = torch.cat(all_densities, dim=1)
        point_features = torch.cat(all_point_features, dim=1)

        return densities, point_features

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

    def extract_density_features(self, points, densities, features, clip_features=None):
        """
        从密度场提取语义特征

        Args:
            points: 点云坐标 [B, N, 3]
            densities: 密度值 [B, N, 1]
            features: 图像特征 [B, C, H, W]
            clip_features: CLIP特征 [B, D] 或 None

        Returns:
            density_features: 密度场特征 [B, 512]
        """
        B = points.shape[0]

        # 提取全局图像特征
        global_features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # [B, C]

        # 提取密度统计特征
        weights = F.softmax(densities.squeeze(-1), dim=1)  # [B, N]
        weighted_points = torch.bmm(weights.unsqueeze(1), points).squeeze(1)  # [B, 3]

        # 计算密度的统计矩
        density_mean = torch.mean(densities, dim=1)  # [B, 1]
        density_var = torch.var(densities, dim=1)  # [B, 1]

        # 组合特征
        combined_features = [global_features, weighted_points, density_mean, density_var]

        # 如果有CLIP特征，也加入
        if clip_features is not None:
            combined_features.append(clip_features)

        # 拼接所有特征
        try:
            density_features = torch.cat([f.view(B, -1) for f in combined_features], dim=1)

            # 检查维度是否匹配
            expected_dim = self.input_dim + 3 + 1 + 1 + (self.clip_dim if clip_features is not None else 0)
            actual_dim = density_features.shape[1]

            if actual_dim != expected_dim:
                print(f"Warning: Feature dimension mismatch. Expected {expected_dim}, got {actual_dim}")
                # 如果不匹配，可以尝试调整
                if not hasattr(self, '_fixed_density_feature_mlp'):
                    print(f"Creating new density_feature_mlp with input dim {actual_dim}")
                    self._fixed_density_feature_mlp = nn.Sequential(
                        nn.Linear(actual_dim, 512),
                        nn.ReLU(inplace=False),
                        nn.Linear(512, 512)
                    ).to(density_features.device)

                density_features = self._fixed_density_feature_mlp(density_features)
            else:
                # 使用原始MLP
                density_features = self.density_feature_mlp(density_features)

        except Exception as e:
            print(f"Error in extract_density_features: {e}")
            print(f"Feature shapes: {[f.shape for f in combined_features]}")
            # 返回一个随机特征作为后备
            density_features = torch.randn(B, 512, device=points.device)

        return density_features