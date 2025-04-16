# models/implicit_field.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lightning import LightingModel

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


class ImplicitField(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=1, num_layers=3, clip_dim=512):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.clip_dim = clip_dim

        # 特征金字塔融合
        # self.feature_pyramid = FeaturePyramidFusion(
        #     feature_dims=[512, 1024, 2048],  # ResNet的C3, C4, C5特征
        #     output_dim=input_dim
        # )

        self.feature_pyramid = FeaturePyramidFusion(
            feature_dims=[256, 256, 256],  # 修改为FPN输出通道数
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
        """
        计算点云的法线

        Args:
            points: 点云坐标 [B, N, 3]
            densities: 点云密度 [B, N, 1]
        Returns:
            normals: 法线向量 [B, N, 3]
        """
        # 检查是否需要梯度
        needs_grad = points.requires_grad
        original_points = points

        # 如果不需要梯度，临时启用
        if not needs_grad:
            points = points.detach().requires_grad_(True)

        try:
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

            # 如果原始点不需要梯度，确保结果也不需要梯度
            if not needs_grad:
                normals = normals.detach()

            return normals

        except Exception as e:
            print(f"计算法线时出错: {e}")
            # 返回随机法线作为后备
            return torch.nn.functional.normalize(torch.randn_like(original_points), dim=2).detach()

    def compute_density_and_features(self, points, features):
        """
        计算点云密度和特征（增强数值稳定性）

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

            try:
                # 对点坐标应用位置编码
                encoded_points = self.positional_encoding(chunk_points)

                # 检查编码是否有效
                if torch.isnan(encoded_points).any() or torch.isinf(encoded_points).any():
                    print(f"警告: 位置编码中检测到NaN/Inf，进行替换")
                    encoded_points = torch.where(
                        torch.isnan(encoded_points) | torch.isinf(encoded_points),
                        torch.zeros_like(encoded_points),
                        encoded_points
                    )

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

                # 数值稳定性处理 - 限制密度值范围
                chunk_densities = torch.clamp(chunk_densities, min=-10.0, max=10.0)

                # 检查并替换NaN/Inf
                if torch.isnan(chunk_densities).any() or torch.isinf(chunk_densities).any():
                    print(f"警告: 密度计算中检测到NaN/Inf，进行替换")
                    chunk_densities = torch.where(
                        torch.isnan(chunk_densities) | torch.isinf(chunk_densities),
                        torch.zeros_like(chunk_densities),
                        chunk_densities
                    )

                all_densities.append(chunk_densities)
                all_point_features.append(combined_features)

            except Exception as e:
                print(f"密度计算错误: {e}")
                # 创建安全的替代值
                safe_densities = torch.zeros((B, chunk_points.shape[1], 1), device=points.device)
                safe_features = torch.zeros((B, chunk_points.shape[1], self.hidden_dim * 2), device=points.device)

                all_densities.append(safe_densities)
                all_point_features.append(safe_features)

        # 合并所有密度预测和特征
        try:
            densities = torch.cat(all_densities, dim=1)
            point_features = torch.cat(all_point_features, dim=1)

            # 最终检查
            if torch.isnan(densities).any() or torch.isinf(densities).any():
                print(f"警告: 合并后的密度中检测到NaN/Inf，进行最终替换")
                densities = torch.where(
                    torch.isnan(densities) | torch.isinf(densities),
                    torch.zeros_like(densities),
                    densities
                )
        except Exception as e:
            print(f"合并密度和特征时出错: {e}")
            # 创建安全的替代值
            densities = torch.zeros((B, N, 1), device=points.device)
            point_features = torch.zeros((B, N, self.hidden_dim * 2), device=points.device)

        return densities, point_features


    def compute_density_and_features_(self, points, features):
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

        # 打印输入形状以便调试
        # print(f"DEBUG: points shape: {points.shape}")
        # print(f"DEBUG: densities shape: {densities.shape}")
        # print(f"DEBUG: features shape: {features.shape}")
        # if clip_featsures is not None:
        #     print(f"DEBUG: clip_features shape: {clip_features.shape}")

        try:
            # 提取全局图像特征
            if len(features.shape) == 4:  # [B, C, H, W]
                global_features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # [B, C]
                # print(f"DEBUG: global_features shape: {global_features.shape}")
            else:  # 如果已经是 [B, C]
                global_features = features
                # print(f"DEBUG: global_features shape (already flattened): {global_features.shape}")

            # 提取密度统计特征
            weights = F.softmax(densities.squeeze(-1), dim=1)  # [B, N]
            weighted_points = torch.bmm(weights.unsqueeze(1), points).squeeze(1)  # [B, 3]
            # print(f"DEBUG: weighted_points shape: {weighted_points.shape}")

            # 计算密度的统计矩
            density_mean = torch.mean(densities, dim=1)  # [B, 1]
            density_var = torch.var(densities, dim=1)  # [B, 1]
            # print(f"DEBUG: density_mean shape: {density_mean.shape}")
            # print(f"DEBUG: density_var shape: {density_var.shape}")

            # 组合特征
            combined_features = []

            # 添加全局特征
            combined_features.append(global_features)

            # 添加加权点坐标
            combined_features.append(weighted_points)

            # 添加密度统计
            combined_features.append(density_mean)
            combined_features.append(density_var)

            # 如果有CLIP特征，也加入
            if clip_features is not None:
                combined_features.append(clip_features)

            # 打印每个特征的形状
            # for i, feat in enumerate(combined_features):
            #     # print(f"DEBUG: Feature {i} shape: {feat.shape}")

            # 确保所有特征都是2D张量 [B, D]
            for i in range(len(combined_features)):
                if len(combined_features[i].shape) > 2:
                    combined_features[i] = combined_features[i].view(B, -1)
                elif len(combined_features[i].shape) < 2:
                    combined_features[i] = combined_features[i].view(B, -1)

            # 拼接所有特征
            density_features = torch.cat([f for f in combined_features], dim=1)
            actual_dim = density_features.shape[1]
            # print(f"DEBUG: Concatenated density_features shape: {density_features.shape}")

            # 使用动态MLP处理任何维度的输入
            if not hasattr(self, '_dynamic_density_mlp') or self._dynamic_density_mlp[0].in_features != actual_dim:
                # print(f"Creating dynamic density_feature_mlp with input dim {actual_dim}")
                self._dynamic_density_mlp = nn.Sequential(
                    nn.Linear(actual_dim, 512),
                    nn.ReLU(inplace=False),
                    nn.Linear(512, 512)
                ).to(density_features.device)

            # 使用动态MLP
            density_features = self._dynamic_density_mlp(density_features)
            # print(f"DEBUG: Final density_features shape: {density_features.shape}")

        except Exception as e:
            print(f"Error in extract_density_features: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个随机特征作为后备
            density_features = torch.randn(B, 512, device=points.device)
            print("WARNING: Returning random features due to error")

        return density_features
