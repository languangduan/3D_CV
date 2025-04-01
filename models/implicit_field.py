# models/implicit_field.py
import torch
import torch.nn as nn


class ImplicitField(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=1, num_layers=4):
        super().__init__()

        self.input_dim = input_dim
        layers = []

        # 第一层
        layers.append(nn.Linear(input_dim + 3, hidden_dim))  # +3 for xyz coordinates
        layers.append(nn.ReLU(inplace=True))

        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features, points=None):
        """
        Args:
            features: 特征体积 [B, C, D, H, W]
            points: 采样点坐标 [B, N, 3] 如果为None，则自动生成网格点
        """
        B = features.shape[0]

        if points is None:
            # 创建均匀网格点
            grid_size = 64
            x = torch.linspace(-1, 1, grid_size)
            y = torch.linspace(-1, 1, grid_size)
            z = torch.linspace(-1, 1, grid_size)

            xx, yy, zz = torch.meshgrid(x, y, z)
            points = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
            points = points.unsqueeze(0).repeat(B, 1, 1).to(features.device)

        # 对每个点进行特征插值
        point_features = self._sample_features(features, points)

        # 连接点坐标和特征
        point_input = torch.cat([points, point_features], dim=-1)

        # 通过MLP预测密度
        densities = self.mlp(point_input)

        return points, densities

    def _sample_features(self, features, points):
        """在特征体积中采样点特征"""
        # 将points从[-1,1]映射到[0,1]
        grid_points = (points + 1) / 2

        # 重塑features以适应grid_sample
        B, C, D, H, W = features.shape
        features_flat = features.view(B, C, D, H * W)

        # 使用grid_sample进行特征插值
        sampled_features = F.grid_sample(
            features_flat,
            grid_points.view(B, 1, 1, -1, 3),
            mode='bilinear',
            align_corners=True
        )

        return sampled_features.view(B, C, -1).permute(0, 2, 1)
