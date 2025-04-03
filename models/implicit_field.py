# models/implicit_field.py
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            features: 特征体积 [B, C, D, H, W] 或 [B, C, H, W]
            points: 采样点坐标 [B, N, 3] 如果为None，则自动生成网格点
        """
        B = features.shape[0]

        # 检查特征维度，处理单视角情况
        if len(features.shape) == 4:  # [B, C, H, W]
            C, H, W = features.shape[1:]
            # 将2D特征转换为3D特征体积
            D = H  # 假设深度维度与高度相同
            features = features.unsqueeze(2).expand(B, C, D, H, W)

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

        # 获取特征维度
        B, C, D, H, W = features.shape

        # 重新排列points以匹配grid_sample的要求
        # grid_sample期望的grid形状为[B, D, H, W, 3]
        # 我们需要将[B, N, 3]重塑为[B, D, H, W, 3]
        N = points.shape[1]

        # 计算采样网格的维度
        sample_D = int(N ** (1 / 3))
        sample_H = sample_D
        sample_W = sample_D

        # 确保维度乘积与N接近
        while sample_D * sample_H * sample_W < N:
            sample_W += 1

        # 如果N不是立方数，我们可能需要填充一些点
        padding_size = sample_D * sample_H * sample_W - N
        if padding_size > 0:
            padding = torch.zeros(B, padding_size, 3, device=points.device)
            points_padded = torch.cat([points, padding], dim=1)
        else:
            points_padded = points

        # 重塑为[B, D, H, W, 3]
        grid = points_padded.reshape(B, sample_D, sample_H, sample_W, 3)

        # 使用grid_sample进行特征插值
        sampled_features = F.grid_sample(
            features,
            grid,
            mode='bilinear',
            align_corners=True
        )

        # 重塑回[B, N, C]
        sampled_features = sampled_features.permute(0, 1, 2, 3, 4).reshape(B, C, -1)
        sampled_features = sampled_features[:, :, :N].permute(0, 2, 1)

        return sampled_features