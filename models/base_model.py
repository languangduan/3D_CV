# models/base_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class SingleViewReconstructor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 1. 图像特征提取
        self.backbone = self._build_backbone()

        # 2. 特征体积生成
        self.volume_encoder = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        # 3. 隐式场MLP
        self.implicit_field = ImplicitField(
            input_dim=256,
            hidden_dim=256,
            output_dim=1,  # 密度
            num_layers=4
        )

        # 4. 渲染器
        self.volume_renderer = VolumeRenderer(
            num_samples=64,
            ray_step_size=0.01
        )

    def _build_backbone(self):
        """构建ResNet50特征提取器"""
        backbone = resnet50(pretrained=True)
        return nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            pred_mesh: 预测的3D网格
            feature_volume: 特征体积
        """
        # 提取2D特征
        features = self.backbone(x)  # [B, 2048, H/32, W/32]

        # 生成特征体积
        feature_volume = self.volume_encoder(features)

        # 使用隐式场生成3D表示
        points, densities = self.implicit_field(feature_volume)

        # 体积渲染生成网格
        pred_mesh = self.volume_renderer(points, densities)

        return pred_mesh, feature_volume
