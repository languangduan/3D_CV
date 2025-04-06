# models/base_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from models.implicit_field import ImplicitField
from models.renderer import VolumeRenderer


class SingleViewReconstructor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 初始化配置
        self.cfg = cfg
        self.feature_dim = 128

        # 1. 构建并初始化backbone
        self._build_backbone()  # 现在直接设置属性而不是返回Sequential

        # 2. 特征编码器
        # 特征编码器
        self.volume_encoder = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),  # 移除 inplace=True
            nn.Conv2d(256, self.feature_dim, 1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=False)  # 移除 inplace=True
        )

        # 确保所有参数需要梯度
        for param in self.volume_encoder.parameters():
            param.requires_grad = True

        # 3. 隐式场网络
        self.implicit_field = ImplicitField(
            input_dim=self.feature_dim,
            hidden_dim=128,
            output_dim=1,
            num_layers=3
        )
        self.implicit_field.requires_grad_(True)

        # 4. 体积渲染器
        self.volume_renderer = VolumeRenderer(
            num_samples=32,
            ray_step_size=0.02
        )

        # 初始化权重
        self._init_weights()

    def _build_backbone(self):
        """构建并优化backbone，分别保存各个组件"""
        backbone = resnet50(pretrained=True)

        # 保存各个组件为模型的属性
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 冻结早期层
        for layer in [self.conv1, self.bn1, self.layer1]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        # 提取多尺度特征
        features_list = self.extract_features(x)

        # 使用隐式场
        points, densities = self.implicit_field(features_list)
        return points, densities

    def train(self, mode=True):
        """重写训练模式设置，保持某些层冻结"""
        super().train(mode)
        # 确保冻结层始终在评估模式
        if mode:
            self.conv1.eval()
            self.bn1.eval()
            self.layer1.eval()
        return self

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, images):
        """提取多尺度特征"""
        # 使用骨干网络提取特征
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 提取多尺度特征
        c1 = self.layer1(x)  # 低级特征
        c2 = self.layer2(c1)  # 中级特征
        c3 = self.layer3(c2)  # 高级特征
        c4 = self.layer4(c3)  # 最高级特征

        # 返回C3, C4, C5特征用于特征金字塔
        return [c2, c3, c4]  # 对应ResNet的C3, C4, C5

