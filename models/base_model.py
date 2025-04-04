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
        # 添加梯度检查点
       #  print(f"Input x requires_grad: {x.requires_grad}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)
        # print(f"After layer4 requires_grad: {features.requires_grad}")

        # 确保特征需要梯度
        if not features.requires_grad:
            features.requires_grad_(True)

        feature_volume = self.volume_encoder(features)
        # print(f"Feature volume requires_grad: {feature_volume.requires_grad}")

        points, densities = self.implicit_field(feature_volume)
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

