# models/base_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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
        self.volume_encoder = self._build_feature_encoder()
        # self.volume_encoder = nn.Sequential(
        #     nn.Conv2d(2048, 256, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=False),  # 移除 inplace=True
        #     nn.Conv2d(256, self.feature_dim, 1),
        #     nn.BatchNorm2d(self.feature_dim),
        #     nn.ReLU(inplace=False)  # 移除 inplace=True
        # )
        self.debug_encoder_type()
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
        # self.volume_renderer = VolumeRenderer(
        #     num_samples=32,
        #     ray_step_size=0.02
        # )

        # 初始化权重
        self._init_weights()

    def debug_encoder_type(self):
        """调试编码器类型"""
        print(f"Volume encoder type: {type(self.volume_encoder)}")
        print(f"Volume encoder structure: {self.volume_encoder}")

        # 测试是否接受张量输入
        try:
            dummy_input = torch.randn(1, 2048, 7, 7)  # 假设的C4尺寸
            output = self.volume_encoder(dummy_input)
            print(f"Successfully processed tensor input. Output shape: {output.shape}")
        except Exception as e:
            print(f"Failed to process tensor input: {e}")

        # 测试是否接受字典输入
        try:
            dummy_dict = {
                '0': torch.randn(1, 256, 28, 28),
                '1': torch.randn(1, 512, 14, 14),
                '2': torch.randn(1, 1024, 7, 7),
                '3': torch.randn(1, 2048, 7, 7)
            }
            output = self.volume_encoder(dummy_dict)
            print(f"Successfully processed dict input. Output keys: {output.keys()}")
        except Exception as e:
            print(f"Failed to process dict input: {e}")

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

    def _build_feature_encoder(self):
        """构建特征编码器，尝试使用预训练模型"""
        try:
            # 尝试使用torchvision的FPN
            import torchvision.ops as ops

            # 创建FPN
            in_channels_list = [256, 512, 1024, 2048]  # ResNet的通道数
            return ops.FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels= 256 # self.feature_dim
            )
        except (ImportError, AttributeError):
            # 回退到自定义编码器
            print("无法导入FPN，使用自定义特征编码器")
            return nn.Sequential(
                nn.Conv2d(2048, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=False),
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False),
                nn.Conv2d(256, self.feature_dim, 1),
                nn.BatchNorm2d(self.feature_dim),
                nn.ReLU(inplace=False)
            )

    def forward(self, x):
        # 提取多尺度特征
        features_list = self.extract_features(x)

        # 使用隐式场
        points, densities,_ = self.implicit_field(features_list)
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

    def extract_features_(self, images):
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

    def extract_features(self, images):
        """提取多尺度特征 - 支持FPN和Sequential编码器"""
        # 使用骨干网络提取特征
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 提取多尺度特征
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # 检测编码器类型并相应处理
        if isinstance(self.volume_encoder, torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork):
            try:
                # 为FPN准备字典输入
                features_dict = {
                    '0': c1,
                    '1': c2,
                    '2': c3,
                    '3': c4
                }

                # 使用FPN处理特征
                encoded_features = self.volume_encoder(features_dict)

                # 返回编码后的特征
                return [encoded_features['1'], encoded_features['2'], encoded_features['3']]

            except Exception as e:
                print(f"FPN特征编码失败: {e}，回退到原始特征")
                return [c2, c3, c4]
        else:
            try:
                # 假设是Sequential或类似结构
                encoded_c4 = self.volume_encoder(c4)
                return [c2, c3, encoded_c4]
            except Exception as e:
                print(f"特征编码失败: {e}，回退到原始特征")
                return [c2, c3, c4]
