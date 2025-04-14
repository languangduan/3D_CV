# models/clip_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torchvision import transforms


class CLIPEncoder(nn.Module):
    def __init__(self, clip_model="ViT-B/32", device=None):
        super().__init__()
        # 加载预训练CLIP模型
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model, self.preprocess = clip.load(clip_model, device=self.device)
        self.clip_model = self.clip_model.float()

        # 冻结CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # CLIP特征维度
        self.clip_dim = 512

        # 投影层：将CLIP特征映射到3D特征空间
        self.projection = nn.Sequential(
            nn.Linear(self.clip_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )

        # 获取CLIP模型的输入尺寸
        if hasattr(self.clip_model.visual, 'input_resolution'):
            self.input_resolution = self.clip_model.visual.input_resolution
        else:
            # 默认为224，大多数CLIP模型使用这个尺寸
            self.input_resolution = 224

        # 创建预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((self.input_resolution, self.input_resolution),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def encode_image(self, image):
        """编码图像获取CLIP特征"""
        # 确保图像尺寸正确
        if image.shape[-1] != self.input_resolution or image.shape[-2] != self.input_resolution:
            # 调整图像尺寸以匹配CLIP模型的预期输入
            image = self.transform(image)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, text):
        """编码文本获取CLIP特征"""
        with torch.no_grad():
            text = clip.tokenize(text).to(next(self.clip_model.parameters()).device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def project_features(self, clip_features):
        """将CLIP特征投影到3D特征空间"""
        return self.projection(clip_features)

    def forward(self, image):
        """前向传播：编码图像并投影到3D特征空间"""
        clip_features = self.encode_image(image)
        return self.project_features(clip_features)
