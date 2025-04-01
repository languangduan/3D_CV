# models/clip_encoder.py
import torch
import torch.nn as nn
import clip


class CLIPEncoder(nn.Module):
    def __init__(self, clip_model="ViT-B/32"):
        super().__init__()
        # 加载预训练CLIP模型
        self.clip_model, _ = clip.load(clip_model, device="cpu")

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

    def encode_image(self, image):
        """编码图像获取CLIP特征"""
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
