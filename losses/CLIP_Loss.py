# losses/clip_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPLoss(nn.Module):
    def __init__(self, clip_model="ViT-B/32", temperature=0.07):
        super().__init__()
        self.model, self.preprocess = clip.load(clip_model, device="cuda")
        self.temperature = temperature

        # 冻结CLIP参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 密度场到CLIP空间的投影网络
        self.density_projector = nn.Sequential(
            nn.Linear(512, 512),  # 假设密度特征维度为512
            nn.ReLU(),
            nn.Linear(512, self.model.visual.output_dim)
        )

    def forward(self, density_features, text_prompts, images=None):
        """
        计算CLIP对比学习损失

        Args:
            density_features: 密度场特征 [B, C]
            text_prompts: 文本提示列表，长度为B
            images: 原始输入图像 [B, 3, H, W]，可选

        Returns:
            loss: CLIP对比学习损失
        """
        if density_features is None or text_prompts is None:
            return torch.tensor(0.0, device="cuda")

        # 将密度特征投影到CLIP空间
        density_features = self.density_projector(density_features)
        density_features = density_features / density_features.norm(dim=1, keepdim=True)

        # 编码文本
        text_tokens = clip.tokenize(text_prompts).to(density_features.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 计算相似度
        logits = (density_features @ text_features.T) / self.temperature

        # 对比损失（每个密度特征与其对应的文本匹配）
        labels = torch.arange(len(density_features), device=density_features.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        # 如果提供了原始图像，可以添加三元组损失
        if images is not None:
            # 确保图像尺寸正确
            if images.shape[2:] != (224, 224):
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            # 提取图像特征
            with torch.no_grad():
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # 计算密度特征与图像特征的一致性损失
            consistency_loss = 1.0 - F.cosine_similarity(density_features, image_features).mean()

            # 添加到总损失
            loss = loss + 0.5 * consistency_loss

        return loss
