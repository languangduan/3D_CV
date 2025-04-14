# losses/clip_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPLoss_(nn.Module):
    def __init__(self, clip_model="ViT-B/32", temperature=0.07):
        super().__init__()
        self.model, self.preprocess = clip.load(clip_model, device="cuda")
        self.model = self.model.float()
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


class CLIPLoss(nn.Module):
    def __init__(self, clip_model="ViT-B/32", temperature=0.07, device='cuda'):
        super().__init__()
        # print(f"Initializing CLIP Loss with model: {clip_model}")
        try:
            self.model, self.preprocess = clip.load(clip_model, device=device)
            # print(f"CLIP model loaded successfully. Output dim: {self.model.visual.output_dim}")
        except Exception as e:
            # print(f"Error loading CLIP model: {e}")
            raise e
        self.model = self.model.float()

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
        # print(f"Density projector initialized: {self.density_projector}")

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
        # 详细记录输入
        print(f"\n=== CLIP Loss Debug ===")
        print(
            f"Density features: {type(density_features)}, shape: {density_features.shape if density_features is not None else 'None'}")
        print(f"Text prompts: {type(text_prompts)}, content: {text_prompts}")
        print(f"Images: {type(images)}, shape: {images.shape if images is not None else 'None'}")

        if density_features is None or text_prompts is None:
            print("CLIP Loss: Missing inputs - returning zero loss")
            return torch.tensor(0.0, device="cuda")

        if isinstance(text_prompts, list) and len(text_prompts) == 0:
            print("CLIP Loss: Empty text prompts list - returning zero loss")
            return torch.tensor(0.0, device="cuda")

        try:
            # 将密度特征投影到CLIP空间
            density_features = self.density_projector(density_features)
            print(f"Projected density features shape: {density_features.shape}")

            density_features = density_features / density_features.norm(dim=1, keepdim=True)
            print(f"Normalized density features shape: {density_features.shape}")

            # 编码文本
            print(f"Tokenizing text prompts: {text_prompts}")
            text_tokens = clip.tokenize(text_prompts).to(density_features.device)
            print(f"Text tokens shape: {text_tokens.shape}")

            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                print(f"Text features shape: {text_features.shape}")

                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                print(f"Normalized text features shape: {text_features.shape}")

            # 计算相似度
            logits = (density_features @ text_features.T) / self.temperature
            print(f"Logits shape: {logits.shape}, values: {logits[:2, :2]}")

            # 对比损失（每个密度特征与其对应的文本匹配）
            labels = torch.arange(len(density_features), device=density_features.device)
            print(f"Labels: {labels}")

            loss_1 = F.cross_entropy(logits, labels)
            loss_2 = F.cross_entropy(logits.T, labels)
            print(f"Loss components: {loss_1.item()}, {loss_2.item()}")

            loss = (loss_1 + loss_2) / 2
            print(f"Combined loss: {loss.item()}")

            # 如果提供了原始图像，可以添加三元组损失
            if images is not None:
                # 确保图像尺寸正确
                if images.shape[2:] != (224, 224):
                    print(f"Resizing images from {images.shape[2:]} to (224, 224)")
                    images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

                # 提取图像特征
                with torch.no_grad():
                    image_features = self.model.encode_image(images)
                    print(f"Image features shape: {image_features.shape}")

                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    print(f"Normalized image features shape: {image_features.shape}")

                # 计算密度特征与图像特征的一致性损失
                consistency_loss = 1.0 - F.cosine_similarity(density_features, image_features).mean()
                print(f"Consistency loss: {consistency_loss.item()}")

                # 添加到总损失
                loss = loss + 0.5 * consistency_loss
                print(f"Final loss with consistency: {loss.item()}")

            print(f"=== CLIP Loss: {loss.item()} ===\n")
            return loss

        except Exception as e:
            print(f"Error in CLIP loss calculation: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device="cuda")
