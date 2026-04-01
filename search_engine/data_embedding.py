import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vit_b_32, ViT_B_32_Weights
from transformers import RobertaModel


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        weights = ViT_B_32_Weights.DEFAULT
        self.model = vit_b_32(weights=weights)
        self.model.heads = nn.Identity()   # 분류 head 제거 -> 768-d feature
        self.fc = nn.Linear(768, embed_dim)

    def forward(self, images):
        x = self.model(images)   # [B, 768]
        x = self.fc(x)           # [B, embed_dim]
        return x


class TextEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.fc = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.fc(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim=512, proj_dim=512):
        super().__init__()
        self.projection = nn.Linear(embed_dim, proj_dim)

    def forward(self, x):
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        logits = image_features @ text_features.T
        logits = logits / self.temperature

        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2


class CLIP(nn.Module):
    def __init__(self, embed_dim=512, proj_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)

        self.image_proj = ProjectionHead(embed_dim, proj_dim)
        self.text_proj = ProjectionHead(embed_dim, proj_dim)

        self.loss_fn = CLIPLoss()

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_proj(self.image_encoder(images))
        text_features = self.text_proj(self.text_encoder(input_ids, attention_mask))

        loss = self.loss_fn(image_features, text_features)
        return loss, image_features, text_features

    @torch.no_grad()
    def encode_image(self, images):
        self.eval()
        image_features = self.image_proj(self.image_encoder(images))
        return image_features

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        self.eval()
        text_features = self.text_proj(self.text_encoder(input_ids, attention_mask))
        return text_features
