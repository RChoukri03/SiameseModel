import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SiameseAttentionModule(nn.Module):
    def __init__(self, channel_size):
        super(SiameseAttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channel_size, channel_size // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_size // 8, channel_size, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)

class RotDecNetwork(nn.Module):
    def __init__(self, base_model_name='mobilenet_v2'):
        super(RotDecNetwork, self).__init__()

        # Load the base model (EfficientNet) with pretrained weights
        if base_model_name == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            channel_size = 1280 #self.base_model._fc.in_features
        else:
            raise ValueError("Unsupported base model, please use 'mobilenet_v2'")

        self.attention_module = SiameseAttentionModule(channel_size)


    def forward_once(self, x):
        x = self.base_model.features(x)
        x = self.attention_module(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculer la distance euclidienne
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Calculer la perte contrastive
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive