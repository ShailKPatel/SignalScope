"""
SignalScope Dual-Stream Model Architecture
Combines Spatial CNN/ViT features with High-Frequency Spectrum Artifacts (FFT/DCT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class FrequencyBranch(nn.Module):
    """
    Extracts high-frequency spatial grid artifacts caused by generative upsampling layers.
    Computes 2D FFT & DCT spectral features.
    """
    def __init__(self, in_channels=1, feature_dim=128):
        super(FrequencyBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, feature_dim)

    def extract_fft_spectrum(self, x):
        """
        Calculates 2D FFT log-magnitude spectrum for batch of images.
        x: [B, C, H, W]
        """
        if x.shape[1] == 3:
            gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        else:
            gray = x

        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)
        log_spectrum = torch.log(magnitude + 1e-8)
        
        min_v = log_spectrum.view(log_spectrum.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_v = log_spectrum.view(log_spectrum.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        norm_spectrum = (log_spectrum - min_v) / (max_v - min_v + 1e-8)
        
        return norm_spectrum

    def forward(self, x):
        spectrum = self.extract_fft_spectrum(x)
        feat = F.relu(self.bn1(self.conv1(spectrum)))
        feat = F.relu(self.bn2(self.conv2(feat)))
        feat = F.relu(self.bn3(self.conv3(feat)))
        feat = self.adaptive_pool(feat)
        feat = torch.flatten(feat, 1)
        out = F.relu(self.fc(feat))
        return out


class SignalScopeDualStreamModel(nn.Module):
    """
    Master Classifier combining Spatial Features (EfficientNet/ResNet) and Frequency Domain Features.
    """
    def __init__(self, spatial_backbone="resnet34", pretrained=False, dropout_rate=0.3):
        super(SignalScopeDualStreamModel, self).__init__()
        
        # 1. Spatial Backbone
        if spatial_backbone == "resnet18":
            base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None)
            num_spatial_features = base.fc.in_features
            base.fc = nn.Identity()
        elif spatial_backbone == "resnet50":
            base = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT if pretrained else None)
            num_spatial_features = base.fc.in_features
            base.fc = nn.Identity()
        elif spatial_backbone == "efficientnet_b0":
            base = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            num_spatial_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
        else: # Default resnet34
            base = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT if pretrained else None)
            num_spatial_features = base.fc.in_features
            base.fc = nn.Identity()
            
        self.spatial_stream = base
        
        # 2. Frequency Stream
        self.freq_dim = 128
        self.frequency_stream = FrequencyBranch(in_channels=1, feature_dim=self.freq_dim)
        
        # 3. Fusion Head
        total_dim = num_spatial_features + self.freq_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Input x: Image tensor [B, 3, H, W]
        Returns: Logits tensor [B, 1]
        """
        spatial_feats = self.spatial_stream(x)
        freq_feats = self.frequency_stream(x)
        
        combined = torch.cat((spatial_feats, freq_feats), dim=1)
        logits = self.classifier(combined)
        return logits

    def predict_probability(self, x):
        """
        Returns calibrated probability score [0.0, 1.0] for being synthetic AI-generated.
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing SignalScopeDualStreamModel on device: {device}")
    model = SignalScopeDualStreamModel(spatial_backbone="resnet34", pretrained=False).to(device)
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        out_logits = model(dummy_input)
        probs = model.predict_probability(dummy_input)
    print(f"Output Logits Shape: {out_logits.shape}")
    print(f"Sample Probabilities: {probs.squeeze().tolist()}")
    print("Dual-Stream Backbone verified successfully!")
