import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Vascular U-Net (Simplified)
# ==========================================
class LightweightUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Simplified encoding/decoding for edge constraints
        self.enc1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Conv2d(16, 16, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.final = nn.Conv2d(16, out_channels, 1)
        
    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(self.pool1(e1)))
        
        d1 = F.relu(self.dec1(self.up1(e2)))
        d2 = torch.sigmoid(self.final(self.up2(d1)))
        
        # Ensure mask matches input dimensions exactly if there are pooling rounding issues
        d2 = F.interpolate(d2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return d2

# ==========================================
# 2. Multi-Task Backbone (MobileNet style)
# ==========================================
class MobileNetMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        # Highly compressed for demo purposes, representing MobileNet feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1), # 4 channels: RGB (3) + Vessel Mask (1)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Branch A: DR Grading (APTOS/Messidor)
        self.grading_head = nn.Linear(64, 5) # 5 DR grades
        
        # Branch B: Lesion Attention Vector (IDRiD grounding)
        self.lesion_head = nn.Linear(64, 4) # 4 Lesion types (MA, HE, EX, SE)

    def forward(self, x, vessel_mask):
        # Concatenate RGB image with Vessel Mask to force attention away from background
        combined = torch.cat([x, vessel_mask], dim=1)
        
        feats = self.features(combined)
        feats = feats.view(feats.size(0), -1)
        
        dr_grade = self.grading_head(feats)
        lesions = self.lesion_head(feats)
        
        return dr_grade, lesions

# ==========================================
# 3. The Full Classic Pipeline
# ==========================================
class ClassicEdgePipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.vascular_unet = LightweightUNet()
        self.multitask_cnn = MobileNetMultiTask()
        
    def forward(self, x):
        # 1. Extract vascular mask (DRIVE task)
        vessel_mask = self.vascular_unet(x)
        
        # 2. Predict DR Grade and Lesions using RGB + Mask (APTOS + IDRiD tasks)
        dr_grade, lesions = self.multitask_cnn(x, vessel_mask)
        
        return dr_grade, lesions, vessel_mask

if __name__ == "__main__":
    model = ClassicEdgePipeline()
    dummy_input = torch.randn(1, 3, 224, 224)
    grade, lesion, mask = model(dummy_input)
    print("=== Classic Edge Architecture ===")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"DR Grade:     {grade.shape}")
    print(f"Lesions:      {lesion.shape}")
    print(f"Vessel Mask:  {mask.shape}")
