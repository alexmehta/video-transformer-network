"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import torch.nn as nn
from torchvision import models
class VideoModel(nn.Module):
    def __init__(self, num_channels=3):
        super(VideoModel, self).__init__()
        self.r2plus1d = models.video.r3d_18(pretrained=True)
        self.r2plus1d.fc = nn.Sequential(            nn.BatchNorm1d(num_features=self.r2plus1d.fc.in_features),
            nn.Linear(in_features=self.r2plus1d.fc.in_features, out_features=128)
            ,nn.ReLU(),nn.BatchNorm1d(num_features=128))
        for layer in self.r2plus1d.children():
            layer.requires_grad_ = False
        self.r2plus1d.fc.requires_grad_ = True
    def forward(self, x):
        return self.r2plus1d(x)
class TwoStreamAuralVisualModel(nn.Module):
    def __init__(self, num_channels=4, audio_pretrained=False):
        super(TwoStreamAuralVisualModel, self).__init__()
        self.video_model = VideoModel(num_channels=num_channels)
        self.fc = nn.Sequential(nn.Linear(in_features=self.video_model.r2plus1d.fc._modules['1'].out_features,out_features=8+12+2))
        self.modes = ['clip']
    def forward(self, x):
        video_model_features = self.video_model(x)
        out = self.fc(video_model_features)
        return out 