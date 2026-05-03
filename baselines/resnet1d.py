"""1-D ResNet-18 adapted for (B, 2, 128) AMC signals."""
import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.downsample = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.downsample(x))


def _make_layer(in_ch, out_ch, blocks, stride):
    layers = [BasicBlock1D(in_ch, out_ch, stride)]
    for _ in range(1, blocks):
        layers.append(BasicBlock1D(out_ch, out_ch))
    return nn.Sequential(*layers)


class ResNet1D18(nn.Module):
    """
    ResNet-18 adapted for 1-D signals.
    Input : (B, 2, 128)  — 2-channel I/Q, 128 samples
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes=11):
        super().__init__()
        # Stem: no stride / no maxpool to preserve the short 128-sample sequence
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _make_layer(64,  64,  2, stride=1)
        self.layer2 = _make_layer(64,  128, 2, stride=2)   # 128 → 64
        self.layer3 = _make_layer(128, 256, 2, stride=2)   # 64  → 32
        self.layer4 = _make_layer(256, 512, 2, stride=2)   # 32  → 16
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
