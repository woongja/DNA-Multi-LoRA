import torch
import torch.nn as nn


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1,1)):
        super().__init__()
        self.downsample = None
        if stride != (1,1) or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.act(out)

# Spectrogram Branch (CNN+LSTM)
class SpectrogramBranchRes(nn.Module):
    def __init__(self, in_channels=1, output_dim=1024, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Stage 1 (keep size)
        self.stage1 = nn.Sequential(
            ResBlock2D(64, 64, stride=(1,1)),
            ResBlock2D(64, 64, stride=(1,1)),
        )
        # Downsample (2x) then Stage 2
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage2 = nn.Sequential(
            ResBlock2D(64, 128, stride=(1,1)),
            ResBlock2D(128, 128, stride=(1,1)),
        )
        # Downsample (2x) then Stage 3
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage3 = nn.Sequential(
            ResBlock2D(128, 256, stride=(1,1)),
            ResBlock2D(256, 256, stride=(1,1)),
        )
        # 마지막에 한 번 더 2x 다운샘플로 T/F를 줄여 LSTM 입력 길이 안정화
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # LSTM: CNN 출력 채널(=256)을 feature로, 시간축은 conv 후의 T'
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):          # x: [B, 1, F, T]
        x = self.stem(x)          # [B, 64, F, T]
        x = self.stage1(x)        # [B, 64, F, T]
        x = self.down1(x)         # [B, 64, F/2, T/2]
        x = self.stage2(x)        # [B, 128, F/2, T/2]
        x = self.down2(x)         # [B, 128, F/4, T/4]
        x = self.stage3(x)        # [B, 256, F/4, T/4]
        x = self.down3(x)         # [B, 256, F/8, T/8]

        # 주파수 평균 풀링으로 [B, 256, T'] 추출
        x = x.mean(dim=2)         # [B, 256, T']
        x = x.permute(0, 2, 1)    # [B, T', 256]

        lstm_out, _ = self.lstm(x)      # [B, T', 2*hidden]
        x = lstm_out.mean(dim=1)        # [B, 2*hidden] (attention pooling으로 교체 가능)
        x = self.fc(x)                  # [B, output_dim]
        return x

# MFCC Branch (CNN+LSTM)
class MFCCBranchRes(nn.Module):
    def __init__(self, in_channels=1, output_dim=1024, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ResBlock2D(64, 64, stride=(1,1)),
            ResBlock2D(64, 64, stride=(1,1)),
        )
        # MFCC는 주파수 차원이 작을 수 있으니 첫 다운샘플은 시간 위주로 (1,2)
        self.down1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.stage2 = nn.Sequential(
            ResBlock2D(64, 128, stride=(1,1)),
            ResBlock2D(128, 128, stride=(1,1)),
        )
        self.down2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.stage3 = nn.Sequential(
            ResBlock2D(128, 256, stride=(1,1)),
            ResBlock2D(256, 256, stride=(1,1)),
        )
        self.down3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.lstm = nn.LSTM(
            input_size=256, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):          # x: [B, 1, F, T]
        x = self.stem(x)          # [B, 64, F, T]
        x = self.stage1(x)        # [B, 64, F, T]
        x = self.down1(x)         # [B, 64, F/2, T/2]
        x = self.stage2(x)        # [B, 128, F/2, T/2]
        x = self.down2(x)         # [B, 128, F/4, T/4]
        x = self.stage3(x)        # [B, 256, F/4, T/4]
        x = self.down3(x)         # [B, 256, F/8, T/8]

        x = x.mean(dim=2)         # [B, 256, T']
        x = x.permute(0, 2, 1)    # [B, T', 256]
        lstm_out, _ = self.lstm(x)
        x = lstm_out.mean(dim=1)  # [B, 2*hidden]
        x = self.fc(x)
        return x


# F0 Branch (동일하게 유지)
class F0Branch(nn.Module):
    def __init__(self, input_len, output_dim=1024, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden*2, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [batch, 1, f0_len]
        x = self.conv(x)           # [batch, 64, T]
        x = x.permute(0, 2, 1)     # [batch, T, 64]
        lstm_out, _ = self.lstm(x) # [batch, T, lstm_hidden*2]
        x = lstm_out.mean(dim=1)   # [batch, lstm_hidden*2] (global average pooling)
        x = self.fc(x)             # [batch, output_dim]
        return x

# FusionNet (branch 교체)
class FusionNet(nn.Module):
    def __init__(self, num_classes, branch_output_dim=1024, spec_shape=(1,1025,126), mfcc_shape=(1,13,401), f0_len=126):
        super().__init__()
        self.spec_branch = SpectrogramBranchRes(in_channels=spec_shape[0], output_dim=branch_output_dim)
        self.mfcc_branch = MFCCBranchRes(in_channels=mfcc_shape[0], output_dim=branch_output_dim)
        self.f0_branch = F0Branch(f0_len, output_dim=branch_output_dim)  # CNN+LSTM

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(branch_output_dim * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, spec, mfcc, f0):
        spec_feat = self.spec_branch(spec)
        mfcc_feat = self.mfcc_branch(mfcc)
        f0_feat = self.f0_branch(f0)
        fused = torch.cat([spec_feat, mfcc_feat, f0_feat], dim=1)
        out = self.classifier(fused)
        return out
