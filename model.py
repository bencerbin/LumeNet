import torch.nn as nn

class ColorizationNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),      # [1, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),     # [64, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                  # [64, 64, 32, 32]

            nn.Conv2d(64, 128, 3, padding=1),    # [64, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),   # [128, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                  # [128, 128, 16, 16]

            nn.Conv2d(128, 256, 3, padding=1),   # [128, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),   # [256, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                   # [256, 256, 8, 8]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # [256, 128, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),   # [128, 64, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),    # [64, 32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1), # [32, 3, 64, 64]
            nn.Sigmoid()  # For output in [0, 1] RGB
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
