import torch

class CNNGRU(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Image encoder
        model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=3, num_classes=2)
        encoder_layers = list(model.children())[:-1]
        encoder_layers.append(torch.nn.Flatten(start_dim=1, end_dim=-1))                
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # GRU over time
        self.gru = torch.nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Fully connected classifier
        self.classifier = torch.nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, images):
        """
        images:   (B, T, 3, D, H, W)
        """

        B, T, C, D, H, W = images.shape
   
        images = images.view(B * T, C, D, H, W)
        features = self.encoder(images)  # (B*T, 512)
        features = features.view(B, T, -1)  # (B, T, 512)

        # apply dropout
        features = self.dropout(features)

        out, _ = self.gru(features)
        out = out[:, -1, :]  # (B, 256)        

        logits = self.classifier(out)

        return logits