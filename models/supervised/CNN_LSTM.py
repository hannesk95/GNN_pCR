import torch
import monai
import torch.nn as nn
from torchsummary import summary

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # Image encoder
        model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=3, num_classes=2)
        encoder_layers = list(model.children())[:-1]
        encoder_layers.append(torch.nn.Flatten(start_dim=1, end_dim=-1))                
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # LSTM
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        def init_classifier(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

        self.classifier.apply(init_classifier)

    def forward(self, images):
        """
        images:   (B, T, C, D, H, W)
        """

        B, T, C, D, H, W = images.shape
   
        images = images.reshape(B * T, C, D, H, W)
        features = self.encoder(images)             # (B*T, 512)
        features = features.reshape(B, T, -1)       # (B, T, 512)

        out, _ = self.lstm(features)
        out = out[:, -1, :]  # (B, 256)        

        logits = self.classifier(out)

        return logits

if __name__ == "__main__":
    
    BATCH_SIZE = 8
    TIMEPOINTS = 4
    CHANNELS = 3
    images = torch.randn(BATCH_SIZE, TIMEPOINTS, CHANNELS, 64, 64, 64).cuda()

    model = CNNLSTM().cuda()
    summary(model, input_size=(TIMEPOINTS, CHANNELS, 64, 64, 64))
    out = model(images)

    print(out.shape)  