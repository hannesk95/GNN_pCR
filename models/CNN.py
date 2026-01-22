import torch
import monai
import torch.nn as nn
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, num_timepoints = None):
        super().__init__()

        assert num_timepoints is not None, "num_timepoints must be specified for CNN model"

        # Image encoder
        model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=3, num_classes=2)
        encoder_layers = list(model.children())[:-1]
        encoder_layers.append(torch.nn.Flatten(start_dim=1, end_dim=-1))                
        self.encoder = torch.nn.Sequential(*encoder_layers)   

        self.dropout = nn.Dropout(p=0.5)

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * num_timepoints, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        self.encoder.apply(self.init_weights)
        self.classifier.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, images):
        """
        images:   (B, T, 3, D, H, W)
        """

        B, T, C, D, H, W = images.shape
   
        images = images.view(B * T, C, D, H, W)
        features = self.encoder(images)  # (B*T, 512)
        features = features.view(B, T, -1)  # (B, T, 512)
        features = features.view(B, T*features.size(2))  # (B, T*512)

        # apply dropout
        features = self.dropout(features)

        logits = self.classifier(features)

        return logits

if __name__ == "__main__":
    
    BATCH_SIZE = 8
    TIMEPOINTS = 4
    images = torch.randn(BATCH_SIZE, TIMEPOINTS, 3, 32, 32, 32)
    images = images.cuda() 

    model = CNN(num_timepoints=TIMEPOINTS).cuda()
    summary(model, input_size=(TIMEPOINTS, 3, 32, 32, 32))
    out = model(images)

    print(out.shape)  