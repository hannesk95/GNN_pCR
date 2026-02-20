import torch
import monai
import torch.nn as nn
from torchsummary import summary

class CNN_distance(nn.Module):
    def __init__(self, num_timepoints = None):
        super().__init__()

        assert num_timepoints is not None, "num_timepoints must be specified for CNN model"

        # Image encoder
        model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=3, num_classes=2)
        encoder_layers = list(model.children())[:-1]
        encoder_layers.append(torch.nn.Flatten(start_dim=1, end_dim=-1))                
        self.encoder = torch.nn.Sequential(*encoder_layers)   

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear((512 + 1) * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, images, distances=None):
        """
        images:   (B, T, C, D, H, W)
        """

        assert distances is not None, "distances must be provided for CNN_distance model"

        B, T, C, D, H, W = images.shape
   
        images = images.reshape(B * T, C, D, H, W)
        features = self.encoder(images)             # (B*T, 512)
        features = features.reshape(B, T, -1)       # (B, T, 512)
        features = features.reshape(B, -1)          # (B, -1)

        features = self.dropout(features)

        features = features.reshape(B, T, -1)       # (B, T, 512)
        
        # append distances to features
        features = torch.cat((features, distances), dim=-1)  # (B, T, 513)
        features = features.reshape(B, -1)          # (B, T*513)

        # if timepoints less than 4, pad with zeros (only for early response prediction)
        if T < 4:
            padding = torch.zeros(B, (4 - T) * (512 + 1)).cuda()
            features = torch.cat((features, padding), dim=1)

        logits = self.classifier(features)

        return logits

if __name__ == "__main__":
    
    BATCH_SIZE = 8
    TIMEPOINTS = 4
    CHANNELS = 3

    time_distances = torch.randn(BATCH_SIZE, TIMEPOINTS, 1).cuda()
    
    images = torch.randn(BATCH_SIZE, TIMEPOINTS, CHANNELS, 64, 64, 64).cuda()

    model = CNN_distance(num_timepoints=TIMEPOINTS).cuda()
    summary(model, input_size=(TIMEPOINTS, CHANNELS, 64, 64, 64))
    out = model(images, time_distances)

    print(out.shape)  