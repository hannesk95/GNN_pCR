import torch
import monai
import torch.nn as nn
from torchsummary import summary
from distance_LSTM import LSTMFromScratch


class CNNdistLSTM(nn.Module):
    def __init__(self, use_time_distances=True):
        super().__init__()

        self.use_time_distances = use_time_distances

        # Image encoder
        model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=3, num_classes=2)
        encoder_layers = list(model.children())[:-1]
        encoder_layers.append(torch.nn.Flatten(start_dim=1, end_dim=-1))                
        self.encoder = torch.nn.Sequential(*encoder_layers)        

        self.lstm = LSTMFromScratch(input_size=512, hidden_size=256, use_time_distances=use_time_distances)

        # Fully connected classifier
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

    def forward(self, images, time_dists=None):
        """
        images:   (B, T, 3, D, H, W)
        """

        if self.use_time_distances and time_dists is None:
            raise ValueError("Time distances must be provided when use_time_distances is True")

        B, T, C, D, H, W = images.shape
   
        images = images.reshape(B * T, C, D, H, W)
        features = self.encoder(images)             # (B*T, 512)
        features = features.reshape(B, T, -1)       # (B, T, 512)

        if self.use_time_distances:
            out, _ = self.lstm(x=features, time_distances=time_dists)
        else:
            raise NotImplementedError("Currently, the model is only implemented to use time distances. Please set use_time_distances=True when initializing the model.")
            # out, _ = self.lstm(x=features)        

        out = out[:, -1, :]  # (B, 256)

        logits = self.classifier(out)

        return logits

if __name__ == "__main__":
    
    BATCH_SIZE = 8
    TIMEPOINTS = 4
    CHANNELS = 3

    images = torch.randn(BATCH_SIZE, TIMEPOINTS, CHANNELS, 64, 64, 64).cuda()
    time_dists = torch.randn(BATCH_SIZE, TIMEPOINTS, 1).cuda()

    model = CNNdistLSTM(use_time_distances=True).cuda()    
    out = model(images, time_dists) # with time distances
    print(out.shape)  

    # model = CNNdistLSTM(use_time_distances=False).cuda()
    # out = model(images)  # without time distances
    # print(out.shape)