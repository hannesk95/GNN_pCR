import torch
import monai
import numpy as np
import torch.nn as nn
from torchsummary import summary
from scipy.special import factorial

class ResNet18EncoderKaczmarek(nn.Module):
    def __init__(self, timepoints: int = None, training: bool = True):
        super().__init__()

        assert timepoints is not None, "Please provide the number of timepoints."
        self.timepoints = timepoints
        self.training = training
        # Image encoder
        model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=3, num_classes=2)
        encoder_layers = list(model.children())[:-1]
        encoder_layers.append(torch.nn.Flatten(start_dim=1, end_dim=-1))                
        self.encoder = nn.Sequential(*encoder_layers)  

        # Classifier
        self.cls = nn.Sequential(
            nn.Linear(in_features=timepoints*512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=int(factorial(timepoints)))
        )

        # Projector
        self.projector = nn.Sequential(
            nn.Linear(in_features=timepoints*512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128)
        )

    def forward(self, images):
        """
        images:   (B, T, C, D, H, W)
        """

        B, T, C, D, H, W = images.shape
   
        images = images.view(B * T, C, D, H, W)
        features = self.encoder(images)  # (B*T, 512)
        features = features.view(B, T, 512)  # (B, T, 512)

        features_original = features[:, :T//2, :]  # (B, T/2, 512)
        features_transformed = features[:, T//2:, :] 

        if self.training:
            ##################################
            # random shuffle along T dimension
            ##################################

            # Create independent permutations for each batch element
            idx = torch.stack([torch.randperm(T//2) for _ in range(B)])
            labels = ["".join(map(str, row.tolist())) for row in idx]
            labels = np.array(labels).reshape(-1, 1)

            # Expand idx to match feature dimension
            idx = idx.unsqueeze(-1).expand(-1, -1, 512)
            idx = idx.to(features_original.device)

            # Gather along T dimension
            features_original = torch.gather(features_original, 1, idx)
            features_transformed = torch.gather(features_transformed, 1, idx)

            # features = torch.cat([features_original, features_transformed], dim=1)  # (B, T, 512)

            # features = features.view(B, T * 512)  # (B, T*512)

            logits = self.cls(features_transformed.view(B, -1))  # (B*T, T!)
            # logits = logits.view(B, -1)  # (B, T, T!)
            
            latent_original = self.projector(features_original.view(B, -1))  # (B*T, 128)
            latent_transformed = self.projector(features_transformed.view(B, -1))  # (B*T, 128)
            # latent = latent.view(B, T, -1)  # (B, T, 128)        

        else:
            logits = features_original
            latent_original = features_original
            latent_transformed = features_transformed
            labels = features_original

        return logits, latent_original, latent_transformed, labels

if __name__ == "__main__":
    
    BATCH_SIZE = 8
    TIMEPOINTS = 3
    images = torch.randn(BATCH_SIZE, TIMEPOINTS, 3, 32, 32, 32)
    images = images.cuda() 

    model = ResNet18EncoderKaczmarek(timepoints=TIMEPOINTS).cuda()
    summary(model, input_size=(TIMEPOINTS, 3, 32, 32, 32))
    logits, latent = model(images)

    print(logits.shape)
    print(latent.shape)  