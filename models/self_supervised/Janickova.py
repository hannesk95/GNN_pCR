import torch
import monai
import torch.nn as nn
from torchsummary import summary

class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, pool_param):
        super(Projector, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(pool_param)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.projector(x)

class ResNet18EncoderJanickova(nn.Module):
    def __init__(self):
        super().__init__()

        # Image encoder
        model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=3, num_classes=2)
        # encoder_layers = list(model.children())[:-1]
        # encoder_layers.append(torch.nn.Flatten(start_dim=1, end_dim=-1))                
        # self.encoder = torch.nn.Sequential(*encoder_layers)  

        filters = [64, 128, 256, 512]
   
        self.block_1 = nn.Sequential(*list(model.children())[0:5])  # 64 filters   
        self.block_2 = nn.Sequential(*list(model.children())[5])    # 128 filters
        self.block_3 = nn.Sequential(*list(model.children())[6])    # 256 filters  
        self.block_4 = nn.Sequential(*list(model.children())[7])    # 512 filters

        self.projector_1 = Projector(in_dim=filters[0], out_dim=filters[0], pool_param=(1,1,1))
        self.projector_2 = Projector(in_dim=filters[1], out_dim=filters[1], pool_param=(1,1,1))
        self.projector_3 = Projector(in_dim=filters[2], out_dim=filters[2], pool_param=(1,1,1))
        self.projector_4 = Projector(in_dim=filters[3], out_dim=filters[3], pool_param=(1,1,1))
        self.attention_blocks = nn.ModuleList([self.att_layer([filters[i], filters[i], filters[i]]) for i in range(4)])


    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv3d(channel[0], channel[1], kernel_size=1, padding=0),
            nn.BatchNorm3d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel[1], channel[2], kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        return att_block
    
    def mask_feature_maps(self, encoder_features, i):
        
        attn_mask = self.attention_blocks[i](encoder_features)
        refined_feature = attn_mask * encoder_features  # Apply attention
        return refined_feature

    def forward(self, images):
        """
        images:   (B, 3, D, H, W)
        """

        B, T, C, D, H, W = images.shape
   
        images = images.view(B * T, C, D, H, W)
        # features = self.encoder(images)  # (B*T, 512)
        # features = features.view(B, T, -1)  # (B, T, 512)

        out1 = self.block_1(images)  # (B, 64, D/2, H/2, W/2)
        out2 = self.block_2(out1)    # (B, 128, D/4, H/4, W/4)
        out3 = self.block_3(out2)    # (B, 256, D/8, H/8, W/8)
        out4 = self.block_4(out3)    # (B, 512, D/16, H/16, W/16)

        # apply attention masks
        out1 = self.mask_feature_maps(out1, 0)
        out2 = self.mask_feature_maps(out2, 1)
        out3 = self.mask_feature_maps(out3, 2)
        out4 = self.mask_feature_maps(out4, 3)
        
        # get the multi-scale embeddings
        out1 = self.projector_1(out1)
        out2 = self.projector_2(out2)
        out3 = self.projector_3(out3)
        out4 = self.projector_4(out4)
        
        latent = torch.cat([out1, out2, out3, out4], dim=-1)
        latent = latent.view(B, T, -1)

        return latent

class ResNet18EncoderJanickova_new(nn.Module):
    def __init__(self):
        super().__init__()

        # Image encoder
        model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=3, num_classes=2)
        encoder_layers = list(model.children())[:-1]
        encoder_layers.append(torch.nn.Flatten(start_dim=1, end_dim=-1))                
        self.encoder = nn.Sequential(*encoder_layers)  

        # Projector
        self.projector = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128)
        )

    def forward(self, images):
        """images:   (B, T, C, D, H, W)
        """
        B, T, C, D, H, W = images.shape
   
        images = images.reshape(B * T, C, D, H, W)
        features = self.encoder(images)             # (B * T, 512)
        latent = self.projector(features)           # (B * T, 128)

        latent =  latent.reshape(B, T, -1)          # (B, T, 128)

        return latent

if __name__ == "__main__":
    
    BATCH_SIZE = 2
    TIMEPOINTS = 4
    CHANNELS = 3
    DEPTH = 32
    HEIGHT = 32
    WIDTH = 32
    
    images = torch.randn(BATCH_SIZE, TIMEPOINTS, CHANNELS, DEPTH, HEIGHT, WIDTH)
    images = images.cuda() 

    model = ResNet18EncoderJanickova().cuda()
    # summary(model, input_size=(TIMEPOINTS, CHANNELS, DEPTH, HEIGHT, WIDTH))
    out = model(images)
    print(out.shape)  

    model = ResNet18EncoderJanickova_new().cuda()
    # summary(model, input_size=(TIMEPOINTS, CHANNELS, DEPTH, HEIGHT, WIDTH))
    out = model(images)
    print(out.shape)
