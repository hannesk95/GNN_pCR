from monai.networks.nets import BasicUNet, AutoEncoder
from models.MTANUNet import MTANRecUnet
from models.RecUNet import RecUnet
import torch 


def build_model(
        mtan_masking=True, 
        filters:list = [16,32,64,128,256,32], 
        device: str = 'cuda'):
    if mtan_masking:
        return build_MTANAE(filters, device)
    else:
        return build_RecUNet(filters, device)
    
    
def build_MTANAE(
        filters:list = [16,32,64,128,256,32], 
        device: str ='cuda'): 
    # unet = BasicUNet(
    #     spatial_dims=2, 
    #     in_channels=3, 
    #     out_channels=3, 
    #     features=filters,
    #     no_skip=False).to(device)

    unet = AutoEncoder(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=[16,32,64,128,256],
        strides=(2, 2, 2, 2, 2)).to(device)

    model = MTANRecUnet(
        unet=unet, 
        filters = filters[1:5]).to(device)
    
    return model


def build_RecUNet(
        filters:list = [16,32,64,128,256,32], 
        device: str ='cuda'): 
    unet = BasicUNet(
        spatial_dims=2, 
        in_channels=3, 
        out_channels=3, 
        features=filters,
        no_skip=False).to(device)
    
    model = RecUnet(
        unet=unet, 
        filters = filters[1:5]).to(device)

    return model