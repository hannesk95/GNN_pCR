import torch
from transformers import AutoImageProcessor, AutoModel

class DINOv3(torch.nn.Module):
    def __init__(self, model_name='facebook/dinov3-vits16-pretrain-lvd1689m'):
        super(DINOv3, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to('cuda')

    def forward(self, images):
        # Preprocess the images
        inputs = self.image_processor(images=images, return_tensors="pt").to('cuda')
        # inputs = inputs.cuda()
        
        # Forward pass through the model
        outputs = self.model(**inputs)

        pooled_output = outputs.pooler_output
        
        return pooled_output