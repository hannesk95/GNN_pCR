import torch
import monai
import torch.nn as nn
from torchsummary import summary
# from distanceLSTM.distanceLSTM import distLSTM

# import sys
# sys.path.append("/home/johannes/Data/SSD_2.0TB/GNN_pCR/dist_conv_lstm/tumor-cifar")
# import crnn

import sys
sys.path.append("/home/johannes/Data/SSD_2.0TB/GNN_pCR/models")
from tLSTM import LSTMFromScratch

# class distLSTMCell(nn.Module):
#     """
#     Minimal, explicit LSTM cell.
    
#     Input:
#         x_t : (B, input_dim)
#         h_t : (B, hidden_dim)
#         c_t : (B, hidden_dim)
#     """

#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim

#         # One linear layer for all 4 gates
#         self.linear = nn.Linear(
#             input_dim + hidden_dim,
#             4 * hidden_dim,
#             bias=True
#         )

#         self.reset_parameters()

#         self.a = nn.Parameter(torch.Tensor(1))
#         self.b = nn.Parameter(torch.Tensor(1))
#         self.c = nn.Parameter(torch.Tensor(1))
#         nn.init.uniform_(self.a.data,a=0.5,b=0.51)  # default 0.5
#         nn.init.uniform_(self.b.data,a=0.,b=1)
#         nn.init.uniform_(self.c.data,a=0.5,b=0.51)  # default 0.5

#     def reset_parameters(self):
#         # Standard initialization
#         nn.init.xavier_uniform_(self.linear.weight)

#         # Forget gate bias to 1 improves stability
#         with torch.no_grad():
#             self.linear.bias[self.hidden_dim:2*self.hidden_dim].fill_(1.0)

#     def forward(self, x_t, h_prev, c_prev, time_dists=None):
#         """
#         One LSTM step.
#         """       

#         # Concatenate input and hidden state
#         combined = torch.cat([x_t, h_prev], dim=1)

#         # Compute gates
#         gates = self.linear(combined)

#         # Split gates
#         i, f, o, g = torch.chunk(gates, 4, dim=1)

#         i = torch.sigmoid(i)
#         f = torch.sigmoid(f)
#         o = torch.sigmoid(o)
#         g = torch.tanh(g)

#         if time_dists is not None:
#             forcoff = self.a.cuda() * torch.exp( - self.c.cuda() * time_dists[0])
#             incoff = self.a.cuda() * torch.exp( - self.c.cuda() * time_dists[1])

#             # if len(forgetgate.shape) == 3: # for conv 1D
#             #     forcoff = forcoff.unsqueeze(1).unsqueeze(2).expand_as(forgetgate)
#             #     incoff = incoff.unsqueeze(1).unsqueeze(2).expand_as(ingate)
                
#             # if len(forgetgate.shape) == 4: # for 2D images
#             #     forcoff = forcoff.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(forgetgate)
#             #     incoff = incoff.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(ingate)

#             # if len(forgetgate.shape) == 5: # for 3D images
#             #     forcoff = forcoff.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(forgetgate)
#             #     incoff = incoff.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(ingate)
            
#             forcoff = forcoff.unsqueeze(1).expand_as(f)
#             incoff = incoff.unsqueeze(1).expand_as(i)
            
#             f = forcoff * f
#             i = incoff * i


#         # Cell update
#         c_t = f * c_prev + i * g

#         # Hidden update
#         h_t = o * torch.tanh(c_t)

#         return h_t, c_t

# class distLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.cell = distLSTMCell(input_dim, hidden_dim)        

#     def forward(self, x, time_dists=None):
#         """
#         x: (B, T, input_dim)
#         """
#         B, T, _ = x.shape
#         device = x.device

#         h = torch.zeros(B, self.cell.hidden_dim, device=device)
#         c = torch.zeros_like(h)

#         outputs = []

#         if time_dists is None:
#             for t in range(T):
#                 h, c = self.cell(x[:, t, :], h, c)
#                 outputs.append(h)
#         else:
#             for t in range(T):            
#                 if t == 0:
#                     hx, cx = self.cell(x[:, t, :], h, c, [time_dists[:, 0], time_dists[:, 0]])
#                 else:
#                     hx, cx = self.cell(x[:, t, :], h, c, [time_dists[:,t-1], time_dists[:,t]])
#                 outputs.append(hx)
#         return torch.stack(outputs, dim=1)  # (B, T, hidden_dim)

class CNNdistLSTM(nn.Module):
    def __init__(self, use_time_distances=False):
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
   
        images = images.view(B * T, C, D, H, W)
        features = self.encoder(images)  # (B*T, 512)
        features = features.view(B, T, -1)  # (B, T, 512)

        if self.use_time_distances:
            out, _ = self.lstm(x=features, time_distances=time_dists)
        else:
            out, _ = self.lstm(x=features)        

        out = out[:, -1, :]  # (B, 256)

        logits = self.classifier(out)

        return logits

if __name__ == "__main__":
    
    BATCH_SIZE = 8
    TIMEPOINTS = 4
    images = torch.randn(BATCH_SIZE, TIMEPOINTS, 3, 32, 32, 32).cuda()
    time_dists = torch.randn(BATCH_SIZE, TIMEPOINTS, 1).cuda()

    model = CNNdistLSTM(use_time_distances=True).cuda()    
    out = model(images, time_dists) # with time distances
    print(out.shape)  

    model = CNNdistLSTM(use_time_distances=False).cuda()
    out = model(images)  # without time distances
    print(out.shape)