import torch
import monai
import torch.nn as nn
import math

class LSTMFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size, use_time_distances=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_time_distances = use_time_distances

        # Combine all gates into one big matrix for efficiency
        self.W = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.U = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))

        if self.use_time_distances:
            self.time_W = nn.Parameter(torch.Tensor(1, 4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x, initial_state=None, time_distances=None):
        """
        x: (batch, seq_len, input_size)
        initial_state: tuple(h0, c0) each (batch, hidden_size)
        """

        batch_size, seq_len, _ = x.size()

        if initial_state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = initial_state

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            gates = x_t @ self.W + h @ self.U + self.bias
            if self.use_time_distances and time_distances is not None:
                time_dist_t = time_distances[:, t]  # (batch, 1)
                gates = gates + time_dist_t * self.time_W

            f, i, o, g = gates.chunk(4, dim=1)

            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            g = torch.tanh(g)

            c = f * c + i * g
            h = o * torch.tanh(c)

            outputs.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, (h, c)

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

        # self.classifier.apply(init_classifier)

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