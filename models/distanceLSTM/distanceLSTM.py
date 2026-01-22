import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/johannes/Data/SSD_2.0TB/GNN_pCR/models/distanceLSTM/')
from module import LSTMdistCell

# class ConvDistLSTM(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, num_classes, time_len):
#         super(ConvDistLSTM, self).__init__()
#         #self.in_channels = in_channels
#         #self.out_channels = out_channels
#         #self.lstmcell = crnn.Conv2dLSTMCell(in_channels, out_channels, kernel_size)
#         # self.dislstmcell = crnn.LSTMdistCell('infor_exp', in_channels, out_channels, kernel_size, convndim = 2)
#         self.dislstmcell = LSTMdistCell('infor_exp', in_channels, out_channels, kernel_size, convndim = 2)
#         #self.lstmcell = crnn.Conv2dLSTMCell(in_channels, out_channels, kernel_size)
#         self.time_len = time_len
#         self.conv1 = nn.Conv2d(out_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # # 2D convolutional layer
#         self.conv2_drop = nn.Dropout2d()           # Dropout layer
#         self.fc1 = nn.Linear(500, 40)             # fully connected layer
#         self.fc2 = nn.Linear(40, num_classes)  
        
#     def forward(self, x, time_dis):
#         #print ('In network', x.shape)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         #time_dis = torch.tensor(cfig['time_dis'])
#         #print (time_dis)
#         #print ('---------------', time_dis.shape)
#         for i in range(self.time_len):
            
#             if i == 0:
#                 hx, cx = self.dislstmcell(x[i], [time_dis[:,0], time_dis[:, 0]])
#             else:
#                 hx, cx = self.dislstmcell(x[i], [time_dis[:,i-1], time_dis[:,i]], (hx, cx))  
                
#         x = F.relu(F.max_pool2d(self.conv1(hx), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         #print ('In network', x.shape)
#         x = x.view(-1, 500)
#         #print ('In network', x.shape)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1) 
    

# if __name__ == "__main__":
#     BATCH_SIZE = 2
#     TIMEPOINTS = 6
#     IN_CHANNELS = 3
#     OUT_CHANNELS = 16
#     KERNEL_SIZE = 3
#     NUM_CLASSES = 2
#     H, W = 28, 28

#     images = torch.randn(TIMEPOINTS, BATCH_SIZE, IN_CHANNELS, H, W).cuda()
#     time_dis = torch.randn(BATCH_SIZE, TIMEPOINTS).cuda()

#     model = ConvDistLSTM(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, NUM_CLASSES, TIMEPOINTS).cuda()
#     output = model(images, time_dis)
#     print(output.shape)

class distLSTMCell(nn.Module):
    """
    Minimal, explicit LSTM cell.
    
    Input:
        x_t : (B, input_dim)
        h_t : (B, hidden_dim)
        c_t : (B, hidden_dim)
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # One linear layer for all 4 gates
        self.linear = nn.Linear(
            input_dim + hidden_dim,
            4 * hidden_dim,
            bias=True
        )

        self.reset_parameters()

        self.a = nn.Parameter(torch.Tensor(1))
        self.b = nn.Parameter(torch.Tensor(1))
        self.c = nn.Parameter(torch.Tensor(1))
        nn.init.uniform_(self.a.data,a=0.5,b=0.51)  # default 0.5
        nn.init.uniform_(self.b.data,a=0.,b=1)
        nn.init.uniform_(self.c.data,a=0.5,b=0.51)  # default 0.5

    def reset_parameters(self):
        # Standard initialization
        nn.init.xavier_uniform_(self.linear.weight)

        # Forget gate bias to 1 improves stability
        with torch.no_grad():
            self.linear.bias[self.hidden_dim:2*self.hidden_dim].fill_(1.0)

    def forward(self, x_t, h_prev, c_prev, time_dists=None):
        """
        One LSTM step.
        """       

        # Concatenate input and hidden state
        combined = torch.cat([x_t, h_prev], dim=1)

        # Compute gates
        gates = self.linear(combined)

        # Split gates
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        if time_dists is not None:
            forcoff = self.a.cuda() * torch.exp( - self.c.cuda() * time_dists[0])
            incoff = self.a.cuda() * torch.exp( - self.c.cuda() * time_dists[1])

            # if len(forgetgate.shape) == 3: # for conv 1D
            #     forcoff = forcoff.unsqueeze(1).unsqueeze(2).expand_as(forgetgate)
            #     incoff = incoff.unsqueeze(1).unsqueeze(2).expand_as(ingate)
                
            # if len(forgetgate.shape) == 4: # for 2D images
            #     forcoff = forcoff.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(forgetgate)
            #     incoff = incoff.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(ingate)

            # if len(forgetgate.shape) == 5: # for 3D images
            #     forcoff = forcoff.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(forgetgate)
            #     incoff = incoff.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(ingate)
            
            forcoff = forcoff.unsqueeze(1).expand_as(f)
            incoff = incoff.unsqueeze(1).expand_as(i)
            
            f = forcoff * f
            i = incoff * i


        # Cell update
        c_t = f * c_prev + i * g

        # Hidden update
        h_t = o * torch.tanh(c_t)

        return h_t, c_t

class distLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cell = distLSTMCell(input_dim, hidden_dim)        

    def forward(self, x, time_dists=None):
        """
        x: (B, T, input_dim)
        """
        B, T, _ = x.shape
        device = x.device

        h = torch.zeros(B, self.cell.hidden_dim, device=device)
        c = torch.zeros_like(h)

        outputs = []

        if time_dists is None:
            for t in range(T):
                h, c = self.cell(x[:, t, :], h, c)
                outputs.append(h)
        else:
            for t in range(T):            
                if t == 0:
                    hx, cx = self.cell(x[:, t, :], h, c, [time_dists[:, 0], time_dists[:, 0]])
                else:
                    hx, cx = self.cell(x[:, t, :], h, c, [time_dists[:,t-1], time_dists[:,t]])
                outputs.append(hx)
        return torch.stack(outputs, dim=1)  # (B, T, hidden_dim)


if __name__ == "__main__":
    B, T, D = 2, 5, 10
    H = 16
    

    x = torch.randn(B, T, D).cuda()
    time_dists = torch.randn(B, T).cuda()
    model = distLSTM(D, H).cuda()

    y = model(x, time_dists)
    print(y.shape)
    # (2, 5, 16)
