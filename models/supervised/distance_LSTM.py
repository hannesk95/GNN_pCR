import torch
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


if __name__ == "__main__":
    batch = 8
    seq_len = 4
    input_size = 512
    hidden_size = 256

    x = torch.randn(batch, seq_len, input_size)
    print(x.shape)  # (8, 4, 512)

    print("Testing LSTM without time distances:")
    lstm = LSTMFromScratch(input_size, hidden_size, use_time_distances=False)
    outputs, (h, c) = lstm(x)
    print(outputs.shape)  # (8, 4, 256)

    print("Testing LSTM with time distances:")
    lstm = LSTMFromScratch(input_size, hidden_size, use_time_distances=True)
    time_distances = torch.randn(batch, seq_len, 1) # (8, 4, 1)
    outputs, (h, c) = lstm(x, time_distances=time_distances)
    print(outputs.shape)  # (8, 4, 256)

    print("Testing PyTorch built-in LSTM:")
    lstm = nn.LSTM(input_size, hidden_size)
    outputs, (h, c) = lstm(x)
    print(outputs.shape)  # (8, 4, 256)
