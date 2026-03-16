import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers=[2, 50, 50, 50, 50, 1]):
        super(PINN, self).__init__()

        self.activation = nn.Tanh()

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)

        out = inputs
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))

        out = self.layers[-1](out)
        return out