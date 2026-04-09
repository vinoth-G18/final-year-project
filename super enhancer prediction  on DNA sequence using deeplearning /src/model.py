import torch
import torch.nn as nn

class EnhancerModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(4, 64, kernel_size=15, padding=7)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=15, padding=7)
        self.pool2 = nn.MaxPool1d(2)

        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):

        # input: (batch, 3000, 4)
        x = x.permute(0,2,1)

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.permute(0,2,1)

        x,_ = self.attention(x,x,x)

        x = torch.mean(x, dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.sigmoid(self.fc2(x))

        return x