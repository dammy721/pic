### データセットはMNIST
### RNN,CNN,LSTM

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# MNISTデータセットのロードと前処理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# RNNモデルの定義
class RNNNet(nn.Module):
    def __init__(self):
        super(RNNNet, self).__init__()
        self.rnn = nn.RNN(input_size=28, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

# CNNモデルの定義
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7*7*64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# LSTMモデルの定義
class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            if isinstance(model, CNNNet):
                data = data.view(data.size(0), 1, 28, 28)  # CNNのための形状変更
            else:
                data = data.squeeze()  # RNNとLSTMのための形状変更

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# モデルのインスタンス化、損失関数、オプティマイザの設定
rnn_model = RNNNet()
cnn_model = CNNNet()
lstm_model = LSTMNet()

criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# 各モデルのトレーニング
train_model(rnn_model, train_loader, criterion, rnn_optimizer, num_epochs=5)
train_model(cnn_model, train_loader, criterion, cnn_optimizer, num_epochs=5)
train_model(lstm_model, train_loader, criterion, lstm_optimizer, num_epochs=5)
