import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import shap

# 基础模块
class TCNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation, dropout):
        super(TCNLayer, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else hidden_size
            self.layers.append(TCNLayer(in_channels, hidden_size, kernel_size, dilation, dropout))

    def forward(self, x):
        # 输入为 (batch_size, seq_len, input_size)，调整为 (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 2, 1)  # 返回 (batch_size, seq_len, hidden_size)


class iTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, dropout=0.1):
        super(iTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(hidden_size, 1)  # 输出维度为 1，预测单个目标值

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.projection(x)


class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, tcn_hidden_size, tcn_layers, transformer_hidden_size, transformer_heads, transformer_layers):
        super(TimeSeriesPredictor, self).__init__()
        self.tcn = TCN(input_size, tcn_hidden_size, tcn_layers)
        self.transformer = iTransformer(tcn_hidden_size, transformer_hidden_size, transformer_heads, transformer_layers)

    def forward(self, x):
        tcn_out = self.tcn(x)
        transformer_out = self.transformer(tcn_out)
        return transformer_out
import numpy as np

# 生成时间序列数据
def generate_data(num_samples, seq_len, num_features):
    x = np.random.rand(num_samples, seq_len, num_features)
    y = np.mean(x, axis=1)  # 目标是预测序列均值
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 训练数据
num_samples = 1000
seq_len = 30
num_features = 10
x_train, y_train = generate_data(num_samples, seq_len, num_features)
x_test, y_test = generate_data(200, seq_len, num_features)
# 定义超参数
input_size = num_features
tcn_hidden_size = 32
tcn_layers = 2
transformer_hidden_size = 64
transformer_heads = 4
transformer_layers = 2
epochs = 20
lr = 1e-3

# 初始化模型
model = TimeSeriesPredictor(input_size, tcn_hidden_size, tcn_layers, transformer_hidden_size, transformer_heads, transformer_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# 训练
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
