import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# 定义 Transformer 模型
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)  # 将多维输入映射到模型维度
        self.decode_input_projection = nn.Linear(1, d_model)  # 将多维输入映射到模型维度
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.output_layer = nn.Linear(d_model, 1)  # 预测单个目标特征

    def forward(self, src, tgt):
        src = self.input_projection(src)
        memory = self.transformer.encoder(src)
        tgt = self.decode_input_projection(tgt)
        output = self.transformer.decoder(tgt, memory)
        output = self.output_layer(output)
        return output


# 训练过程函数
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            tgt = labels.unsqueeze(1)  # 添加一个维度
            outputs = model(inputs, tgt)
            loss = criterion(outputs, labels.unsqueeze(-1))  # 确保标签维度匹配
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')


# 测试过程函数
def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, labels)
            loss = criterion(outputs, labels.unsqueeze(-1))
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_loss = total_loss / total_samples
    print(f'Test Loss: {average_loss:.4f}')


# 数据准备
df = pd.read_csv("../data/DataProcess/station/1037A/1037A-2020-new.csv")
features = df.columns.tolist()  # 所有列名
target = 'PM2.5'  # 待预测特征
feature_cols = [col for col in features if col != target]  # 去掉目标列

# 特征缩放
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df[feature_cols])
scaled_target = scaler.fit_transform(df[[target]].values)

# 准备数据用于 Transformer
X_scaled = scaled_features.reshape((scaled_features.shape[0], 1, scaled_features.shape[1]))
y_scaled = scaled_target.reshape((scaled_target.shape[0], 1))

# 划分数据集
train_size = int(0.8 * len(X_scaled))  # 80% 训练集，20% 测试集
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# 创建 TensorDataset
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))

# 创建 DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
input_dim = len(feature_cols)  # 输入特征维度
d_model = 64
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(input_dim=input_dim, d_model=d_model, nhead=nhead,
                              num_encoder_layers=num_encoder_layers,
                              num_decoder_layers=num_decoder_layers,
                              dim_feedforward=128).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_model(model, train_loader, criterion, optimizer, device, num_epochs=100)

# 测试模型
test_model(model, test_loader, criterion, device)
