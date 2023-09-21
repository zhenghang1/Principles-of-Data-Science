import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def train_AE(autoencoder, train_data, lr):
    train_data_tensor = torch.from_numpy(train_data).float()

    # 训练auto-encoder模型
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=0.0001)

    num_epochs = 100
    batch_size = 1024
    num_batches = len(train_data) // batch_size

    autoencoder.train()

    for epoch in range(num_epochs):
        for i in range(num_batches):
            batch_X = train_data_tensor[i*batch_size:(i+1)*batch_size]
            outputs = autoencoder(batch_X)
            loss = criterion(outputs, batch_X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}",flush=True)
    return autoencoder