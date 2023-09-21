import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import mahalanobis
from torch.nn.parallel import DataParallel
from draw import draw_fig

class Siamese_Agent():
    def __init__(self,x_train,y_train,x_test,y_test,device,out_dim=128) -> None:
        self.model = SiameseNetwork(out_dim)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.device = device
        self.out_dim = out_dim
        self.test_dict = {tuple(self.x_test[i,:5]):i for i in range(self.x_test.shape[0])}
        self.train_dict = {tuple(self.x_train[i,:5]):i for i in range(self.x_train.shape[0])}
        self.train_data_tensor = torch.tensor(self.x_train).float().to(self.device)
        self.test_data_tensor = torch.tensor(self.x_test).float().to(self.device)        

    def train(self, epochs=1000,lr=0.001,epsilon=0.2):
        self.model.to(self.device)
        self.model.train()

        train_dataset = SiameseDataset(self.x_train, self.y_train, epsilon)
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # 超参数
        learning_rate = lr
        num_epochs = epochs

        # 实例化模型和损失
        criterion = ContrastiveLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 训练循环
        best_acc = 0
        for epoch in range(num_epochs):
            for batch_idx, (x1, x2, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()
                x1 = x1.float().to(self.device)
                x2 = x2.float().to(self.device)
                labels = labels.to(self.device)
                output1, output2 = self.model(x1), self.model(x2)
                loss = criterion(output1, output2, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print("Epoch: {} \tBatch: {} \tLoss: {:.6f}".format(epoch, batch_idx, loss.item()),flush=True)

            if (epoch+1) % 10 == 0:
                acc = self.evaluate()
                if best_acc < acc:
                    best_acc = acc
                    torch.save(self.model.state_dict(),'model/Siamese_model_epsilon{}.pth'.format(epsilon))
                    print("Best model accuracy: {}".format(acc),flush=True)
        #  dim=2:
        if self.out_dim==2:
            self.model.eval()
            train_trans = self.model(self.train_data_tensor).to('cpu').detach().numpy()
            test_trans = self.model(self.test_data_tensor).to('cpu').detach().numpy()
            draw_fig(train_trans,self.y_train,'Sa_train')
            draw_fig(test_trans,self.y_test,'Sa_test')
        return best_acc

    def evaluate(self):
        self.model.eval()
        train_embed = torch.stack([self.model(self.train_data_tensor[i]) for i in range(self.x_train.shape[0])]).to(self.device)
        test_embed = torch.stack([self.model(self.test_data_tensor[i]) for i in range(self.x_test.shape[0])]).to(self.device)
        s_time = time.time()
        # self.Siamese_distance_metrix = [[float(torch.dist(self.train_embed[i],self.test_embed[j]).cpu())
        #                         # for i in range(self.train_data.shape[0])] for j in range(self.test_data.shape[0])]
        #                         for i in range(self.train_data.shape[0])] for j in range(1)]
        self.Siamese_distance_metrix = torch.cdist(test_embed,train_embed,p=2).to('cpu')
        del train_embed
        del test_embed
        print("Distance metrix calculated, shape {}, takes {} s".format(self.Siamese_distance_metrix.shape, time.time()-s_time),flush=True)
        distance = lambda x1,x2:self.Siamese_distance(x1,x2)
        # s_time = time.time()
        # for i in range(self.train_data.shape[0]):
        #     d = distance(self.test_data[0],self.train_data[i])
        # print("Time: {}".format(time.time()-s_time),flush=True)

        knn = KNeighborsClassifier(n_neighbors=5,metric=distance,n_jobs=-1)
        knn.fit(self.x_train,self.y_train)
        s_time = time.time()
        idx = np.arange(self.x_test.shape[0])
        np.random.shuffle(idx)
        idx = idx[:1000]
        accuracy = knn.score(self.x_test[idx],self.y_test[idx])
        print("Evaluation takes {} s, accuracy {}".format(time.time()-s_time,accuracy),flush=True)
        return accuracy

    def Siamese_distance(self,x,y):
        x_idx = self.test_dict[tuple(x[:5])]
        y_idx = self.train_dict[tuple(y[:5])]
        # return self.Siamese_distance_dict[(tuple(x[:5]),tuple(y[:5]))]
        return float(self.Siamese_distance_metrix[x_idx,y_idx])

class SiameseNetwork(nn.Module):
    def __init__(self,out_dim):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class SiameseDataset(Dataset):
    def __init__(self, x, y, epsilon):
        self.x = x
        self.y = y
        self.epsilon = epsilon

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x1 = self.x[idx]
        y1 = self.y[idx]
        e = (1e-6)* np.random.randint(1e6)
        if e < self.epsilon:
            idx = np.random.choice(np.where(self.y == y1)[0])
            x2 = self.x[idx]
            y2 = self.y[idx]
        else:
            idx = np.random.choice(self.x.shape[0])
            x2 = self.x[idx]
            y2 = self.y[idx]
        return x1, x2, torch.from_numpy(np.array(y1 == y2, dtype=np.float32))

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                          (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss