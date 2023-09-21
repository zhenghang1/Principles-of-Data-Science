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

class Triplet_Agent():
    def __init__(self,x_train,y_train,x_test,y_test,device,out_dim=128) -> None:
        self.model = TripletNetwork(out_dim)
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

    def train(self, epochs=1000,lr=0.001,ohsm_flag=False):
        self.model.to(self.device)
        self.model.train()

        train_dataset = TripletDataset(self.x_train, self.y_train,ohsm_flag)
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # 超参数
        learning_rate = lr
        num_epochs = epochs

        # 实例化模型和损失
        criterion = TripletLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 训练循环
        best_acc = 0
        for epoch in range(num_epochs):
            for batch_idx, (anchor, positive, negative) in enumerate(train_dataloader):
                optimizer.zero_grad()
                anchor = anchor.float().to(self.device)
                positive = positive.float().to(self.device)
                if not ohsm_flag:
                    negative = negative.float().to(self.device)
                    output1, output2, output3 = self.model(anchor), self.model(positive), self.model(negative)
                else:
                    output1, output2= self.model(anchor), self.model(positive)
                    negative = negative.float().to(self.device)
                    neg = []
                    for i in range(negative.shape[0]):
                        neg_vec = self.model(negative[i])
                        loss = np.array([criterion(output1[i].repeat(10,1).view(10,-1),output2[i].repeat(10,1).view(10,-1),neg_vec).item()])
                        min_idx = np.argmax(loss)
                        neg.append(neg_vec[min_idx])           
                    output3 = torch.stack(neg)
                loss = criterion(output1, output2, output3)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print("Epoch: {} \tBatch: {} \tLoss: {:.6f}".format(epoch, batch_idx, loss.item()),flush=True)

            if (epoch+1) % 10 == 0:
                acc = self.evaluate()
                if best_acc < acc:
                    best_acc = acc
                    torch.save(self.model.state_dict(),'model/Triplet_model_dim{}.pth'.format(self.out_dim))
                    print("Best model accuracy: {}".format(acc),flush=True)
        # dim=2:
        if self.out_dim==2:
            self.model.eval()
            train_trans = self.model(self.train_data_tensor).to('cpu').detach().numpy()
            test_trans = self.model(self.test_data_tensor).to('cpu').detach().numpy()
            draw_fig(train_trans,self.y_train,'Tri_train')
            draw_fig(test_trans,self.y_test,'Tri_test')
        return best_acc

    def evaluate(self):
        self.model.eval()
        train_embed = torch.stack([self.model(self.train_data_tensor[i]) for i in range(self.x_train.shape[0])]).to(self.device)
        test_embed = torch.stack([self.model(self.test_data_tensor[i]) for i in range(self.x_test.shape[0])]).to(self.device)
        s_time = time.time()
        # self.Triplet_distance_metrix = [[float(torch.dist(self.train_embed[i],self.test_embed[j]).cpu())
        #                         # for i in range(self.train_data.shape[0])] for j in range(self.test_data.shape[0])]
        #                         for i in range(self.train_data.shape[0])] for j in range(1)]
        self.Triplet_distance_metrix = torch.cdist(test_embed,train_embed,p=2).to('cpu')
        del train_embed
        del test_embed
        print("Distance metrix calculated, shape {}, takes {} s".format(self.Triplet_distance_metrix.shape, time.time()-s_time),flush=True)
        distance = lambda x1,x2:self.Triplet_distance(x1,x2)
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

    def Triplet_distance(self,x,y):
        x_idx = self.test_dict[tuple(x[:5])]
        y_idx = self.train_dict[tuple(y[:5])]

        return float(self.Triplet_distance_metrix[x_idx,y_idx])

class TripletNetwork(nn.Module):
    def __init__(self, out_dim):
        super(TripletNetwork, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class TripletDataset(Dataset):
    def __init__(self, x, y, ohsm_flag):
        self.x = x
        self.y = y
        self.flag = ohsm_flag

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x1 = self.x[idx]
        y1 = self.y[idx]
        idx = np.random.choice(np.where(self.y == y1)[0])
        x2 = self.x[idx]
        y2 = self.y[idx]
        if not self.flag:
            y3 = y1
            while y3 == y1:
                idx = np.random.choice(self.x.shape[0])
                x3 = self.x[idx]
                y3 = self.y[idx]
            return x1, x2, x3
        else:
            neg_list = []
            while len(neg_list) < 10:
                y3 = y1
                while y3 == y1:
                    idx = np.random.choice(self.x.shape[0])
                    x3 = self.x[idx]
                    y3 = self.y[idx]
                neg_list.append(x3)
            return x1, x2, np.array(neg_list)
        

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(loss)