import os
import time
from draw import draw_fig
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
from Siamese import Siamese_Agent
from Triplet import Triplet_Agent
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn import metrics


class MetrixLearning():
    def __init__(self,train_data, train_label, test_data, test_label, norm=True, cuda=0) -> None:
        self.train_data, self.train_label = train_data, train_label
        self.test_data, self.test_label = test_data, test_label

        # for i in range(1,50):
        #     train_data = self.train_data[:,:i]
        #     test_data = self.test_data[:,:i]
        #     all_data = np.concatenate((train_data,test_data))
        #     print(all_data.shape)
        #     length = all_data.shape[0]
        #     all_data = np.unique(all_data,axis=0)
        #     length2 = all_data.shape[0]
        #     print(length,length2,flush=True)

        #     if length == length2:
        #         print("Find i {}".format(i))

        # exit()

        if norm:
            self.data_norm()
        self.norm = norm

        # GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(cuda))
            print(f'There are {torch.cuda.device_count()} GPU(s) available.',flush=True)
            print('Device name:', torch.cuda.get_device_name(0),flush=True)

        else:
            print('No GPU available, using the CPU instead.',flush=True)
            self.device = torch.device("cpu")

    def runMetrixLearning(self,method,epsilon=0.2,epoch=100,nca_n=128,out_dim=128):
        if method == 'Mahalanobis':
            idx = np.arange(self.test_data.shape[0])
            np.random.shuffle(idx)
            idx = idx[:1000]
            self.test_data = self.test_data[idx]
            self.train_data = self.train_data[idx]
            self.test_label = self.test_label[idx]
            self.train_label = self.train_label[idx]
            # if not self.norm:
            #     self.data_norm()
            #     print("Data normalized",flush=True)
            self.cov_matrix = np.cov(self.train_data.T)
            self.cov_matrix_inv = torch.inverse(torch.from_numpy(self.cov_matrix)).to(self.device)
            # distance = lambda x1,x2:mahalanobis(x1,x2,cov_matrix)

            # ma_time_com1.txt
            # s_time = time.time()
            # for i in range(self.train_data.shape[0]):
            #     d = self.mahalanobis_distance_cal(self.train_data_tensor[i].double(),self.test_data_tensor[0].double()) 
            # print("Time: {}".format(time.time()-s_time),flush=True)

            # s_time = time.time()
            # for i in range(self.train_data.shape[0]):
            #     d = mahalanobis(self.train_data[i],self.test_data[0],cov_matrix) 
            # print("Time: {}".format(time.time()-s_time),flush=True)      

            # exit()      

            distance = lambda x1,x2:self.mahalanobis_distance(x1,x2)

            knn = KNeighborsClassifier(n_neighbors=5,metric=distance,n_jobs=-1)
            knn.fit(self.train_data,self.train_label)
            accuracy = knn.score(self.test_data, self.test_label)
            return accuracy

        if method == 'Siamese':
            agent = Siamese_Agent(x_train=self.train_data,y_train=self.train_label,x_test=self.test_data,y_test=self.test_label,device=self.device,out_dim=out_dim)
            accuracy = agent.train(epochs=epoch,lr=1e-4,epsilon=epsilon)
            return accuracy
        
        if method == 'Triplet':
            agent = Triplet_Agent(x_train=self.train_data,y_train=self.train_label,x_test=self.test_data,y_test=self.test_label,device=self.device,out_dim=out_dim)
            accuracy = agent.train(epochs=epoch,lr=1e-4,ohsm_flag=True)
            return accuracy
        
        if method == 'NCA':
            nca = NeighborhoodComponentsAnalysis(n_components=nca_n, random_state=42)
            nca.fit(self.train_data, self.train_label)
           
            X_train_nca = nca.transform(self.train_data)
            X_test_nca = nca.transform(self.test_data)

            if nca_n == 2:
                draw_fig(X_train_nca,self.train_label,'NCA_train')
                draw_fig(X_test_nca,self.test_label,'NCA_test')

            knn = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
            knn.fit(X_train_nca, self.train_label)

            y_pred = knn.predict(X_test_nca)
            accuracy = metrics.accuracy_score(self.test_label, y_pred)
            precision = metrics.precision_score(self.test_label, y_pred, average='macro')
            recall = metrics.recall_score(self.test_label, y_pred, average='macro')
            f1_score = metrics.f1_score(self.test_label, y_pred, average='macro')

            path = 'log/nca/log_norm.npy' if self.norm else 'log/nca/log_nonorm.npy'
            if not os.path.exists(path):
                log_list = [[] for i in range(5)]
            else:
                log_list = list(np.load(path))
            
            log_list[0] = np.append(log_list[0],nca_n)
            log_list[1] = np.append(log_list[1],accuracy)
            log_list[2] = np.append(log_list[2],precision)
            log_list[3] = np.append(log_list[3],recall)
            log_list[4] = np.append(log_list[4],f1_score)
            np.save(path,np.array(log_list))
            
            return accuracy
            
    def mahalanobis_distance(self,x,y):
        x = torch.tensor(x).to(self.device)
        y = torch.tensor(y).to(self.device)

        return self.mahalanobis_distance_cal(x,y)

    def mahalanobis_distance_cal(self,x,y):
        diff = x - y
        distance = torch.sqrt(torch.sum(torch.matmul(diff, self.cov_matrix_inv * diff),dim=-1))
        # print('diff',diff.shape)
        # print('m1',(self.cov_matrix_inv * diff).shape)
        # print('m2',torch.matmul(diff, self.cov_matrix_inv * diff).shape)
        # sum = torch.sum(torch.matmul(diff, self.cov_matrix_inv * diff),dim=-1)
        # print('sum',sum,sum.shape)
        return float(distance.cpu())
    
    def data_norm(self):
        scaler = StandardScaler()
        self.train_data = scaler.fit_transform(self.train_data)
        self.test_data = scaler.fit_transform(self.test_data)       





