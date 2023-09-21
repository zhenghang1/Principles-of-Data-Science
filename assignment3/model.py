import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
import numpy as np
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt

class SemanticRelatedness():
    def __init__(self, similarity, classifier='SVM') -> None:
        self.similarity = similarity
        self.classifier = classifier

    def train(self,X_train,y_train):
        print("\n-----------------------------------Training---------------------------------------",flush=True)
        if self.classifier == 'SVM':
            self.model = SVC(kernel='linear', C=1.0, random_state=42, max_iter=2000)
            model_path = 'model/svm.pkl'
        elif self.classifier == 'KNN':
            self.model = KNeighborsClassifier(n_neighbors=100)
            model_path = 'model/knn_100.pkl'

        if not os.path.exists(model_path):
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, model_path)
        else:
            self.model = joblib.load(model_path)
            print(f"Load trained {self.classifier} model {model_path}",flush=True)

        # accuracy = self.model.score(X_train, y_train)
        # print("Training accuracy: {:.2f}%".format(accuracy * 100))
        
    def evaluate(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100),flush=True)

    def test(self, X_test, y_test, test_label):
        print("\n-----------------------------------Testing---------------------------------------",flush=True)
        
        if self.classifier=='SVM':
            scores = self.model.decision_function(X_test) # test_samples * 40
        elif self.classifier=='KNN':
            scores = self.model.predict_proba(X_test) # test_samples * 40
        classes = self.model.classes_.astype(int) # 40

        semantic_factor = self.similarity[(test_label-1).reshape(-1,1), classes-1].T   # (40, 10)

        y_score = np.matmul(scores, semantic_factor) 
        y_pred = test_label[np.argmax(y_score,axis=1)]

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100),flush=True)   

        exit()     


class SemanticEmbedding():
    def __init__(self, args, semantic_space, train_class, test_class) -> None:
        self.semantic_space = semantic_space
        self.args = args
        self.distance = self.args.distance
        self.train_class, self.test_class = train_class, test_class

        if args.cuda == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{args.cuda}')

    def load_data(self,X_train,y_train,X_val,y_val,X_test,y_test):
        self.X_train = X_train
        self.y_train = self.semantic_space[y_train.astype(int)-1]
        self.X_val = X_val
        self.y_val = y_val.astype(int)
        self.X_test = X_test
        self.y_test = y_test.astype(int)
        
        self.train_dataset = TensorDataset(torch.from_numpy(self.X_train),torch.from_numpy(self.y_train))
        self.train_dataloader = DataLoader(self.train_dataset,batch_size=self.args.batch_size,shuffle=True)

        self.val_dataset = TensorDataset(torch.from_numpy(self.X_val),torch.from_numpy(self.y_val))
        self.val_dataloader = DataLoader(self.val_dataset,batch_size=self.args.batch_size,shuffle=True)

        self.test_dataset = TensorDataset(torch.from_numpy(self.X_test),torch.from_numpy(self.y_test))
        self.test_dataloader = DataLoader(self.test_dataset,batch_size=self.args.batch_size,shuffle=True)


    def train(self):
        print("\n-----------------------------------Training---------------------------------------",flush=True)
        self.model = Embedding(input_dim=2048, hidden_dim=256, output_dim=85).to(self.device)
        self.train_model(self.model, self.train_dataloader)

    def train_model(self, model, dataloader):
        model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        self.train_batch_loss = []
        self.train_epoch_loss = []
        self.val_acc_list = []
        self.test_acc_list = []
        best_acc = 0
        self.best_model_path = 'xxx'
        for epoch in range(self.args.num_epoch):
            epoch_loss = 0
            for features, targets in dataloader:
                features = features.float().to(self.device)
                targets = targets.float().to(self.device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                batch_loss = loss.item()
                epoch_loss += batch_loss
                self.train_batch_loss.append(batch_loss)
            epoch_loss = epoch_loss / len(dataloader)
            self.train_epoch_loss.append(epoch_loss)
            print(f"\nEpoch [{epoch + 1}/{self.args.num_epoch}], Loss: {epoch_loss:.4f}",flush=True)

            if (epoch + 1) % self.args.eval_interval == 0:
                test_acc = self.evaluate_model(model, self.test_dataloader, test_flag=True)
                val_acc = self.evaluate_model(model, self.val_dataloader, test_flag=False)
                print(f"Accuracy on train set: {val_acc:.4f}", f"Accuracy on test set: {test_acc:.4f}",flush=True)
                self.val_acc_list.append(val_acc)
                self.test_acc_list.append(test_acc)  

                if test_acc > best_acc:
                    os.makedirs('model/embed/',exist_ok=True)
                    os.system(f'rm {self.best_model_path}')
                    best_acc = test_acc
                    self.best_model_path = f'model/embed/model_acc{best_acc:.4f}.pt'
                    torch.save(model.state_dict(),self.best_model_path)
                    print(f"New model saved: {self.best_model_path}",flush=True)
                    

    def evaluate_model(self, model, dataloader, test_flag):
        model.eval()

        with torch.no_grad():
            accuracy = []
            for features, targets in dataloader:
                features = features.float().to(self.device)
                outputs = model(features)
                pred = self.semantic_to_class(outputs.detach().cpu().numpy(), test_flag, distance=self.distance)  # 最可能的label
                accuracy.extend(list(pred==targets.numpy().astype(int)))
            accuracy = np.array(accuracy).mean()

        return accuracy

    def semantic_to_class(self, outputs, test_flag, distance):
        if distance == 'cos':
            metric = 'cosine'
        else:
            metric = distance
        distance_matrix = pairwise_distances(outputs,self.semantic_space,metric=metric)
        if test_flag:
            classes = self.test_class
        else:
            classes = self.train_class
        distance_matrix = distance_matrix[:,classes-1]
        # if distance == 'cos':
        #     pred = np.argmax(distance_matrix, axis=1)
        # else:
        #     pred = np.argmin(distance_matrix, axis=1)
        pred = np.argmin(distance_matrix, axis=1)
        pred_classes = np.tile(classes,outputs.shape[0])[pred]

        return pred_classes

    def test(self):
        print("\n-----------------------------------Testing---------------------------------------",flush=True)
        model = copy.deepcopy(self.model)
        model.load_state_dict(torch.load(self.best_model_path)) 
        print("Model load successfully",flush=True)

        acc = self.evaluate_model(model, self.test_dataloader, test_flag=True)

        print("Testing accuracy: {:.2f}%".format(acc * 100),flush=True)   

        plt.cla()
        y = np.array(self.train_batch_loss)
        x = np.arange(y.shape[0])
        plt.plot(x, y)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training loss')
        plt.savefig(f'{self.distance}_{self.args.norm}.png')

        exit()


class Embedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Embedding, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out
    

