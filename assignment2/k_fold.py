import numpy as np
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from sklearn.neighbors import KNeighborsClassifier

class kFoldCrossValidation():
    def __init__(self,train_data, train_label, test_data, test_label, norm=False) -> None:
        self.train_data, self.train_label = train_data, train_label
        self.test_data, self.test_label = test_data, test_label

        if norm:
            scaler = StandardScaler()
            self.train_data = scaler.fit_transform(self.train_data)
            self.test_data = scaler.fit_transform(self.test_data)

    def runValidation(self,k,n_list,w_list):
        self.data_processing(k)

        acc_matrix = []
        for n in n_list:
            acc_list = []
            for l in w_list:
                acc = self.validate(k,n,l)
                acc_list.append(acc)
            acc_matrix.append(acc_list)
        
        row_header = ['n_neighbors = '+str(n) for n in n_list]
        column_header = ['weight = '+str(l) for l in w_list]

        table = tabulate(acc_matrix, headers=column_header, tablefmt='fancy_grid',showindex=row_header)
        print('\n\n')
        print(table,flush=True)
        np.save('acc.npy',acc_matrix)

    def data_processing(self,k):
        k_train_data, k_train_label= self.k_spilt(self.train_data,self.train_label,k)
        k_test_data, k_test_label= self.k_spilt(self.test_data,self.test_label,k)
        k_data = np.concatenate((k_train_data,k_test_data),axis=1)
        k_label = np.concatenate((k_train_label,k_test_label),axis=1)

        self.train_set = []
        self.test_set = []
        for i in range(k):
            index = list(range(k))
            index.remove(i)
            self.train_set.append((k_data[index,:,:].reshape(-1,2048),k_label[index,:].reshape(-1)))
            self.test_set.append((k_data[i,:,:].reshape(-1,2048),k_label[i,:].reshape(-1)))

    def k_spilt(self,data,label,k):
        index = np.arange(data.shape[0])[:-1*int(data.shape[0]%k)]
        assert index.shape[0] % k == 0
        np.random.shuffle(index)
        k_index = np.array(np.array_split(index, k))
        return data[k_index], label[k_index]
    
    def validate(self,k,n,w):
        accuracy = []
        for i in range(k):
            knn = KNeighborsClassifier(n_neighbors=n,weights=w)
            knn.fit(self.train_set[i][0], self.train_set[i][1])

            # 在测试集上评估KNN分类器
            acc = knn.score(self.test_set[i][0], self.test_set[i][1])
            accuracy.append(acc)
        
        avg_acc = np.mean(accuracy)
        print("Using n_neighbors {} and weight {}, the average accuracy is {}".format(n,w,avg_acc),flush=True)
        return avg_acc
    