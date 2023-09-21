import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist

def load_label():
    classes = {}
    with open("data/base/classes.txt",'r') as f:
        lines = f.readlines()
        for l in lines:
            context = l.strip().split('\t')
            assert len(context) == 2
            classes[context[1]] = int(context[0])

    trainval_classes = []
    with open("data/class_split/trainvalclasses.txt",'r') as f:
        lines = f.readlines()
        for l in lines:
            context = l.strip()
            trainval_classes.append(context)

    test_classes = []
    with open("data/class_split/testclasses.txt",'r') as f:
        lines = f.readlines()
        for l in lines:
            context = l.strip()
            test_classes.append(context)
    
    val_classes = []
    for i in range(3):
        val_class = []
        with open(f"data/class_split/valclasses{i+1}.txt",'r') as f:
            lines = f.readlines()
            for l in lines:
                context = l.strip()
                val_class.append(context)
        val_classes.append(val_class)

    trainval_label = [classes[i] for i in trainval_classes]
    test_label = [classes[i] for i in test_classes]
    val_label = [[classes[i] for i in val_class] for val_class in val_classes]

    return np.array(trainval_label), np.array(test_label), np.array(val_label)

def load_data():
    start_time = time.time()

    if not os.path.exists('data/trainval_data.npy'):
        trainval_label, test_label, val_label = load_label()

        predicates = {}
        with open("data/base/predicates.txt",'r') as f:
            lines = f.readlines()
            for l in lines:
                context = l.strip().split('\t')
                assert len(context) == 2
                predicates[context[1]] = int(context[0])

        feature = np.loadtxt("data/AwA2-features.txt",delimiter=' ')
        label = np.loadtxt("data/AwA2-labels.txt",delimiter=' ')

        trainval_data, trainval_y = [], []
        test_data, test_y = [], []
        val_datas, val_labels = [[] for i in range(3)], [[] for i in range(3)]
        for i in np.unique(label):
            idx = np.argwhere(label == i).flatten()
            if i in trainval_label:
                trainval_data.append(feature[idx])
                trainval_y.append(np.full(idx.shape[0],i))
            if i in test_label:
                test_data.append(feature[idx])
                test_y.append(np.full(idx.shape[0],i))
            for j in range(3):
                if i in val_label[j]:
                    val_datas[j].append(feature[idx])
                    val_labels[j].append(np.full(idx.shape[0],i))
                    

        trainval_data = np.concatenate(trainval_data,axis=0)
        trainval_label = np.concatenate(trainval_y,axis=0)
        test_data = np.concatenate(test_data,axis=0)
        test_label = np.concatenate(test_y,axis=0)
        for i in range(3):
            val_datas[i] = np.concatenate(val_datas[i],axis=0)
            val_labels[i] = np.concatenate(val_labels[i],axis=0)

        length = trainval_label.shape[0]
        idx = np.arange(length)
        np.random.shuffle(idx)
        trainval_data = trainval_data[idx]
        trainval_label = trainval_label[idx]

        length = test_label.shape[0]
        idx = np.arange(length)
        np.random.shuffle(idx)
        test_data = test_data[idx]
        test_label = test_label[idx]
        
        for i in range(3):
            length = val_labels[i].shape[0]
            idx = np.arange(length)
            np.random.shuffle(idx)
            val_datas[i] = val_datas[i][idx]
            val_labels[i] = val_labels[i][idx]

        np.save('data/trainval_data.npy',trainval_data)
        np.save('data/test_data.npy',test_data)
        np.save('data/trainval_label.npy',trainval_label)
        np.save('data/test_label.npy',test_label)
        for i in range(3):
            np.save(f'data/val_data{i+1}.npy',val_datas[i])
            np.save(f'data/val_label{i+1}.npy',val_labels[i])      

    else:
        trainval_data, trainval_label, test_data, test_label = np.load('data/trainval_data.npy'), np.load(
            'data/trainval_label.npy'), np.load('data/test_data.npy'), np.load('data/test_label.npy')
        val_datas = [np.load(f'data/val_data{i+1}.npy') for i in range(3)]
        val_labels = [np.load(f'data/val_label{i+1}.npy') for i in range(3)]

    print("Training data shape: {}".format(trainval_data.shape),flush=True)
    print("Testing data shape: {}".format(test_data.shape),flush=True)
    print("Training lable shape: {}".format(trainval_label.shape),flush=True)
    print("Testing label shape: {}".format(test_label.shape),flush=True)
    print("Validating datas shape: {}, {}, {}".format(val_datas[0].shape,val_datas[1].shape,val_datas[2].shape),flush=True)
    print("Validating labels shape: {}, {}, {}".format(val_labels[0].shape,val_labels[1].shape,val_labels[2].shape),flush=True)
    print("Load data succuessfully, takes {}".format(time.time()-start_time),flush=True)  

    return trainval_data, trainval_label, test_data, test_label, val_datas, val_labels

class Similarity():
    def __init__(self, matrix_type='continuous', distance='cos', norm=True, sigma=15) -> None:
        self.matrix = self.load_matrix(matrix_type)
        self.classes_num = self.matrix.shape[0]
        self.predicates_num = self.matrix.shape[1]
        if norm:
            scaler = StandardScaler()
            self.matrix = scaler.fit_transform(self.matrix)
        self.semantic_space = self.matrix
        self.similarity = self.calSimilarity(distance, sigma=sigma)

    def load_matrix(self, matrix_type):
        if matrix_type == 'binary':
            matrix_binary = np.loadtxt("data/base/predicate-matrix-binary.txt",delimiter=' ')
            return matrix_binary
        elif matrix_type == 'continuous':
            matrix_continuous = []
            with open("data/base/predicate-matrix-continuous.txt",'r') as f:
                lines = f.readlines()
                for l in lines:
                    context = l.strip().split(' ')
                    while len(context) > 85:
                        context.remove('')
                    context = list(map(float,context))
                    matrix_continuous.append(context)
            matrix_continuous = np.array(matrix_continuous)
            return matrix_continuous
        else:
            raise ValueError('Invalid matrix type, should be one of [binary, continuous]')
        
    def calSimilarity(self, distance, sigma=15):
        similarity = np.zeros((self.classes_num,self.classes_num))
        if distance == 'cos':
            norms = np.sqrt(np.sum(self.matrix ** 2, axis=1))
            self.matrix_norm = (self.matrix.T/norms).T
            similarity = np.matmul(self.matrix_norm, self.matrix_norm.T)

        elif distance == 'euclidean' or distance == 'cityblock' or distance == 'chebyshev':
            distance = pairwise_distances(self.matrix, metric=distance)
            sigma = sigma  # 带宽参数
            similarity = rbf_kernel(distance, gamma=1.0/(2*sigma**2))

        elif distance == 'correlation':
            similarity = pairwise_distances(self.matrix, metric=distance)
            similarity = -1 * similarity

        else:
            raise ValueError('Invalid distance type, should be one of [cos, euclidean]')
        
        similarity = similarity / np.linalg.norm(similarity, axis=1, keepdims=True)

        return similarity

    def getSimilarity(self):
        return self.similarity

    def getSemanticSpace(self):
        return self.semantic_space

