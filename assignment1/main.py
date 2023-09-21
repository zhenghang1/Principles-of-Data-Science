import argparse
import time
import numpy as np
from draw import draw_fig
from auto_encoder import AutoEncoder,train_AE
import sklearn.svm as svm
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', default='FS',type=str, help='DR methods, should be in [FS, BS]')
parser.add_argument('-c', '--penalty', default=1,type=float, help='Penalty of SVM')
parser.add_argument('-k', '--kernel', default='linear',type=str, help='Kernel Function of SVM')
parser.add_argument('-p', '--probability', default=1, type=int,help='Activate SVM Probability Estimation')
parser.add_argument('--vt_threshold', default=0.5, type=float,help='threshold in Variance Threshold')
parser.add_argument('--tb_threshold', default=1e-3, type=float,help='threshold in Tree-based selection')
parser.add_argument('--pca', default=1, type=int, help='Conduct pca')
parser.add_argument('-d','--dimention', default=0.99, type=float,help='pca parameter n_components')
parser.add_argument('--lr', default=1e-2, type=float,help='Auto encoder learning rate')
parser.add_argument('--pca_kernel', default='poly', type=str,help='pca kernel')
parser.add_argument('--norm', default=1, type=int,help='Conduct normalization')
parser.add_argument('--draw', default=0, type=int,help='Flag indicates whether to draw')
parser.add_argument('--fig_name', default='fig', type=str,help='Flag indicates whether to draw')


args = parser.parse_args()


def load_data():
    start_time = time.time()
    # feature = np.loadtxt("data\AwA2-features.txt",delimiter=' ')
    # # file_names = np.loadtxt("data\AwA2-filenames.txt",delimiter=' ')
    # label = np.loadtxt("data\AwA2-labels.txt",delimiter=' ')

    # train_data, train_label = None, None
    # test_data, test_label = None, None

    # for i in np.unique(label):
    #     idx = np.argwhere(label == i)
    #     rows = len(idx)
    #     ran_row_idx = np.arange(rows)
    #     np.random.shuffle(ran_row_idx)
    #     train_idx, test_idx = idx[ran_row_idx[:int(rows*0.6)]].flatten(), idx[ran_row_idx[int(rows*0.6):]].flatten()

    #     if i == 1 :
    #         train_data = feature[train_idx]
    #         train_label = label[train_idx]
    #         test_data = feature[test_idx]
    #         test_label = label[test_idx]
    #     else:
    #         train_data = np.concatenate((train_data,feature[train_idx]))
    #         test_data= np.concatenate((test_data,feature[test_idx]))
    #         train_label = np.concatenate((train_label,label[train_idx]))
    #         test_label = np.concatenate((test_label,label[test_idx]))

    train_data, train_label, test_data, test_label = np.load('data/train_data.npy'), np.load(
        'data/train_label.npy'), np.load('data/test_data.npy'), np.load('data/test_label.npy')

    print("Training data shape: {}".format(train_data.shape),flush=True)
    print("Testing data shape: {}".format(test_data.shape),flush=True)
    print("Training lable shape: {}".format(train_label.shape),flush=True)
    print("Testing label shape: {}".format(test_label.shape),flush=True)
    print("Load data succuessfully, takes {}".format(time.time()-start_time),flush=True)

    # np.save('data/train_data.npy',train_data)
    # np.save('data/test_data.npy',test_data)
    # np.save('data/train_label.npy',train_label)
    # np.save('data/test_label.npy',test_label)

    return train_data, train_label, test_data, test_label


class DR_Transformer():
    def __init__(self, train_data, train_label, test_data, test_label) -> None:
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        
    def runDementinoalityReduction(self, method):
        if method == 'FS':
            start_time = time.time()
            print("Starting dementionality reduction with forward selection",flush=True)
            model = LogisticRegression(max_iter=2000)
            sfs = SequentialFeatureSelector(model,n_features_to_select=10,direction='forward')
            transformed_train_data = sfs.fit_transform(self.train_data,self.train_label)
            transformed_test_data = sfs.transform(self.test_data)
            print("Dementionality reduction with forward selection done, takes {}".format(time.time()-start_time),flush=True)
            
        elif method == 'BS':
            start_time = time.time()
            print("Starting dementionality reduction with backward selection",flush=True)
            model = LogisticRegression(max_iter=2000)
            sfs = SequentialFeatureSelector(model,n_features_to_select=0.05,direction='backward')
            transformed_train_data = sfs.fit_transform(self.train_data,self.train_label)
            transformed_test_data = sfs.transform(self.test_data)
            print("Dementionality reduction with backward selection done, takes {}".format(time.time()-start_time),flush=True)
        elif method == 'VT':
            start_time = time.time()
            print("Starting dementionality reduction with Variance Threshold",flush=True)
            vt = VarianceThreshold(threshold=args.vt_threshold)
            transformed_train_data = vt.fit_transform(self.train_data,self.train_label)
            transformed_test_data = vt.transform(self.test_data)
            print("Dementionality reduction with Variance Threshold done, takes {}".format(time.time()-start_time),flush=True)
            
        elif method == 'TB':
            start_time = time.time()
            print("Starting dementionality reduction with Tree-Based selection",flush=True)
            tb = ExtraTreesClassifier(n_estimators=10, random_state=42)
            sfm = SelectFromModel(estimator=tb, threshold=args.tb_threshold)
            transformed_train_data = sfm.fit_transform(self.train_data,self.train_label)
            transformed_test_data = sfm.transform(self.test_data)
            print("Dementionality reduction with Tree-Based selection done, takes {}".format(time.time()-start_time),flush=True)
            
        elif method == 'AE':
            start_time = time.time()
            print("Starting dementionality reduction with Auto Encoder",flush=True)
            scaler = StandardScaler()
            self.train_data, self.test_data = scaler.fit_transform(self.train_data),scaler.fit_transform(self.test_data)
            input_dim = self.train_data.shape[1]
            hidden_dim = args.dimention
            if hidden_dim>1.0:
                hidden_dim = int(hidden_dim)

            model = AutoEncoder(input_dim, hidden_dim)
            model = train_AE(model,self.train_data,args.lr)
            model.eval()
            encoder = model.encoder
            train_data_tensor = torch.from_numpy(self.train_data).float()
            test_data_tensor = torch.from_numpy(self.test_data).float()
            transformed_train_data = encoder(train_data_tensor).detach().numpy()
            transformed_test_data = encoder(test_data_tensor).detach().numpy()
            print("Dementionality reduction with Auto Encoder done, takes {}".format(time.time()-start_time),flush=True)
            
        elif method == 'PCA':
            start_time = time.time()
            print("Starting dementionality reduction with PCA",flush=True)
            n = args.dimention
            if n > 1.0:
                n = int(n)
            pca = PCA(n_components = n)
            all_data = np.concatenate((self.train_data,self.test_data))
            pca.fit(all_data)
            transformed_train_data = pca.transform(self.train_data)
            transformed_test_data = pca.transform(self.test_data)
            print("Dementionality reduction with PCA done, takes {}".format(time.time()-start_time),flush=True)
            
        elif method == 'K_PCA':
            start_time = time.time()
            print("Starting dementionality reduction with Kernel PCA",flush=True)
            n = args.dimention
            kernel = args.pca_kernel
            if n > 1.0:
                n = int(n)
            pca = KernelPCA(n_components = n, kernel=kernel)
            transformed_train_data = pca.fit_transform(self.train_data)
            transformed_test_data = pca.transform(self.test_data)
            print("Dementionality reduction with Kernel PCA done, takes {}".format(time.time()-start_time),flush=True)
            
        elif method == 'LDA':
            start_time = time.time()
            print("Starting dementionality reduction with LDA",flush=True)
            n = args.dimention
            if n > 1.0:
                n = int(n)
            lda = LinearDiscriminantAnalysis(n_components = n)
            transformed_train_data = lda.fit_transform(self.train_data,self.train_label)
            transformed_test_data = lda.transform(self.test_data)
            print("Dementionality reduction with LDA done, takes {}".format(time.time()-start_time),flush=True)

        elif method == 'TSNE':
            start_time = time.time()
            print("Starting dementionality reduction with t-SNE",flush=True)
            n = args.dimention
            if n > 1.0:
                n = int(n)
            tsne = TSNE(n_components=n, random_state=42)
            transformed_train_data = tsne.fit_transform(self.train_data)
            transformed_test_data = tsne.fit_transform(self.test_data)
            print("Dementionality reduction with t-SNE done, takes {}".format(time.time()-start_time),flush=True)

        elif method == 'LLE':
            start_time = time.time()
            print("Starting dementionality reduction with LLE",flush=True)
            n = args.dimention
            if n > 1.0:
                n = int(n)
            lle = LocallyLinearEmbedding(n_components=n, random_state=42, n_neighbors=20)
            transformed_train_data = lle.fit_transform(self.train_data,self.train_label)
            transformed_test_data = lle.transform(self.test_data)
            print("Dementionality reduction with LLE done, takes {}".format(time.time()-start_time),flush=True)

        if args.draw==1:
            draw_fig(transformed_train_data,self.train_label,args.fig_name+'_train')
            draw_fig(transformed_test_data,self.test_label,args.fig_name+'_test')
            exit()
        return transformed_train_data, transformed_test_data


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_data()

    # # Non-DR
    # start_time = time.time()
    # svm = svm.SVC(C=args.penalty, kernel=args.kernel, probability=args.probability, max_iter=2000)
    # svm.fit(train_data,train_label)
    # score = svm.score(test_data,test_label)
    # print("\n------------------------Non Dementionality Reduction---------------------------")
    # print("Accuracy: {}".format(score))
    # print("Time: {}".format(time.time()-start_time))


    trans = DR_Transformer(train_data, train_label, test_data, test_label)

    if args.method == 'FS':
        # Forward Selection
        print("\n####### Forward Selection ########",flush=True)
    elif args.method == 'BS':
    # Backward Selection
        print("\n####### Backward Selection ########",flush=True)
    elif args.method == 'AE':
    # Auto Encoder
        print("\n####### Auto Encoder ########",flush=True)
    elif args.method == 'VT':
    # Variance Threshold
        print("\n####### Variance Threshold ########",flush=True)
    elif args.method == 'TB':
    # Tree-Based selection
        print("\n####### Tree-Based selection ########",flush=True)
    # PCA
    elif args.method == 'PCA':
        print("\n####### PCA ########",flush=True)
    # Kernel PCA
    elif args.method == 'K_PCA':
        print("\n####### Kernel PCA ########",flush=True)
    # LDA
    elif args.method == 'LDA':
        print("\n####### LDA ########",flush=True)
    # t-SNE
    elif args.method == 'TSNE':
        print("\n####### t-SNE ########",flush=True)
    # LLE
    elif args.method == 'LLE':
        print("\n####### LLE ########",flush=True)

    transformed_train_data, transformed_test_data = trans.runDementinoalityReduction(method = args.method)
    d = transformed_train_data.shape[1]
    print("Reduced dementionality: {}".format(d),flush=True)
    start_time = time.time()
    svm = svm.SVC(C=args.penalty, kernel=args.kernel, probability=args.probability, max_iter=2000)
    svm.fit(transformed_train_data,train_label)
    score = svm.score(transformed_test_data,test_label)
    print("Accuracy: {}".format(score),flush=True)
    print("SVM fitting and predicting time: {}".format(time.time()-start_time),flush=True)  

