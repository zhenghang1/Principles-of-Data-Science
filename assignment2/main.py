import argparse
import time
import numpy as np
from draw import draw_fig
from sklearn.neighbors import KNeighborsClassifier
from model import MetrixLearning
from k_fold import kFoldCrossValidation
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', default=1,type=int, help='task type, should be in [1,2,3,4]')
parser.add_argument('-m', '--method', default=1,type=int, help='Matrix learning method, should be in [1,2,3,4]')
parser.add_argument('-n', '--norm', default=0,type=int, help='Normalized flag')
parser.add_argument('-e', '--epsilon', default=0.2,type=float, help='Epsilon in Siamese')
parser.add_argument('-o','--out_dim', default=128,type=int, help='output dimensionality in Siamese and Triplet')
parser.add_argument('--nca_n', default=128,type=int, help='n_components in NCA')
parser.add_argument('--epoch', default=100,type=int, help='epochs')
parser.add_argument('--draw', default=0, type=int,help='Flag indicates whether to draw')
parser.add_argument('-c','--cuda', default=0, type=int,help='Cuda index')
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

if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_data()

    if args.task == 1:
        # KNN
        start_time = time.time()  
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(train_data, train_label)
        accuracy = knn.score(test_data, test_label)
        print("Accuracy: {}".format(accuracy),flush=True)
        print("KNN fitting and predicting time: {}".format(time.time()-start_time),flush=True)  

    elif args.task == 2:
        # k-Fold Cross Validation
        tester = kFoldCrossValidation(train_data, train_label, test_data, test_label, norm=args.norm)
        tester.runValidation(k=5,n_list=list(range(1,11)),w_list=['uniform','distance'])
    
    elif args.task == 3:
        start_time = time.time()
        for w in ['uniform','distance']:
            print('\n\n******************** {} ************************'.format(w),flush=True)
            euclidean_acu = []
            manhattan_acu = []
            chebyshev_acu = []
            for k in list(range(1,11))+[15,20,50,100,200,1000]:
                print("\n-------------------  k={}  --------------------".format(k),flush=True)
                knn_euclidean = KNeighborsClassifier(n_neighbors=k,metric='euclidean',weights=w,n_jobs=-1)
                knn_manhattan = KNeighborsClassifier(n_neighbors=k,metric='manhattan',weights=w,n_jobs=-1)
                knn_chebyshev = KNeighborsClassifier(n_neighbors=k,metric='chebyshev',weights=w,n_jobs=-1)

                knn_euclidean.fit(train_data, train_label)
                accuracy = knn_euclidean.score(test_data, test_label)
                euclidean_acu.append(accuracy)
                print("Accuracy of knn_euclidean: {}".format(accuracy),flush=True)

                knn_manhattan.fit(train_data, train_label)
                accuracy = knn_manhattan.score(test_data, test_label)
                manhattan_acu.append(accuracy)
                print("Accuracy of knn_manhattan: {}".format(accuracy),flush=True)

                knn_chebyshev.fit(train_data, train_label)
                accuracy = knn_chebyshev.score(test_data, test_label)
                chebyshev_acu.append(accuracy)
                print("Accuracy of knn_chebyshev: {}".format(accuracy),flush=True)

            print(euclidean_acu,flush=True)
            print(manhattan_acu,flush=True)
            print(chebyshev_acu,flush=True)

    elif args.task == 4:
        start_time = time.time()
        agent = MetrixLearning(train_data, train_label, test_data, test_label, norm=args.norm, cuda=args.cuda)
        
        # Mahalanobis
        if args.method == 1:
            accuracy = agent.runMetrixLearning('Mahalanobis')
        
        if args.method == 2:
            accuracy = agent.runMetrixLearning('Siamese',epsilon=args.epsilon, epoch=args.epoch, out_dim=args.out_dim)

        if args.method == 3:
            accuracy = agent.runMetrixLearning('Triplet', epoch=args.epoch, out_dim=args.out_dim)

        if args.method == 4:
            accuracy = agent.runMetrixLearning('NCA', epoch=args.epoch,nca_n=args.nca_n)

        print("\nMetrix Learning using {}".format(args.method),flush=True)
        print("Accuracy: {}".format(accuracy),flush=True)
        print("Fitting and predicting time: {}".format(time.time()-start_time),flush=True)    

    elif args.task == 5:
        pca = PCA(n_components = 48)
        s_time = time.time()
        pca_train_data = pca.fit_transform(train_data)
        pca_test_data = pca.transform(test_data)
        print("PCA done, takes {} s".format(time.time()-s_time),flush=True)
        np.save('pca_train.npy',pca_train_data)
        np.save('pca_test.npy',pca_test_data)

        lda = LinearDiscriminantAnalysis(n_components = 48)
        lda_train_data = lda.fit_transform(train_data,train_label)
        lda_test_data = lda.transform(test_data)

        acc = []
        pca_acc = []
        lda_acc = []
        for k in list(range(1,11))+[15,20,50,100,200,1000]:
            print("\n-------------------  k={}  --------------------".format(k),flush=True)
            knn = KNeighborsClassifier(n_neighbors=k,metric='euclidean',n_jobs=-1)
            pca_knn = KNeighborsClassifier(n_neighbors=k,metric='euclidean',n_jobs=-1)
            lda_knn = KNeighborsClassifier(n_neighbors=k,metric='euclidean',n_jobs=-1)

            knn.fit(train_data, train_label)
            accuracy = knn.score(test_data, test_label)
            acc.append(accuracy)
            print("Accuracy of knn: {}".format(accuracy),flush=True)

            pca_knn.fit(pca_train_data, train_label)
            accuracy = pca_knn.score(pca_test_data, test_label)
            pca_acc.append(accuracy)
            print("Accuracy of pca_knn: {}".format(accuracy),flush=True)

            lda_knn.fit(lda_train_data, train_label)
            accuracy = lda_knn.score(lda_test_data, test_label)
            lda_acc.append(accuracy)
            print("Accuracy of lda_knn: {}".format(accuracy),flush=True)

        print(acc,flush=True)
        print(pca_acc,flush=True)
        print(lda_acc,flush=True)

