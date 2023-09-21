import argparse
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from data import load_label, load_data, Similarity
from model import SemanticRelatedness, SemanticEmbedding

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', default=1,type=int, help='task type, should be in [1,2,3,4]')
parser.add_argument('--norm', default=1,type=int, help='Normalized flag')
parser.add_argument('-m','--matrix_tpye', default='continuous', type=str,help='Matrix type')
parser.add_argument('--distance', default='cos', type=str,help='Distance metric')
parser.add_argument('-c','--classifier', default='SVM', type=str,help='Classifier type')
parser.add_argument('-b','--batch_size', default=64, type=int,help='batch size')
parser.add_argument('--lr', default=1e-5,type=float, help='learning')
parser.add_argument('--num_epoch', default=100,type=int, help='epochs')
parser.add_argument('--eval_interval', default=1,type=int, help='eval interval')
parser.add_argument('-s', '--sigma', default=15.0,type=float, help='Sigma in rbf kernel')
parser.add_argument('-e', '--epsilon', default=0.2,type=float, help='Epsilon in Siamese')
parser.add_argument('-o','--out_dim', default=128,type=int, help='output dimensionality in Siamese and Triplet')
parser.add_argument('--nca_n', default=128,type=int, help='n_components in NCA')
parser.add_argument('--draw', default=0, type=int,help='Flag indicates whether to draw')
parser.add_argument('--cuda', default=0, type=int,help='Cuda index')
parser.add_argument('--fig_name', default='fig', type=str,help='Flag indicates whether to draw')

args = parser.parse_args()


if __name__ == '__main__':
    trainval_data, trainval_label, test_data, test_label, val_datas, val_labels = load_data()

    if args.task == 1:
        similarity = Similarity(matrix_type=args.matrix_tpye, distance=args.distance, norm=args.norm, sigma=args.sigma).getSimilarity()
        _, unseen_label, _ = load_label()
        agent = SemanticRelatedness(similarity=similarity,classifier=args.classifier)
        agent.train(X_train=trainval_data,y_train=trainval_label)
        agent.test(X_test=test_data,y_test=test_label,test_label=unseen_label)

    elif args.task == 2:
        semantic_space = Similarity(matrix_type=args.matrix_tpye, distance=args.distance, norm=args.norm).getSemanticSpace()
        train_class, test_class, _ = load_label()
        agent = SemanticEmbedding(args,semantic_space,train_class,test_class)
        agent.load_data(trainval_data, trainval_label, val_datas[0], val_labels[0], test_data, test_label)
        agent.train()
        agent.test()
    
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
        # for k in [2,5,16,32,64,128,1024]:
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

    elif args.task == 6:
        pca_acc = []
        lda_acc = []
        # for k in [2,5,16,32,64,128,1024]:
        for k in [2]:
            print("\n-------------------  k={}  --------------------".format(k),flush=True)
            pca = PCA(n_components = k)
            pca_train_data = pca.fit_transform(train_data)
            pca_test_data = pca.transform(test_data)
            pca_knn = KNeighborsClassifier(n_neighbors=5,metric='euclidean',n_jobs=-1)
            pca_knn.fit(pca_train_data, train_label)
            accuracy = pca_knn.score(pca_test_data, test_label)
            draw_fig(pca_train_data,train_label,'PCA_train')
            draw_fig(pca_test_data,test_label,'PCA_test')
            pca_acc.append(accuracy)
            print("Accuracy of pca_knn: {}".format(accuracy),flush=True)
                     
            if k <40:
                lda = LinearDiscriminantAnalysis(n_components = k)
                lda_train_data = lda.fit_transform(train_data,train_label)
                lda_test_data = lda.transform(test_data)
                lda_knn = KNeighborsClassifier(n_neighbors=5,metric='euclidean',n_jobs=-1)  
                lda_knn.fit(lda_train_data, train_label)
                draw_fig(lda_train_data,train_label,'lda_train')
                draw_fig(lda_test_data,test_label,'lda_test')
                accuracy = lda_knn.score(lda_test_data, test_label)
                lda_acc.append(accuracy)
                print("Accuracy of lda_knn: {}".format(accuracy),flush=True)

        print(pca_acc,flush=True)
        print(lda_acc,flush=True)