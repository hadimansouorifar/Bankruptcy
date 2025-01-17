from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import optimizers
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD,Adam
import numpy
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_curve
from imblearn.under_sampling import RandomUnderSampler
import random

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import csv
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import arff
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

def centroid(arr):
    ind = random.randint(0, len(minority) - 1)
    b = len(arr)
    center = arr[0]
    c = 0
    for i in range(1, b):
        center = center + arr[i]

    center = center / b
    return center



def regression():
    model = Sequential()
    model.add(Dense(112, input_dim=128))

    model.add(Dense(86))
    model.add(Dense(72))

    model.add(Dense(output_dim=64))
    return model




reg= regression()
reg.compile(loss='binary_crossentropy', optimizer=OPTIMIZER)



dataset = arff.load(open('1year.arff'))
data = np.array(dataset['data'])

x = data[:, range(0, 64)]
imp = Imputer(missing_values="NaN", strategy='median', axis=0)
x = imp.fit_transform(x)
w = data[:, 64]
w = numpy.array(w).astype('float')


acc = []
f1 = []
reca = []
prec = []
auc=[]
##################3
accsvm = []
f1svm = []
recasvm = []
precsvm = []
aucsvm=[]
#####################
accnaive = []
f1naive = []
recanaive = []
precnaive = []
aucnaive=[]
#####################
acc2 = []
f12 = []
reca2 = []
prec2 = []
auc2=[]
##################3
acc2svm = []
f12svm = []
reca2svm = []
prec2svm = []
auc2svm=[]
#####################
acc2naive = []
f12naive = []
reca2naive = []
prec2naive = []
auc2naive=[]
#####################

for mm in range(0,3):
    print(mm)
    #kf = KFold(n_splits=10, shuffle=True)
    kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
    c = 0

    for train, test in kf.split(x,w):

        train_x = x[train, :]
        test_x = x[test, :]
        train_y = w[train]
        test_y = w[test]

        y = []

        dim = train_x.shape
        #print(dim[0])
        c1 = 0
        c2 = 0
        y = []
        for i in range(0, dim[0]):

            if (int(train_y[i]) == 1):
                y.append(1)
                c1 = c1 + 1
            else:
                y.append(0)
                c2 = c2 + 1
        #print(c1)
        ntrain = 2000
        majority = numpy.zeros((c2, 64), dtype=numpy.float)
        minority = numpy.zeros((c1, 64), dtype=numpy.float)
        r1 = numpy.zeros((c2 - c1, 64), dtype=numpy.float)
        r2 = numpy.zeros((c2 - c1, 64), dtype=numpy.float)
        cent = numpy.zeros((ntrain, 64), dtype=numpy.float)

        new = numpy.zeros((c2 - c1, 128), dtype=numpy.float)

        r10 = numpy.zeros((ntrain, 64), dtype=numpy.float)
        r20 = numpy.zeros((ntrain, 64), dtype=numpy.float)
        new2 = numpy.zeros((ntrain, 128), dtype=numpy.float)

        c1 = 0
        c2 = 0
        for i in range(0, len(train_x)):
            if (int(train_y[i]) == 1):
                minority[c1] = (train_x[i])
                c1 = c1 + 1
            else:
                majority[c2] = (train_x[i])
                c2 = c2 + 1

        #############################################
        for i in range(0, len(majority) - len(minority)):
            ind = random.randint(0, len(minority) - 1)
            r1[i] = (minority[ind])

        for i in range(0, len(majority) - len(minority)):
            ind = random.randint(0, len(minority) - 1)
            r2[i] = (minority[ind])

        c = 0
        for i in range(0, len(majority) - len(minority)):
            new[i] = np.concatenate((r1[i], r2[i]), axis=None)

        ######################################################3

        for i in range(0, ntrain):
            ind = random.randint(0, len(minority) - 1)
            r10[i] = (minority[ind])

        for i in range(0, ntrain):
            ind = random.randint(0, len(minority) - 1)
            r20[i] = (minority[ind])

        c = 0
        for i in range(0, ntrain):
            new2[i] = np.concatenate((r10[i], r20[i]), axis=None)
        #############################################################




        for k in range(0, len(r10)):
            ce = []
            ce.append(r10[k])
            ce.append(r20[k])
            cent[k] = centroid(ce)


        # hist = ae.fit(minority,minority, epochs=150, verbose=0, validation_split=0.2)

        def train(X_train, epochs=150, batch=23, save_interval=200):
            gloss = []
            dloss = []
            epoch = []
            maxauc = 0
            d_loss = 10
            old = 0
            while (d_loss - old > 0.00000001):
                d_loss = reg.train_on_batch(X_train, cent)
                print(d_loss)
                old = d_loss


        train(new2)
        y_pred = reg.predict(new)

        pca = PCA(n_components=2)
        pca.fit(y_pred)
        sam = pca.transform(y_pred)
        pca.fit(minority)
        mino = pca.transform(minority)

        plt.scatter(mino[:, 0], mino[:, 1], c='red')
        plt.scatter(sam[:, 0], sam[:, 1], c='blue', marker='.')
        plt.show()

        x_combined_mino = np.concatenate((y_pred, minority))
        y_combined_mino = np.concatenate((np.ones((len(majority) - len(minority), 1)), np.ones((len(minority), 1))))

        x_combined = np.concatenate((x_combined_mino, majority))
        y_combined = np.concatenate((np.ones((len(majority), 1)), np.zeros((len(majority), 1))))


        clf = DecisionTreeClassifier(random_state=0)

        clf.fit(x_combined, y_combined)
        predictions = clf.predict(test_x)

        tt1 = numpy.array(test_y).astype('int')
        tt2 = numpy.array(predictions).astype('int')
        auct = roc_auc_score(tt1, tt2)

        acc.append(accuracy_score(test_y, predictions))
        rec = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        reca.append(rec)
        prec.append(precision_score(test_y, predictions, average='binary'))
        f1.append(f1_score(test_y, predictions, pos_label=1, average='binary'))
        auc.append(auct)
        # score = roc_auc_score(y, round)
        # print(score)

        ###################################################
        clf = svm.SVC(gamma=0.001)

        clf.fit(x_combined, y_combined)
        predictions = clf.predict(test_x)

        tt1 = numpy.array(test_y).astype('int')
        tt2 = numpy.array(predictions).astype('int')
        auct = roc_auc_score(tt1, tt2)

        accsvm.append(accuracy_score(test_y, predictions))
        rec = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        recasvm.append(rec)
        precsvm.append(precision_score(test_y, predictions, average='binary'))
        f1svm.append(f1_score(test_y, predictions, pos_label=1, average='binary'))
        aucsvm.append(auct)
        ###########################################################################

        clf = GaussianNB()
        clf.fit(x_combined, y_combined)
        predictions = clf.predict(test_x)

        tt1 = numpy.array(test_y).astype('int')
        tt2 = numpy.array(predictions).astype('int')
        auct = roc_auc_score(tt1, tt2)

        accnaive.append(accuracy_score(test_y, predictions))
        rec = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        recanaive.append(rec)
        precnaive.append(precision_score(test_y, predictions, average='binary'))
        f1naive.append(f1_score(test_y, predictions, pos_label=1, average='binary'))
        aucnaive.append(auct)
        ###########################################################################


        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_sample(train_x, train_y)

        clf = DecisionTreeClassifier(random_state=0)

        clf.fit(X_train_res, y_train_res)
        predictions = clf.predict(test_x)
        tt1 = numpy.array(test_y).astype('float')
        tt2 = numpy.array(predictions).astype('float')
        auct = roc_auc_score(tt1, tt2)

        acc2.append(accuracy_score(test_y, predictions))
        rec2 = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        reca2.append(rec2)
        prec2.append(precision_score(test_y, predictions, average='binary'))
        f12.append(f1_score(test_y, predictions, pos_label=1, average='binary'))
        auc2.append(auct)

        pca = PCA(n_components=2)
        pca.fit(X_train_res)
        sam = pca.transform(X_train_res)
        pca.fit(minority)
        mino = pca.transform(minority)

        plt.scatter(mino[:, 0], mino[:, 1], c='red')
        plt.scatter(sam[:, 0], sam[:, 1], c='blue', marker='.')
        plt.show()

        ###################################################
        clf = svm.SVC(gamma=0.001)
        clf.fit(train_x, train_y)
        predictions = clf.predict(test_x)

        tt1 = numpy.array(test_y).astype('int')
        tt2 = numpy.array(predictions).astype('int')
        auct = roc_auc_score(tt1, tt2)

        acc2svm.append(accuracy_score(test_y, predictions))
        rec = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        reca2svm.append(rec)
        prec2svm.append(precision_score(test_y, predictions, average='binary'))
        f12svm.append(f1_score(test_y, predictions, pos_label=1, average='binary'))
        auc2svm.append(auct)
        ###########################################################################

        clf = GaussianNB()
        clf.fit(train_x, train_y)
        predictions = clf.predict(test_x)

        tt1 = numpy.array(test_y).astype('int')
        tt2 = numpy.array(predictions).astype('int')
        auct = roc_auc_score(tt1, tt2)

        acc2naive.append(accuracy_score(test_y, predictions))
        rec = recall_score(test_y, predictions, labels=None, pos_label=1, average='binary', sample_weight=None)
        reca2naive.append(rec)
        prec2naive.append(precision_score(test_y, predictions, average='binary'))
        f12naive.append(f1_score(test_y, predictions, pos_label=1, average='binary'))
        auc2naive.append(auct)
###########################################################################
print("DT Results")
print(np.mean(acc))
print(np.mean(reca))
print(np.mean(prec))
print(np.mean(f1))
print(np.mean(auc))
print('####################')

print(np.mean(acc2))
print(np.mean(reca2))
print(np.mean(prec2))
print(np.mean(f12))
print(np.mean(auc2))

print('#################### STD Deep ********')
print(np.std(acc))
print(np.std(reca))
print(np.std(prec))
print(np.std(f1))
print(np.std(auc))

print('#################### STD SMOTE ********')

print(np.std(acc2))
print(np.std(reca2))
print(np.std(prec2))
print(np.std(f12))
print(np.std(auc2))

####################################################################################

print('##############################')
print("SVM Results")
print(np.mean(accsvm))
print(np.mean(recasvm))
print(np.mean(precsvm))
print(np.mean(f1svm))
print(np.mean(aucsvm))
print('####################')

print(np.mean(acc2svm))
print(np.mean(reca2svm))
print(np.mean(prec2svm))
print(np.mean(f12svm))
print(np.mean(auc2svm))
print('##########################################')

print('#################### STD Deep ********')
print(np.std(accsvm))
print(np.std(recasvm))
print(np.std(precsvm))
print(np.std(f1svm))
print(np.std(aucsvm))

print('#################### STD SMOTE ********')

print(np.std(acc2svm))
print(np.std(reca2svm))
print(np.std(prec2svm))
print(np.std(f12svm))
print(np.std(auc2svm))

print('###########################')
print("NB Results")
print(np.mean(accnaive))
print(np.mean(recanaive))
print(np.mean(precnaive))
print(np.mean(f1naive))
print(np.mean(aucnaive))
print('####################')

print(np.mean(acc2naive))
print(np.mean(reca2naive))
print(np.mean(prec2naive))
print(np.mean(f12naive))
print(np.mean(auc2naive))

print('#####################')

print('#################### STD Deep ********')
print(np.std(accnaive))
print(np.std(recanaive))
print(np.std(precnaive))
print(np.std(f1naive))
print(np.std(aucnaive))

print('#################### STD SMOTE ********')

print(np.std(acc2naive))
print(np.std(reca2naive))
print(np.std(prec2naive))
print(np.std(f12naive))
print(np.std(auc2naive))