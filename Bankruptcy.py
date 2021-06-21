import numpy as np
import numpy
from LP import *
from sklearn.preprocessing import Imputer
from sklearn import metrics,svm
import csv
import arff
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn import tree
dataset = arff.load(open('1year22.arff'))
data = np.array(dataset['data'])


x = data[:,range(0,64)]
imp = Imputer(missing_values="NaN", strategy='median', axis=0)
x = imp.fit_transform(x)
w = data[:,64]
w = numpy.array(w).astype('str')
y=w
precision=[]
recall=[]
sensitivity=[]
specificity=[]
accuracy=[]
auc=[]
f1=[]
ttt=[]
vvv=[]
kf = KFold(n_splits=10,shuffle=True)
for train, test in kf.split(x):
    trainx = x[train, :]
    testx = x[test, :]
    trainy = y[train]
    testy = y[test]

    #y_pred = lp(trainx, trainy, testx, testy, -3, 3, 10, 10, 1, 0.5)

    #clf = svm.SVC( kernel="rbf", gamma=0.5, probability=True)
    clf = tree.DecisionTreeClassifier()
    clf.fit(trainx,trainy)
    y_pred =clf.predict(testx)

    #fpr, tpr, thresholds = metrics.roc_curve(testy, y_pred, pos_label='0')
    #auc.append(metrics.auc(fpr, tpr))
    c = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    reca=recall_score(testy, y_pred, labels=None, pos_label='1', average='binary', sample_weight=None)
    prec=precision_score(testy, y_pred, labels=None, pos_label='1', average='binary', sample_weight = None)
    fone=f1_score(testy, y_pred, labels=None, pos_label='1', average='binary', sample_weight=None)

    precision.append(prec)
    recall.append(reca)
    f1.append(fone)
    accuracy.append(accuracy_score(testy, y_pred))
    #sensitivity.append(tpr)
    #specificity.append(tnr)


print('accuracy : ', numpy.mean(accuracy))
print('precision : ', numpy.mean(precision))
print('recall : ', numpy.mean(recall))
print('f1 : ', numpy.mean(f1))
print('sensitivity : ', numpy.mean(sensitivity))
print('specificity : ', numpy.mean(specificity))
print('auc : ', numpy.mean(auc))

#accuracy.append(accuracy_score(testy, y_pred))

#print('accuracy : ', numpy.mean(accuracy))
print(ttt)
print(vvv)





