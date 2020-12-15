from sklearn import svm
import numpy


class SVM:

    def __init__(self,c,x,y):
        self.clf = svm.SVC(kernel='linear', C=c)
        self.clf.fit(x, y)

    def test(self,x):
        return self.clf.predict(x)