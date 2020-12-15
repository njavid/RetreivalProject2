from PreProcess import PreProcess
from NaiveBayes import NaiveBayes
from Evaluation import Evaluation
from SVM import SVM
import numpy as np


def main():
    PreProcess.set_all_terms()

    evaluation = Evaluation()

    # Naive Bayes :
    # train = PreProcess('train',0,1)
    # test = PreProcess('test',0,1)
    # naive_bayes = NaiveBayes(train.x,train.y,train.all_terms)
    # print(evaluation.get_accuracy(naive_bayes.test(train.x[0:200]), test.y[0:200]))

    # SVM :

    train = PreProcess('train',0,.9)
    validation = PreProcess('train',.9,1)
    out = []
    for c in [1/2,1,3/2,2]:
        svm = SVM(c,train.vectors,train.y)
        out.append(evaluation.get_accuracy(svm.test(validation.vectors),validation.y))
    print(out)



    #knn:

    #random forest :

if __name__ == "__main__":
    main()
