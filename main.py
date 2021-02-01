from PreProcess import PreProcess
from NaiveBayes import NaiveBayes
from Evaluation import Evaluation
from RandomForest import RandomForest
from sklearn.metrics import classification_report
from SVM import SVM
from KNN import KNN
import numpy as np


def main():
    PreProcess.set_all_terms()

    evaluation = Evaluation()

    # Naive Bayes :
    # train = PreProcess('train',0,1)
    test = PreProcess('test',0,1)
    # naive_bayes = NaiveBayes(train.x,train.y,train.all_terms)
    # print(evaluation.get_accuracy(naive_bayes.test(train.x[0:200]), test.y[0:200]))

    # SVM :

    train = PreProcess('train',0,.9)
    validation = PreProcess('train',.9,1)
    out = []
    #for c in [1/2,1,3/2,2]:
    #    svm = SVM(c,train.vectors,train.y)
    #    out.append(evaluation.get_accuracy(svm.test(validation.vectors),validation.y))
    #print(out)

    
    #knn:
    #train_set_size = 10000
    #validation_set_size = 15
    #for k in [1, 5, 9]:
    #    knn = KNN(k,train.vectors, train.y, train_set_size)
    #    print("For k = ", k, "accuracy is equal to: ", evaluation.get_accuracy(knn.test(validation.vectors[:validation_set_size]) , validation.y[:validation_set_size]))

    #random forest:
    #train_set_size = 1000
    #validation_set_size = 10
    #random_forest = RandomForest(train.vectors, train.y, train_set_size)
    #print("Accuracy: ", evaluation.get_accuracy(random_forest.test(validation.vectors[:validation_set_size]) , validation.y[:validation_set_size]))

    #final evaluation and report:
    #naive_bayes = NaiveBayes(train.x[:10000],train.y[:10000],train.all_terms)
    #print("For Naive Bayes:")
    #print(classification_report(naive_bayes.test(train.x[:15]), test.y[:15]))

    #svm = SVM(2,train.vectors[:10000],train.y[:10000])
    #print("For SVM:")
    #print(classification_report(svm.test(validation.vectors[:15]),validation.y[:15]))

    #knn = KNN(1,train.vectors, train.y, 10000)
    #print("For KNN:")
    #print(classification_report(knn.test(validation.vectors[:15]) , validation.y[:15]))

    #random_forest = RandomForest(train.vectors, train.y, 10000)
    #print("For Random Forest:")
    #print(classification_report(random_forest.test(validation.vectors[:15]) , validation.y[:15]))
    


    
    



if __name__ == "__main__":
    main()
