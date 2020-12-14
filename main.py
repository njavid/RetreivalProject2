from PreProccess import PreProccess
from NaiveBayes import NaiveBayes
from Evaluation import Evaluation


def main():
    evaluation = Evaluation()

    #Naive Bayes :
    train = PreProccess('train')
    naive_bayes = NaiveBayes(train.x,train.y,train.all_terms)
    naive_bayes.train()

    test = PreProccess('test')
    print(evaluation.get_accuracy(naive_bayes.test(train.x[0:200]), test.y[0:200]))

    #SVM :



    #knn:

    #random forest :

if __name__ == "__main__":
    main()
