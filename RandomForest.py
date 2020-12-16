from sklearn.ensemble import RandomForestClassifier
import numpy

class RandomForest:
    def __init__(self, train_vectors, train_category, train_set_size):
        #Create a Gaussian Classifier
        self.clf = RandomForestClassifier(n_estimators=100)
        #Train the model using the training sets y_pred=clf.predict(X_test)
        self.clf.fit(train_vectors[: train_set_size], train_category[: train_set_size])


    def test(self,test_vectors):
        return self.clf.predict(test_vectors)

        


        

                


