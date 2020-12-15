import math


class NaiveBayes:
    def __init__(self,x,y,all_terms):
        self.x = x
        self.y = y
        self.all_terms = all_terms
        self.N = len(all_terms)
        self.neg_docs = []
        self.pos_docs = []
        for i in range(0,len(self.y)):
            if self.y[i] == 1:
                self.pos_docs.append(self.x[i])
            else:
                self.neg_docs.append(self.x[i])

    def test(self,X):
        y = []
        for x in X:
            msum = 0
            for term in x:
                msum += math.log((sum(x.count(term) for x in self.pos_docs)+1)/(sum(len(arr) for arr in self.pos_docs)+len(self.all_terms)))
            pos_prob = math.log(len(self.pos_docs)/self.N) + msum

            msum = 0
            for term in x:
                msum += math.log((sum(x.count(term) for x in self.neg_docs)+1)/(sum(len(arr) for arr in self.neg_docs)+len(self.all_terms)))
            neg_prob = math.log(len(self.neg_docs)/self.N) + msum

            if pos_prob>neg_prob:
                y.append(1)
            else:
                y.append(-1)

        return y


