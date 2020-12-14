import csv
import nltk,math


class PreProccess:

    def __init__(self,file): # file will be test or train
        self.all_terms= []
        self.filePath = "./"+file+".csv"
        self.x = []
        self.y = []
        self.N=0
        self.vectors = []
        self.get_vectors()


    @staticmethod
    def prepare_text(content):
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(content)
        ps = nltk.stem.PorterStemmer()
        s_tokens = [ps.stem(x) for x in tokens]
        return s_tokens

    def get_data(self):
        rows = []
        x = []
        y = []
        with open(self.filePath, encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file)

            for row in csv_reader:
                rows.append(row)

        self.N = len(rows)
        for i in range(1, len(rows)):
            x.append(self.prepare_text(rows[i][1] + ' ' + rows[i][14]))
            y.append(int(rows[i][16]))

        self.x = x
        self.y = y
        return x, y

    def get_vectors(self):
        all_terms = {}
        vectors = []
        x, y = self.get_data()
        for doc in x:
            tf ={}
            for term in doc:
                if term in tf :
                    tf[term]+=1
                else:
                    if term not in all_terms:
                        all_terms[term] = 1
                    else:
                        all_terms[term] += 1
                    tf[term] = 1
            vector= []
            for term in all_terms.keys():
                if term not in tf:
                    vector.append(0)
                else :
                    vector.append(tf[term]*math.log(self.N/all_terms[term],10))
            vectors.append(vector)
        self.all_terms = all_terms
        self.vectors = vectors
        return vectors,y


