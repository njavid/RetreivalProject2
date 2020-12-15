import csv
import nltk,math


class PreProcess:
    all_terms = {}
    N = 0

    # s & t : for set train and validation data in range [0,1]
    def __init__(self,file,s,t):
        self.s = s; self.t = t
        self.filePath = "./"+file+".csv"
        self.x = []; self.y = []
        self.vectors = []
        self.get_vectors()

    def get_data(self):
        rows = []
        x = []
        y = []
        with open(self.filePath, encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file)

            for row in csv_reader:
                rows.append(row)

        for i in range(1, len(rows)):
            x.append(self.prepare_text(rows[i][1] + ' ' + rows[i][14]))
            y.append(int(rows[i][16]))

        n = len(rows)
        self.x = x[int(self.s*n):int(self.t*n)]
        self.y = y[int(self.s*n):int(self.t*n)]

    def get_vectors(self):
        vectors = []
        self.get_data()
        x,y = self.x, self.y
        for doc in x:
            tf ={}
            for term in doc:
                if term in tf :
                    tf[term]+=1
                else:
                    tf[term] = 1
            vector= []
            for term in self.all_terms.keys():
                if term not in tf:
                    vector.append(0)
                else :
                    vector.append(tf[term]*math.log(self.N/self.all_terms[term],10))
            vectors.append(vector)
        self.all_terms = self.all_terms
        self.vectors = vectors
        return vectors,y

    @staticmethod
    def prepare_text(content):
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(content)
        ps = nltk.stem.PorterStemmer()
        s_tokens = [ps.stem(x) for x in tokens]
        return s_tokens

    @staticmethod
    def set_all_terms():
        rows = []
        x = []
        with open("./train.csv", encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                rows.append(row)
        with open("./test.csv", encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                rows.append(row)

        for i in range(1, len(rows)):
            x.append(PreProcess.prepare_text(rows[i][1] + ' ' + rows[i][14]))

        for doc in x:
            tf = {}
            for term in doc:
                if term not in tf :
                    if term not in PreProcess.all_terms:
                        PreProcess.all_terms[term] = 1
                    else:
                        PreProcess.all_terms[term] += 1
                    tf[term] = 1
        PreProcess.N = len(rows)
