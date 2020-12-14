
class Evaluation:

    @staticmethod
    def get_accuracy(y1,y2):
        count = 0
        for i in range(len(y1)):
            if y1[i] == y2[i] :
                count +=1

        return count/len(y1)