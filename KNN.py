import numpy 

class KNN:
    def __init__(self, k, train_vectors, train_category, train_set_size):
        self.k = k
        self.train_vectors = train_vectors[: train_set_size]
        self.train_category = train_category[: train_set_size]


    def test(self,test_vectors):
        calculated_category = []
        for i in range(len(test_vectors)):
            distance_dict = {}
            for j in range(len(self.train_vectors)):
                point1 = numpy.array(test_vectors[i])
                point2 = numpy.array(self.train_vectors[j])

                # calculating Euclidean distance 
                # using linalg.norm() 
                distance_dict[j] = numpy.linalg.norm(point1 - point2)

            #sort distances 
            sorted_dict = sorted(distance_dict.items(), key=lambda kv: kv[1])

            #k_nearest neighbors and their category
            k_top_neighbor = [a for (a,b) in sorted_dict[:self.k]]
            k_top_category = [self.train_category[i] for i in k_top_neighbor]

            #Assign the test data point to that category for which the number of the neighbor is maximum
            calculated_category.append(max(set(k_top_category), key = k_top_category.count))

        return calculated_category
                


