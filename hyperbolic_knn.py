import geoopt, torch
from tqdm import tqdm

ball = geoopt.PoincareBall()

def dist(row1, row2):
    return ball.dist(torch.tensor(row1), torch.tensor(row2)).item()
 
    
class HyperbolicKNN(object):
    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for test_row in tqdm(X_test):
            cands = self._predict_single(test_row)
            prediction = max(set(cands), key=cands.count)
            predictions.append(prediction)
        return predictions
    
    def _predict_single(self, row):
        distances = []
        for i, train_row in enumerate(self.X_train):
            distances.append(
                (self.y_train[i], dist(train_row, row))
            )
        distances.sort(key=lambda tup: tup[1])
        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][0])
        return neighbors

 