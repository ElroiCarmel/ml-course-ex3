import logging
import experiments_csv as ex
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data() -> tuple[pd.DataFrame, LabelEncoder]:
    data = pd.read_csv(
        "./iris.txt",
        sep=r"\s+",
        header=None,
        usecols=[1, 2, 4],
    )
    data.columns = list(range(3))
    data.iloc[:, 2] = data[2].str.removeprefix("Iris-")
    lb = LabelEncoder()
    lb.fit(data.iloc[:, -1])
    return data, lb


data, label_enc = load_data()


class KNNClassifier:
    def __init__(self, k: int = 3, p: int = 2):
        self.k = k
        self.p = p

    def fit(self, x: np.ndarray, y: np.ndarray, condense: bool = False) -> None:
        self.x_train = np.array(x)
        self.y_train = np.array(y)

    def _condense(self, x: np.ndarray, y: np.ndarray) -> None:
         # 1. find the epsilon
         # divide the 2 classes
         labels = np.unique(y)
         if len(labels) != 2:
             raise ValueError(f"There should be exactly 2 unique classes, while you have {len(labels)} classes")
         
         labels_it = iter(labels)
         blue_pts = x[y == next(labels_it)]
         red_pts = x[y == next(labels_it)]
         epsilon = np.inf
         for bp, rp in product(blue_pts, red_pts):
             epsilon = min(epsilon, self._)

    
    def _distance(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        measures the distance between set points u and one point p
        >>> m = KNNClassifier()
        >>> m._distance([[0, 0]], [3,4]).astype(int)
        array([5])
        >>> m.p = np.inf
        >>> m._distance([[0, 0]], [3,4]).astype(int)
        array([4])
        >>> m.p = 2
        >>> m._distance([[0, 0], [0, 4]], [3,4]).astype(int)
        array([5, 3])
        """
        u = np.atleast_2d(u)
        v = np.atleast_1d(v)
        if self.p == np.inf:
            return np.max(np.abs(u - v), axis=1)
        return np.pow(np.sum(np.pow(np.abs(u - v), self.p), axis=1), 1 / self.p)

    def pred(self, x: pd.DataFrame | np.ndarray) -> pd.Series | np.ndarray:
        x = np.atleast_2d(x)
        labels = [self._pred_one(u) for u in x]
        return np.array(labels)

    def _pred_one(self, x: np.ndarray) -> np.number | np.ndarray | int:
        distances = self._distance(self.x_train, x)
        k_top_idx = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_top_idx]
        votes = Counter(k_nearest_labels)
        majority = max(votes, key=votes.__getitem__)
        return majority

def classify(flowers: list[str], k: int, p: float, seed: int, condense: bool) -> dict:
    knn = KNNClassifier(k, p)
    ndata = data[data.iloc[:, -1].isin(flowers)]
    
    x, y = ndata.drop(columns=ndata.columns[-1]), label_enc.transform(ndata.iloc[:, -1])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=seed, train_size=0.5
    )
    
    knn.fit(x_train, y_train, condense=condense)
    y_pred = knn.pred(x_train)
    empirical_error = mean_error(y_pred, y_train)
    y_pred = knn.pred(x_test)
    true_error = mean_error(y_pred, y_test)
    
    return {"empirical_error": empirical_error.item(), "true_error": true_error.item()}
    
def mean_error(pred: np.ndarray, test: np.ndarray) -> np.number:
    return np.mean(pred != test)

def run_experiment():
    exp = ex.Experiment()
    input_ranges = {
        "flowers": [['versicolor', 'virginica']],
        "k": range(1, 10, 2),
        "p": [1, 2, np.inf],
        "seed": range(1, 101),
        "condense": [False]
    }
    exp.clear_previous_results()
    exp.run(classify, input_ranges)

if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
    run_experiment()
