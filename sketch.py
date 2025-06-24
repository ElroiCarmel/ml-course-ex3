import logging
from typing import Optional
import experiments_csv as ex
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.Logger(__name__)


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

    def fit(
        self, x: np.ndarray, y: np.ndarray, condense: bool = False
    ) -> Optional[int]:
        if not condense:
            self.x_train = np.array(x)
            self.y_train = np.array(y)
        else:
            logger.debug(
                "Condensing the train set. Initial size: %s points.", x.shape[0]
            )
            return self._condense(x, y)

    def _condense(self, x: np.ndarray, y: np.ndarray) -> int:
        # 1. find the epsilon
        # divide the 2 classes
        logger.debug("Shapes: x -> %s, y -> %s", x.shape, y.shape)
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError(
                f"There should be exactly 2 unique classes, while you have {len(labels)} classes."
            )

        labels_it = iter(labels)
        blue_pts = x[y == next(labels_it)]
        logger.debug("Size of blue points set: %d.", blue_pts.shape[0])
        red_pts = x[y == next(labels_it)]
        logger.debug("Size of red points set: %d.", red_pts.shape[0])
        epsilon = np.inf
        for bp, rp in product(blue_pts, red_pts):
            epsilon = min(epsilon, self.distance(bp, rp, self.p))

        logger.debug("Minimum distance from red to blue points: %s.", epsilon.item())
        # Greedy construction
        T_x = [x[0]]
        T_y = [y[0]]
        for i in range(x.shape[0]):
            p, label = x[i], y[i]
            d2T = self.distance(T_x, p)
            if d2T.min() > epsilon:
                logger.debug("Point %s is added to the condensed set.", p.tolist())
                T_x.append(p)
                T_y.append(label)
        self.x_train = np.array(T_x)
        self.y_train = np.array(T_y)
        condesned_size = self.x_train.shape[0]
        logger.debug("Final condensed set size: %d.", condesned_size)
        return condesned_size

    @staticmethod
    def distance(u: np.ndarray, v: np.ndarray, p: float = 2) -> np.ndarray:
        """
        measures the distance between set points u and one point p
        >>> KNNClassifier.distance([[0, 0]], [3,4]).astype(int)
        array([5])
        >>> KNNClassifier.distance([[0, 0]], [3,4], np.inf).astype(int)
        array([4])
        >>> KNNClassifier.distance([[0, 0], [0, 4]], [3,4]).astype(int)
        array([5, 3])
        """
        u = np.atleast_2d(u)
        v = np.atleast_1d(v)
        if p == np.inf:
            return np.max(np.abs(u - v), axis=1)
        return np.pow(np.sum(np.pow(np.abs(u - v), p), axis=1), 1 / p)

    def pred(self, x: pd.DataFrame | np.ndarray) -> pd.Series | np.ndarray:
        x = np.atleast_2d(x)
        labels = [self._pred_one(u) for u in x]
        return np.array(labels)

    def _pred_one(self, x: np.ndarray) -> np.number | np.ndarray | int:
        distances = self.distance(self.x_train, x, self.p)
        k_top_idx = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_top_idx]
        votes = Counter(k_nearest_labels.tolist())
        logger.debug("For point %s the knn votes is %s.", x.tolist(), votes)
        majority = max(votes, key=votes.__getitem__)
        logger.debug(
            "Flower is predicted to be %s.", label_enc.inverse_transform([majority])
        )
        return majority


def classify(flowers: list[str], k: int, p: float, seed: int, condense: bool) -> dict:
    knn = KNNClassifier(k, p)
    ndata = data[data.iloc[:, -1].isin(flowers)]
    logger.debug("Initiating experiment. Flowers participating: %s", flowers)
    logger.debug(
        "Parameters of experiment: k=%s, p=%s, seed=%s, condense=%s",
        k,
        p,
        seed,
        condense,
    )
    x, y = ndata.drop(columns=ndata.columns[-1]).values, label_enc.transform(
        ndata.iloc[:, -1]
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=seed, train_size=0.5
    )

    condensed_size = knn.fit(x_train, y_train, condense=condense)
    y_pred = knn.pred(x_train)
    empirical_error = mean_error(y_pred, y_train)
    y_pred = knn.pred(x_test)
    true_error = mean_error(y_pred, y_test)

    return {
        "empirical_error": empirical_error.item(),
        "true_error": true_error.item(),
        "train_init_size": x_train.shape[0],
        "condensed_size": condensed_size,
    }


def mean_error(pred: np.ndarray, test: np.ndarray) -> np.number:
    return np.mean(pred != test)


def run_experiment():
    exp = ex.Experiment()
    input_ranges = {
        "flowers": [["versicolor", "virginica"]],
        "k": range(1, 10, 2),
        "p": [1, 2, np.inf],
        "seed": range(1, 101),
        "condense": [False],
    }
    exp.clear_previous_results()
    exp.run(classify, input_ranges)


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
    # run_experiment()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    params = {
        "flowers": ["versicolor", "virginica"],
        "k": 3,
        "p": 2,
        "seed": 34,
        "condense": True,
    }
    res = classify(**params)
    print(f"res: {res}")
