import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据训练数据集X获得数据的均值和方差"""
        assert X.ndim == 2, "The dimension of X must be 2"

        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):
        """将X根据这个StandardScale中的mean_,scale_进行归一化"""
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None,\
            "must fit before transform"
        assert X.shape[1] == len(self.mean_), \
            "the feature numbers of X must be equal to mean_ and scale_"

        resX = np.empty(shape = X.shape, dtype = float)

        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        """上方for循环与下面的语句等价"""
        #resX = np.array([((X[:, i] - self.mean_[i]) / self.scale_) for i in range(X.shape[1])])

        return resX
