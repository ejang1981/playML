import numpy as np

class PCA:

    def __init__(self, n_components = 1):
        """初始化PCA主成分分析"""
        assert n_components >= 1, "n_components must be valid"

        self.n_components = n_components   ###要有几个主成分，由用户传进
        self.components_ = None   ###每一个主成分是什么

    def fit(self, X, eta = 0.01, n_iters = 1e4):
        """获得数据集X的前n个主成分"""
        assert self.n_components <= X.shape[1], \
            "n_components must not be greater than the feature number of X"

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta = 0.01, n_iters=1e4, epsilon=1e-8):

            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break

                cur_iter += 1

            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta, n_iters)
            self.components_[i,:] = w   ###最终得到的主成分

            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        """高维到低维，将给定的X映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]   ##等于主成分列数

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """低维到高维，将给定的X反向映射回原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]   ##等于主成分行数

        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components = %d)" % self.n_components


