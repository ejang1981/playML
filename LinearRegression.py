import numpy as np
from .metrics import r2_score


class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None   ##系数
        self.interception_ = None  ##截距
        self._theta = None  ##截距+系数

    def fit_normal(self, X_train, y_train):
        """正规计算，根据训练数据集X_train，y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones([len(X_train), 1]), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果问题"""
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones([len(X_predict), 1]), X_predict])
        y_predict = X_b.dot(self._theta)

        return y_predict

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def fit_gd(self, X_train, y_train, eta = 0.01, n_iters = 1e4, epsilon=1e-8):
        """梯度计算，根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        y = y_train

        def J(theta, X_b, y):  ##目标函数，损失函数
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:  ##异常检测
                return float('inf')  ##返回一个最大的float值，代表其达到了最大

        def dJ(theta, X_b, y):   ##目标函数求导值
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            #
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            #
            # return res * 2 / len(X_b)

            #以下为向量计算的运算过程，上述注释代码为不用向量计算的过程
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)



        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):  ###迭代次数10000次

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])  ##初始theta全0，且全0的列数等于X_b的列数
        self._theta = gradient_descent(X_b, y, initial_theta, eta, n_iters, epsilon)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_sgd(self, X_train, y_train, n_iter=5, t0=5, t1=50):
        """随机梯度计算，根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iter >= 1  ##至少看一圈，至少所有样本看一次

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iter):

            def learning_rate(t):  ##计算学习率
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)

            for cur_iter in range(n_iter):      ####把所有样本整体看n_iters圈，有可能有样本永远也看不到
                # ##为了保证所有样本都看到，而且是随机看到，可将样本乱序排序后，从头到尾看一遍
                indexes = np.random.permutation(m)  ##得到样本数量的乱序排列索引值
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter*m + i) * gradient

                # 在Jupyter中的代码，此时不用，此时是为了所有样本都看到至少一次，所以用了上述代码
                # rand_i = np.random.randint(m)
                # gradient = dJ_sgd(theta, X_b[rand_i], y[rand_i])
                # theta = theta - learning_rate(cur_iter) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iter)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def __repr__(self):
        return "LinearRegression()"

