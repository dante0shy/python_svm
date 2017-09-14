import random
import numpy as np
from numpy import *
import sklearn.svm
import random
from scipy.optimize import minimize_scalar

class SVMClassifier(object):
    def __init__(self, max_itr=1000, lr=0.01, C=1, threhold=0.5):
        self.max_iter = max_itr
        self.w = np.array([])
        self.alapha = np.array([])
        self.b = 0  # random.uniform(-0.001,0.001)
        self.lr = lr
        self.C = C
        self.threhold = threhold
        self.acc = []

    def error(self, x, y):
        output_k = self.predict(x)
        error_k = output_k == y
        return float(len(np.nonzero(error_k)[0])) / float(y.shape[0])

    def predict(self, x):
        output_k = np.array(map(lambda a: 1 if a + self.b > 0 else 0, np.dot(x, self.w.T).tolist()))  # + self.b
        return output_k

    # def select

    def fit(self, x, y):
        if (not x.shape[0] == y.shape[0]) or x.shape == 0:
            raise ValueError("input error!")
        self.w = np.zeros(x.shape[1])
        for idx, _ in enumerate(self.w):
            self.w[idx] = random.uniform(-1, 1)
        alapha_change = 0
        iterNum = 0
        accuracy = self.error(x, y)
        accuracy_cahange = 0
        print "iter {iter}: alapha_change {alapha_change}, accuracy {accuracy}".format(
            iter=iterNum,
            alapha_change=alapha_change,
            accuracy=accuracy
        )
        self.acc.append(accuracy)
        while (iterNum < self.max_iter):
            print "E :" + str(self.w)
            print "E :" + str(self.b)
            tmp = []
            tmp2 = 0.0
            for i, yi in enumerate(y):
                if yi == 0:
                    yi = -1
                if i == 0:
                    if yi * (np.dot(self.w, x[i].T) + self.b) < 1:
                        tmp = self.C * x[i]
                        tmp2 = self.C * yi
                    else:
                        tmp = 0
                        tmp2 = 0
                    continue
                if yi * (np.dot(self.w, x[i].T) + self.b) < 1:
                    tmp = tmp + self.C * yi * x[i]  # np.array(map(lambda a : a*yi,x[i]))
                    tmp2 = tmp2 + self.C * yi

            grad = self.w - tmp
            tmp2 = self.b - tmp2  # /y.shape[0]
            self.w = self.w - self.lr * grad / y.shape[0]
            self.b = self.b - self.lr * tmp2 / y.shape[0]

            old_accuracy = accuracy
            accuracy = self.error(x, y)
            accuracy_cahange = accuracy - old_accuracy
            iterNum = iterNum + 1
            print "iter {iter}: alapha_change {alapha_change}, accuracy {accuracy}".format(
                iter=iterNum,
                alapha_change=alapha_change,
                accuracy=accuracy
            )
            self.acc.append(accuracy)
            # print self.w
            # print self.b
        # self.plot()
        print "E :" + str(self.w)
        print "E :" + str(self.b)

    def plot(self):
        import matplotlib.pylab as pl
        pl.plot(1 * range(len(self.acc)), self.acc)

        pl.show()


class SVMClassifierES(object):
    def __init__(self, max_itr=500, lr=0.01, C=1, threhold=0.5):
        self.max_iter = max_itr
        self.w = np.array([])
        self.alapha = np.array([])
        self.b = 0  # random.uniform(-0.001,0.001)
        self.lr = lr
        self.C = C
        self.threhold = threhold
        self.acc = []

    def error(self, x, y):
        output_k = self.predict(x)
        error_k = output_k == y
        return float(len(np.nonzero(error_k)[0])) / float(y.shape[0])

    def predict(self, x):
        output_k = np.array(map(lambda a: 1 if a + self.b > 0 else -1, np.dot(x, self.w.T).tolist()))  # + self.b
        return output_k

    # def select
    def get_e(self, i, j, x, y):
        tmp = 0.0
        tmp2 = 0.0
        tmp3 = 0.0
        for i0, yi in enumerate(y):
            if not i0 == i and not i0 == j:
                tmp = tmp + self.alapha[i0] * yi
                tmp2 = tmp2 + (self.alapha[i0] * yi * x[i0, 0]) ** 2
                tmp3 = tmp3 + (self.alapha[i0] * yi * x[i0, 1]) ** 2
        return -tmp, tmp2, tmp3

    def fit(self, x, y):
        if (not x.shape[0] == y.shape[0]) or x.shape == 0:
            raise ValueError("input error!")
        self.alapha = np.zeros(x.shape[0])
        y = np.array([yi if yi == 1 else -1 for yi in y])
        self.w = np.zeros(x.shape[1])
        self.b = 0

        iterNum = 0
        accuracy = self.error(x, y)
        self.acc.append(accuracy)
        kernel = np.dot(x, x.T)
        while (iterNum < self.max_iter):
            numy = random.sample(range(0, len(y)), 1)[0]
            # for numy, yi in enumerate(y):
            # yi = 1 if yi>0 else -1
            alapha = self.alapha

            t1 = np.sum(alapha)-alapha[numy]
            def funcionL(alaphaN):
                t2 = 0.0
                for i0, yi0 in enumerate(y):
                    for j0, yj0 in enumerate(y):
                        t2 = t2 + y[i0] * y[j0] * kernel[i0, j0] * \
                                  (alapha[i0] if not i0 == numy else alaphaN) * (
                                  alapha[j0] if not j0 == numy else alaphaN)
                return -1*(t1 + alaphaN + (t2) * 0.5)

            opt = minimize_scalar(funcionL, self.alapha[numy], method='Bounded', bounds=[0, self.C])
            self.alapha[numy] = opt.x
            # qweq = (self.alapha * y * x.T).T
            self.w = reduce(lambda a, b: a + b, (
                        self.alapha * y * x.T).T)  # np.array([self.alapha[i] * (y[i] if y[i]==1 else -1) * xi for i, xi in enumerate(x)]))
            tmp = dot(self.w, x.T)
            self.b = (max([tmp[i] for i, yi in enumerate(y) if yi == -1]) + min(
                [tmp[i] for i, yi in enumerate(y) if yi == 1])) / 2
            accuracy = self.error(x, y)
            self.acc.append(accuracy)
            print iterNum, 'acc', accuracy
            iterNum = iterNum + 1

        # self.alapha=opt.x

        self.w = reduce(lambda a, b: a + b, (self.alapha * y * x.T).T)
        tmp = dot(self.w, x.T)
        self.b = (max([tmp[i] for i, yi in enumerate(y) if yi == -1]) + min(
            [tmp[i] for i, yi in enumerate(y) if yi == 1])) / 2
        old_accuracy = accuracy
        accuracy = self.error(x, y)
        self.acc.append(accuracy)

        print "E :" + str(self.w)
        print "E :" + str(self.b)
        self.plot()
        print self.alapha

    def plot(self):
        import matplotlib.pylab as pl
        pl.plot(1 * range(len(self.acc)), self.acc)

        pl.show()


if __name__ == '__main__':
    x,y = [], []
    for k in range(300):
        x.append(np.array([10*random.gauss(0.4, 3), 10*random.gauss(0.5, 3)]))
    y = [-1 if e[0]+e[1]>0+random.gauss(0, 0.3) else 1 for e in x ]
    y = np.array(y)
    print np.count_nonzero(y<0)
    print np.count_nonzero(y>0)
    # c = SVMClassifier(C=1.2)
    # c.fit(np.array(x),np.array(y))
    # print 'way 1 finish'
    c2 = SVMClassifierES(C=1)
    c2.fit(np.array(x), np.array(y))
    print c2.alapha
    print 'way 2 finish'
    from IPython import embed

    embed()
