import numpy as np
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups


class StringKernelSVM(svm.SVC):
    """
    Implementation of string kernel from article
    H. Lodhi, C. Saunders, J. Shawe-Taylor, N. Cristianini, and C. Watkins.
    Text classification using string kernels. Journal of Machine Learning Research, 2, 2002 .
    svm.SVC is a basic class from scikit-learn for SVM classification (in multiclass case, it uses one-vs-one approach)
    """


    def __init__(self, lambda_decay=0.5, subseq_length=6):
        svm.SVC.__init__(self, kernel=self.string_kernel)
        self.lambda_decay = lambda_decay
        self.subseq_length = subseq_length

# def SSK(lamb, p):
#         """Return subsequence kernel"""
#         def SSKernel(xi,xj,lamb,p):
#             mykey = (xi, xj) if xi>xj else (xj, xi)
#             if not mykey in cache:
#                 dps = []
#                 for i in xrange(len(xi)):
#                     dps.append([lamb**2 if xi[i] == xj[j] else 0 for j in xrange(len(xj))])
#                 dp = []
#                 for i in xrange(len(xi)+1):
#                     dp.append([0]*(len(xj)+1))
#                 k = [0]*(p+1)
#                 for l in xrange(2, p + 1):
#                     for i in xrange(len(xi)):
#                         for j in xrange(len(xj)):
#                             dp[i+1][j+1] = dps[i][j] + lamb * dp[i][j+1] + lamb * dp[i+1][j] - lamb**2 * dp[i][j]
#                             if xi[i] == xj[j]:
#                                 dps[i][j] = lamb**2 * dp[i][j]
#                                 k[l] = k[l] + dps[i][j]
#                 cache[mykey] = k[p]
#             return cache[mykey]
#         return lambda xi, xj: SSKernel(xi,xj,lamb,p)/(SSKernel(xi,xi,lamb,p) * SSKernel(xj,xj,lamb,p))**0.5



    def K(self, s, t):
        return


    def K1(self):
        return


    def K2(self):
        return


    def string_kernel(self,X):
        """
        String Kernel computation
        :param X: list of documents (m rows, 1 column); each row is a single document (string)
        """
        gram_matrix = np.zeros((len(X), len(X)), dtype=np.float)
        for i in X:
            for j in X:
                if i == j:
                    gram_matrix[i, j] = 1
                else:
                    gram_matrix[i, j] = self.K(X[i], X[j]) / (self.K(X[i], X[i]) * self.K(X[j], X[j])) ** 0.5

        # numpy array of Gram matrix
        return gram_matrix


if __name__ == '__main__':
    #The dataset is the 20 newsgroups dataset. It will be automatically downloaded, then cached.
    news_train = fetch_20newsgroups(subset='train')
    X_train = news_train.data
    Y_train = news_train.target

    clf = StringKernelSVM()
    clf.fit(X_train, Y_train)
