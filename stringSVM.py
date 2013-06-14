import numpy as np
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups


class StringKernelSVM(svm.SVC):
    """
    Implementation of string kernel from article
    H. Lodhi, C. Saunders, J. Shawe-Taylor, N. Cristianini, and C. Watkins.
    Text classification using string kernels. Journal of Machine Learning Research, 2, 2002 .
    svm.SVC is basic class from scikit-learn for SVM classification (in multiclass case, it uses one-vs-one approach)
    """

    def __init__(self, lambda_decay=0.5, subseq_length=6):
        svm.SVC.__init__(self, kernel=self.string_kernel)
        self.lambda_decay = lambda_decay
        self.subseq_length = subseq_length


    def string_kernel(X):
        """
        String Kernel computation
        :param X: matrix of documents (m rows, 1 column); each row is a single document (string)
        """

        return


if __name__ == '__main__':
    news_train = fetch_20newsgroups(subset='train')
    X = news_train.data
    Y = news_train.target

    clf = StringKernelSVM()
    clf.fit(X, Y)
