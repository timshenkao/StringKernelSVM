import numpy as np
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import libsvm


class StringKernelSVM(svm.SVC):
    """
    Implementation of string kernel from article:
    H. Lodhi, C. Saunders, J. Shawe-Taylor, N. Cristianini, and C. Watkins.
    Text classification using string kernels. Journal of Machine Learning Research, 2, 2002 .
    svm.SVC is a basic class from scikit-learn for SVM classification (in multiclass case, it uses one-vs-one approach)
    """


    def __init__(self, lambda_decay=0.5, subseq_length=6):
        """
        constructor
        :param lambda_decay: lambda parameter for the algorithm
        :type  lambda_decay: float
        :param subseq_length: maximal subsequence length
        :type subseq_length: int
        """

        #svm.SVC.__init__(self, kernel=self.string_kernel)
        self.lambda_decay = lambda_decay
        self.subseq_length = subseq_length
        self.sim_docs_kernel_value = {}
        svm.SVC.__init__(self, kernel='precomputed')


    def K(self, n, s, t):
        """
        K_n(s,t) in the original article; recursive function
        :param n: length of subsequence
        :type n: int
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :return: float value for similarity between s and t
        """
        if min(len(s),len(t)) < n:
            return 0
        else:
            part_sum = 0
            for j in range(1,len(t)):
                if t[j] == s[-1]:
                    #not t[:j-1] as in the article but t[:j] because of Python slicing rules!!!
                    part_sum += self.K1(n-1, s[:-1], t[:j])
            result = self.K(n, s[:-1], t) + self.lambda_decay**2 * part_sum
            return result


    def K1(self, n, s, t):
        """
        K'_n(s,t) in the original article; auxiliary intermediate function; recursive function
        :param n: length of subsequence
        :type n: int
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :return: intermediate float value
        """
        if n == 0:
            return 1
        elif min(len(s),len(t)) < n:
            return 0
        else:
            part_sum = 0
            for j in range(1,len(t)):
                if t[j] == s[-1]:
                    #not t[:j-1] as in the article but t[:j] because of Python slicing rules!!!
                    part_sum += self.K1(n-1, s[:-1], t[:j]) * (self.lambda_decay ** (len(t) - (j + 1) + 2))
            result = self.lambda_decay * self.K1(n, s[:-1], t) + part_sum
            return result


    def string_kernel(self,X):
        """
        String Kernel computation
        :param X: list of documents (m rows, 1 column); each row is a single document (string)
        :type X: list
        :return: Gram matrix for the given parameters
        """
        # numpy array of Gram matrix
        gram_matrix = np.zeros((len(X), len(X)), dtype=np.float32)
        #store K(s,s) values in dictionary to avoid recalculations
        for i in range(len(X)):
            self.sim_docs_kernel_value[i] = self.K(self.subseq_length, X[i], X[i])
        print self.sim_docs_kernel_value
        #calculate Gram matrix
        for i in range(len(X)):
            for j in range(len(X)):
                if i == j:
                    gram_matrix[i, j] = 1
                else:
                    gram_matrix[i, j] = self.K(self.subseq_length, X[i], X[j]) / \
                                        (self.sim_docs_kernel_value[i] * self.sim_docs_kernel_value[j]) ** 0.5
        return gram_matrix


    def fit(self, X, Y):
        gram_matr = self.string_kernel(X)
        self.gram_matrix = gram_matr
        super(svm.SVC, self).fit(self.gram_matrix, Y)


    def predict(self, X):
        return libsvm.predict(
            self.string_kernel(X), self.support_, self.support_vectors_, self.n_support_,
            self.dual_coef_, self._intercept_,
            self._label, self.probA_, self.probB_,
            svm_type=self.svm_type,
            kernel=self.kernel, C=self.C, nu=self.nu,
            probability=self.probability, degree=self.degree,
            shrinking=self.shrinking, tol=self.tol, cache_size=self.cache_size,
            coef0=self.coef0, gamma=self._gamma, epsilon=self.epsilon)


if __name__ == '__main__':
    #The dataset is the 20 newsgroups dataset. It will be automatically downloaded, then cached.
    # news_train = fetch_20newsgroups(subset='train')
    # X_train = news_train.data
    # Y_train = news_train.target

    X_train = ['card', 'cat', 'bar', 'bat']
    Y_train = np.array([1, 1, 0, 0])


    clf = StringKernelSVM(subseq_length=2)
    clf.fit(X_train, Y_train)
    clf.predict(X_train)
