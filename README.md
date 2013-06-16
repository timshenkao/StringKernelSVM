My implementation of string kernel from article
    H. Lodhi, C. Saunders, J. Shawe-Taylor, N. Cristianini, and C. Watkins.
    Text classification using string kernels. Journal of Machine Learning Research, 2, 2002 .

As far as I know, there is no implementation of this algorithm in scikit-learn.

Also you can look at http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#libsvm_for_string_data for libsvm implementation
of string kernel (with no warranty).

svm.SVC is a basic class from scikit-learn for SVM classification. It uses one-vs-one approach in multiclass case.

FILES:
stringSVM.py - basic algorithm from the article
stringSVM_optimized_K2.py - algorithm with K''() from the article