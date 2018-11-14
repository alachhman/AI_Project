import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC


def clf_example():
    x = [[1], [2]]
    y = ["spam", "not_spam"]
    clf = MultinomialNB()
    clf.fit(x, y)
    print(clf.predict([[4]]))


def main():
    clf_example()


main()




