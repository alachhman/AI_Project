import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC


def clf_example():
    x = np.array([[-1, -1], [-2, -1], [-1, -2], [-2, -2], [1, 1], [2, 1], [2, 2], [1, 2]])
    y = np.array(["spam", "spam", "spam", "spam", "not_spam", "not_spam", "not_spam", "not_spam"])
    clf = SVC(gamma='auto')
    clf.fit(x, y)
    print(clf.predict([[0.5, 0.5]]))


def main():
    clf_example()


main()



