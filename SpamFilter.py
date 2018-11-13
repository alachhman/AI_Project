from sklearn import svm


def clf_example():

    x = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC(gamma='scale')
    clf.fit(x, y)
    clf.predict([[2., 2.]])
