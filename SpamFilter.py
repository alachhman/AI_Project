from sklearn import svm


def clf_example():
    x = [[1], [2]]
    y = ["spam", "not_spam"]
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    clf.fit(x, y)
    print(clf.support_vectors_)
    print(clf.predict([[4]]))
    #dec = clf.decision_function([[1]])
    #dec.shape[1]
    #clf.decision_function_shape = "ovr"
    #dec = clf.decision_function([[1]])
    #print(dec.shape[1])


def main():
    clf_example()


main()




