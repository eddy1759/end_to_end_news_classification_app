from sklearn.linear_model import LogisticRegression


def run_model_training(X_train,X_test,y_train,y_test):
    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    score1 = clf.score(X_train,y_train)
    score2 = clf.score(X_test,y_test)
    print(f"The accuracy of the model is as follow:\nTraining set: {round(score1),2}\nTest set: {round(score2),2}")

    return clf