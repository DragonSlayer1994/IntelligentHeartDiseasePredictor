from django.http import HttpResponse
from django.template import Template, Context
from django.template import loader
from sklearn import tree
import pickle
from array import *
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn import metrics
from django.shortcuts import render
from sklearn.linear_model import LogisticRegression


class systemController:

    def systemController(request):
        return render(request, 'systemController.html')

    def hdpage(request):
        return render(request, 'trainHD.html')

    def chdpage(request):
        return render(request, 'trainCHD.html')

    def trainhd(request):

        import pandas as pd
        import numpy as np
        import pickle

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import make_scorer, accuracy_score
        from sklearn.model_selection import GridSearchCV
        from sklearn.cross_validation import KFold
        from sklearn import metrics

        train_data = pd.read_csv('D:/fypCode/processed_cleveland_data_train.csv')
        test_data = pd.read_csv('D:/fypCode/processed_cleveland_data_test.csv')
        dataset = pd.read_csv('D:/fypCode/cleavelandFomatted.csv')

        def make_prediction_var_binary(df):
            df['num'] = df['num'].replace([1, 2, 3, 4, 5, 6], 1)

        make_prediction_var_binary(dataset)

        X = dataset.loc[:, "age":"thal"]
        y = dataset["num"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier()

        parameters = {'n_estimators': [4, 6, 9],
                      'max_features': ['log2', 'sqrt', 'auto'],
                      'criterion': ['entropy', 'gini'],
                      'max_depth': [2, 3, 5, 10],
                      'min_samples_split': [2, 3, 5],
                      'min_samples_leaf': [1, 5, 8]}

        acc_scorer = make_scorer(accuracy_score)
        grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
        grid_obj = grid_obj.fit(X_train, y_train)
        clf = grid_obj.best_estimator_

        predictions = clf.predict(X_test)
        y_pred = clf.predict(X_test)
        print("printing chances: ", y_pred[:10])
        accu=accuracy_score(y_test,predictions)
        print("acuracy score is ", accu)
        print("printing the confusion matrix of the system ", metrics.classification_report(y_test, y_pred))

        testing = [60, 1, 4, 130, 206, 0, 2, 132, 1, 2.4, 2, 2, 7]
        testing1 = [50, 0, 3, 120, 219, 0, 0, 158, 0, 1.6, 2, 0, 3]
        testing2 = [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0, 3]
        prediction = clf.predict([testing, testing1, testing2])

        print("prediction is ", prediction)

        all_data = pd.read_csv('D:/fypCode/processed_cleveland_data.csv')
        make_prediction_var_binary(all_data)

        X_all = all_data.drop(['num'], axis=1)
        y_all = all_data['num']

        y_all.head()

        def run_kfold(clf):
            kf = KFold(297, n_folds=5)
            outcomes = []
            fold = 0
            for train_index, test_index in kf:
                fold += 1
                X_train, X_test = X_all.values[train_index], X_all.values[test_index]
                y_train, y_test = y_all.values[train_index], y_all.values[test_index]
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                outcomes.append(accuracy)
                print("Fold {0} accuracy: {1}".format(fold, accuracy))
                mean_outcome = np.mean(outcomes)
                print("Mean Accuracy: {0}".format(mean_outcome))

        run_kfold(clf)

        print(metrics.confusion_matrix(y_test, y_pred))
        print(metrics.classification_report(y_test, y_pred))

        metrics.precision_recall_curve(y_test, y_pred)

        return render(request, 'trainHD.html',
                      {'content': ['accuracy score is :', accu], 'c2': ['classifier successfully trained !!', '']})

    def trainchd(request):

        dataset = pd.read_csv("datasets/framingham.csv", na_values="?")

        dataset.dropna(inplace=True, axis=0, how="any")
        X = dataset.loc[:, "male":"glucose"]
        y = dataset["TenYearCHD"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        freqs = pd.DataFrame(
            {"Training dataset": y_train.value_counts().tolist(), "Test dataset": y_test.value_counts().tolist(),
             "Total": y.value_counts().tolist()}, index=["Healthy", "Sick"])
        print(freqs[["Training dataset", "Test dataset", "Total"]])

        model = LogisticRegression()
        model.fit(X_train, y_train)
        test = [1, 68, 1, 0, 0, 0, 0, 1, 0, 176, 168, 97, 23.14, 60, 79]
        test2 = list(map(int, test))
        print('printing test2 ', test2)
        prediction = model.predict([test2])
        print('prediction is ', prediction)
        with open('logisticRegression.pickle', 'wb') as f:
            pickle.dump(model, f)

        print("training set accuracy:", model.score(X_train, y_train))

        print("test set accutacy: ", model.score(X_test, y_test))

        pred_y = model.predict(X_test)

        print(metrics.classification_report(y_test, pred_y))

        predicted1 = model.predict(X_train)
        predicted2 = model.predict(X_test)

        all_data = pd.read_csv("datasets/framingham.csv", na_values="?")
        all_data.dropna(inplace=True, axis=0, how="any")
        X_all = all_data.drop(['TenYearCHD'], axis=1)
        y_all = all_data['TenYearCHD']

        def run_kfold(model):
            kf = KFold(3658, n_folds=6)
            outcomes = []
            fold = 0
            for train_index, test_index in kf:
                fold += 1
                X_train, X_test = X_all.values[train_index], X_all.values[test_index]
                y_train, y_test = y_all.values[train_index], y_all.values[test_index]
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                outcomes.append(accuracy)
                print("Fold {0} accuracy: {1}".format(fold, accuracy))
                mean_outcome = np.mean(outcomes)
                print("Mean Accuracy: {0}".format(mean_outcome))

        run_kfold(model)

        print(metrics.classification_report(y_test, pred_y))
        print(metrics.confusion_matrix(y_test, pred_y))
        accu = metrics.accuracy_score(y_test, pred_y)

        return render(request, 'trainCHD.html',
                      {'content': ['accuracy score is :', accu], 'c2': ['classifier successfully trained !!', '']})
