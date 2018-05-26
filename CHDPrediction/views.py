from django.http import HttpResponse
from django.template import loader

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn import metrics
from django.shortcuts import render


class CoronaryPrediction:

    def CHDPrediction(request):

        sex = request.POST.get('sex')
        age = request.POST.get('age')
        education = request.POST.get('education')
        currentSmoker = request.POST.get('currentSmoker')
        cigsPerDay = request.POST.get('cigsPerDay')
        BPMeds = request.POST.get('bpMeds')
        prevalentStroke = request.POST.get('ps')
        prevalentHyp = request.POST.get('prevalentHyp')
        diabetes = request.POST.get('diabetes')
        totChol = request.POST.get('totChol')
        sysBP = request.POST.get('sysBp')
        diaBP = request.POST.get('diaBp')
        BMI = request.POST.get('BMI')
        heartRate = request.POST.get('heartRate')
        glucose = request.POST.get('glucose')

        test = [sex, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes,
                totChol, sysBP, diaBP, BMI, heartRate, glucose]

        if age is None:
            print("its null")
        else:
            test3 = list(map(int, test))
            print('printing array', test3)

            clf = pickle.load(open('logisticRegression.pickle', 'rb'))

            dataset = pd.read_csv("D:/fyp/framingham.csv", na_values="?")

            dataset.dropna(inplace=True, axis=0, how="any")
            X = dataset.loc[:, "male":"glucose"]
            y = dataset["TenYearCHD"]

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
            pred_y = clf.predict(X_test)
            accuracy1 = []
            accuracy = accuracy_score(y_test, pred_y)
            print('accuracy score is ', accuracy)
            accuracy1.append(accuracy)
            prediction = clf.predict([test3])
            print('prediction is ',prediction)
            loader.get_template('chdTemp.html')
            template = loader.get_template('prediction1.html')
            context = {
                'allUsers': prediction,
                'accuracy': accuracy1,
            }

            return HttpResponse(template.render(context, request))
        template = loader.get_template('chdTemp.html')

        context = {

        }
        return HttpResponse(template.render(context, request))

    def train(request):

        dataset = pd.read_csv("D:/fyp/framingham.csv", na_values="?")

        dataset.dropna(inplace=True, axis=0, how="any")
        X = dataset.loc[:, "male":"glucose"]
        y = dataset["TenYearCHD"]

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        freqs = pd.DataFrame(
            {"Training dataset": y_train.value_counts().tolist(), "Test dataset": y_test.value_counts().tolist(),
             "Total": y.value_counts().tolist()}, index=["Healthy", "Sick"])
        print(freqs[["Training dataset", "Test dataset", "Total"]])

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        model.fit(X_train, y_train)

        with open('logisticRegression.pickle', 'wb') as f:
            pickle.dump(model, f)

        pickle_in = open('logisticRegression.pickle', 'rb')
        model = pickle.load(pickle_in)

        print("training set accuracy:", model.score(X_train, y_train))

        print("test set accutacy: ", model.score(X_test, y_test))

        pred_y = model.predict(X_test)

        print(metrics.classification_report(y_test, pred_y))

        predicted1 = model.predict(X_train)
        predicted2 = model.predict(X_test)

        all_data = pd.read_csv("D:/fyp/framingham.csv", na_values="?")
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
        print(metrics.roc_curve(y_test, pred_y))
