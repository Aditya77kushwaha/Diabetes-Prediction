from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def index(request):
    return render(request, "index.html")

def home(request):
    data = -1
    if request.method == "GET":
        df = pd.read_csv(
            r'C:/Users/The Great Aditya/ml-datascience/mlpractice/Datasets/diabetes.csv')
        y = df['Outcome'].values
        X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
        model = LogisticRegression()
        model.fit(X, y, 1000)

        var1 = float(request.GET['n1'])
        var2 = float(request.GET['n2'])
        var3 = float(request.GET['n3'])
        var4 = float(request.GET['n4'])
        var5 = float(request.GET['n5'])
        var6 = float(request.GET['n6'])
        var7 = float(request.GET['n7'])
        var8 = float(request.GET['n8'])
        data = model.predict(
            np.array([var1, var2, var3, var4, var5, var6, var7, var8]).reshape(1, -1))
        if data == 1:
            data = "Diabetic"
        else :
            data = "Non Diabetic"

    return render(request, "home.html", {'data': data})
