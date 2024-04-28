# train_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

def read_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    data.dropna(inplace=True)

    label_encoder = LabelEncoder()
    T_vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=5000)

    X = data['Email Text'].values
    y = data['Email Type'].values

    for i in range(len(y)):
        if y[i] == 'Phishing Email':
            y[i] = 1
        else:
            y[i] = 0

    y = label_encoder.fit_transform(y)
    return X, y

def print_report(y_val, y_pred, fold):
    print(f'Fold: {fold}')
    print(f'Accuracy Score: {accuracy_score(y_val, y_pred)}')
    print(f'Confusion Matrix: \n {confusion_matrix(y_val, y_pred)}')
    print(f'Classification Report: \n {classification_report(y_val, y_pred)}')


df= pd.read_csv("./Phishing_Email.csv")

Safe_Email = df[df["Email Type"]== "Safe Email"]
Phishing_Email = df[df["Email Type"]== "Phishing Email"]
Data= pd.concat([Safe_Email, Phishing_Email], ignore_index = True)

X = Data["Email Text"].values
y = Data["Email Type"].values
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

data = pd.read_csv('./Phishing_Email.csv')

X, y = read_data('./Phishing_Email.csv')
num_folds = 2
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

fold = 1

for train_index, val_index in kfold.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 3), max_features=10000)
    vectorizer.fit(X_train)

    X_train = vectorizer.transform(X_train)
    X_val = vectorizer.transform(X_val)

    model = XGBClassifier(n_estimators=800, learning_rate=0.1, max_depth=4, colsample_bytree=0.2, n_jobs=-1,
                          random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print_report(y_val, y_pred, fold)
    fold += 1

# save the model and vectorizer
joblib.dump(model, 'phishingemaildetection_model.pkl')
joblib.dump(vectorizer, 'phishingemaildetection_vectorizer.pkl')