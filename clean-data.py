import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.svm import SVC


dataFile = "data/adult.data"
testFile = "data/adult.test"
headers = ["age","workclass","fnlwgt","education","education-num",	"marital-status",	"occupation",	"relationship",	"race",	"sex",	"capital-gain",	"capital-loss",	"hours-per-week",	"native-country",	"income" ]
continuous_features = [ "age", "fnlwgt",  "education-num", "capital-gain", "capital-loss", "hours-per-week" ]
categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "income"]
countries = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]


def cleanData(fileName: str):
    df = pd.read_csv(fileName, names=headers, na_values=['?'], skipinitialspace=True)
    df.dropna(inplace=True)
    df.drop(columns=continuous_features, axis=1, inplace=True)

    # Remove all periods from the values
    for column in categorical_features:
        df[column] = df[column].str.rstrip('.')

    X = df.drop(columns=["income"], axis=1)
    Y = df["income"]
    X = pd.get_dummies(X, drop_first=True)

    # Add missing columns
    for category in countries:
        cat_col = "native-country_" + category
        if cat_col not in X.columns:
            X[cat_col] = 0

    X = X.sort_index(axis=1)
    return X, Y



trainX, trainY = cleanData(dataFile)
testX, testY = cleanData(testFile)

def getStats(model: DecisionTreeClassifier | CategoricalNB):
    model.fit(trainX, trainY)

    testY = trainY
    testX = trainX

    predictions = model.predict(testX);
    report = classification_report(testY, predictions)

    print(report)

    cm = confusion_matrix(y_true=testY, y_pred=predictions)

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (FP + FN + TP)

    TPR = TP / (TP + FN)  # Recall
    FPR = FP / (FP + TN)

    for i, class_label in enumerate([0,1]):
        print(f"Class {class_label}: TP Rate: {TPR[i]:.2f}, FP Rate: {FPR[i]:.2f}")


## Decision Tree
model = DecisionTreeClassifier(criterion="entropy")
print("\n\n\n Decision Tree:")
getStats(model)


## Naive Bayes
model = CategoricalNB()
print("\n\n\n Naive Bayes:")
getStats(model)



### ============ Part 2 ============

def clean2(fileName: str):
    df = pd.read_csv(fileName, names=headers, na_values=['?'], skipinitialspace=True)
    df.dropna(inplace=True)

    # Remove all periods from the values
    for column in categorical_features:
        df[column] = df[column].str.rstrip('.')

    # for each continious column, map it to if it is greater than the mean of that column
    for column in continuous_features:
        df[column] = df[column].astype(int)
        mean = df[column].mean()
        df[column] = df[column].map(lambda x: 1 if x > mean else 0)


    X = pd.get_dummies(df, drop_first=True)

    # Add missing columns
    for category in countries:
        cat_col = "native-country_" + category
        if cat_col not in X.columns:
            X[cat_col] = 0

    X = X.sort_index(axis=1)
    return X

trainDf = clean2(dataFile)
testDf = clean2(testFile)

k_vals = [3,5,10]

for k in k_vals:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(trainDf)
    print(f"KMeans with k={k}", clusters)


# ============ Part 3 =============
def clean3(fileName: str):
    df = pd.read_csv(fileName, names=headers, na_values=['?'], skipinitialspace=True)
    df.dropna(inplace=True)

    # Remove all periods from the values
    for column in categorical_features:
        df[column] = df[column].str.rstrip('.')

    # for each continious column, map it to if it is greater than the mean of that column
    for column in continuous_features:
        df[column] = df[column].astype(int)
        mean = df[column].mean()
        df[column] = df[column].map(lambda x: 1 if x > mean else 0)

    X = df.drop(columns=["income"], axis=1)
    Y = df["income"]
    X = pd.get_dummies(X, drop_first=True)

    # Add missing columns
    for category in countries:
        cat_col = "native-country_" + category
        if cat_col not in X.columns:
            X[cat_col] = 0

    X = X.sort_index(axis=1)
    return X, Y

trainX, trainY = clean3(dataFile)
testX, testY = clean3(testFile)

# Save trainX to a file
trainX.to_csv("trainX.csv", index=False)

model = SVC(kernel="linear")

print("Fitting SVM")

model.fit(trainX, trainY)

print("Fit SVM")

predictions = model.predict(testX);
report = classification_report(testY, predictions)

print("\n\n\nSVM: ")

print(report)


