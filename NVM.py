# DATING APP MATCH OUTCOME PREDICTION PROJECT
# K-NEAREST NEIGHBORS (KNN) VERSION

# IMPORT LIBRARIES

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# LOAD DATASET

df = pd.read_csv('dating_app_behavior_dataset.csv')

print("FIRST 5 ROWS")
print(df.head())


# CHECK MISSING VALUES

print("\nMISSING VALUES")
print(df.isnull().sum())


# REMOVE MISSING VALUES

df = df.dropna()


# FEATURE ENGINEERING

# Engagement score
df['engagement_score'] = (
    df['likes_received']
    + df['mutual_matches']
    + df['message_sent_count']
)

# Profile quality score
df['profile_quality'] = (
    df['profile_pics_count']
    + df['bio_length']
)


# DEFINE TARGET

y = df['match_outcome']


# FEATURE SELECTION

# Remove weak / redundant columns
X = df.drop([
    'match_outcome',
    'app_usage_time_label',
    'swipe_right_label',
    'interest_tags'
], axis=1)


# ONE HOT ENCODING

X = pd.get_dummies(X)


# ENCODE TARGET LABEL

le = LabelEncoder()

y = le.fit_transform(y)


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# FEATURE SCALING

# VERY IMPORTANT FOR KNN

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# CREATE KNN MODEL

knn = KNeighborsClassifier(
    n_neighbors=15,
    weights='distance',
    metric='minkowski'
)


# TRAIN MODEL

knn.fit(X_train, y_train)


# MAKE PREDICTIONS

y_pred = knn.predict(X_test)


# EVALUATE MODEL

print("\nKNN ACCURACY")
print(accuracy_score(y_test, y_pred))


print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))


print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))


# TEST DIFFERENT K VALUES

print("\nTESTING DIFFERENT K VALUES")

for k in range(1, 21):

    temp_knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance'
    )

    temp_knn.fit(X_train, y_train)

    temp_pred = temp_knn.predict(X_test)

    accuracy = accuracy_score(y_test, temp_pred)

    print(f"K = {k} | Accuracy = {accuracy}")


# OPTIONAL VISUALIZATION

k_values = []
accuracies = []

for k in range(1, 21):

    temp_knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance'
    )

    temp_knn.fit(X_train, y_train)

    temp_pred = temp_knn.predict(X_test)

    accuracy = accuracy_score(y_test, temp_pred)

    k_values.append(k)
    accuracies.append(accuracy)


plt.figure(figsize=(10,6))

plt.plot(k_values, accuracies, marker='o')

plt.title("KNN Accuracy for Different K Values")

plt.xlabel("K Value")
plt.ylabel("Accuracy")

plt.grid(True)

plt.show()