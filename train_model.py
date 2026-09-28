import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("data/Titanic-Dataset.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# =========================================================
# 2. BASIC DATA ANALYSIS
# =========================================================

print("\nSurvival distribution:")
print(df["Survived"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Survived")
plt.title("Survival Distribution")
plt.savefig("survival_distribution.png")
plt.close()


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.savefig("survival_by_gender.png")
plt.close()


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.savefig("survival_by_class.png")
plt.close()


# =========================================================
# 3. FEATURE SELECTION
# =========================================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = df[features]
y = df["Survived"]


# =========================================================
# 4. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================================================
# 5. PREPROCESSING
# =========================================================

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]


numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# =========================================================
# 6. MACHINE LEARNING MODELS
# =========================================================

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])


decision_tree = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ))
])


random_forest = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])


gradient_boosting = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(
        random_state=42
    ))
])


xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ))
])


ml_models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Gradient Boosting": gradient_boosting,
    "XGBoost": xgb_model
}


# =========================================================
# 7. TRAIN ML MODELS
# =========================================================

ml_results = []


for name, model in ml_models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    ml_results.append(results)


ml_results_df = pd.DataFrame(ml_results)

print("\nMachine Learning Results:")
print(ml_results_df)


# =========================================================
# 8. DEEP LEARNING PREPROCESSING
# =========================================================

dl_numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


dl_categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


dl_preprocessor = ColumnTransformer([
    ("num", dl_numeric, numeric_features),
    ("cat", dl_categorical, categorical_features)
])


X_train_dl = dl_preprocessor.fit_transform(X_train)
X_test_dl = dl_preprocessor.transform(X_test)


print("\nDeep Learning input shape:")
print(X_train_dl.shape)


# =========================================================
# 9. ANN MODEL
# =========================================================

ann_model = Sequential([
    Dense(
        64,
        activation="relu",
        input_shape=(X_train_dl.shape[1],)
    ),
    Dropout(0.30),

    Dense(32, activation="relu"),

    Dropout(0.20),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")
])


ann_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


ann_model.fit(
    X_train_dl,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    verbose=1
)


ann_prob = ann_model.predict(X_test_dl).ravel()

ann_pred = (ann_prob >= 0.5).astype(int)


ann_results = {
    "Model": "ANN",
    "Accuracy": accuracy_score(y_test, ann_pred),
    "Precision": precision_score(y_test, ann_pred),
    "Recall": recall_score(y_test, ann_pred),
    "F1 Score": f1_score(y_test, ann_pred),
    "ROC-AUC": roc_auc_score(y_test, ann_prob)
}


print("\nANN Results:")
print(ann_results)


# =========================================================
# 10. 1D CNN MODEL
# =========================================================

X_train_cnn = X_train_dl.reshape(
    X_train_dl.shape[0],
    X_train_dl.shape[1],
    1
)

X_test_cnn = X_test_dl.reshape(
    X_test_dl.shape[0],
    X_test_dl.shape[1],
    1
)


cnn_model = Sequential([
    Conv1D(
        32,
        kernel_size=3,
        activation="relu",
        input_shape=(X_train_cnn.shape[1], 1)
    ),

    MaxPooling1D(pool_size=2),

    Conv1D(
        64,
        kernel_size=3,
        activation="relu"
    ),

    Flatten(),

    Dense(32, activation="relu"),

    Dropout(0.30),

    Dense(1, activation="sigmoid")
])


cnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


cnn_model.fit(
    X_train_cnn,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    verbose=1
)


cnn_prob = cnn_model.predict(X_test_cnn).ravel()

cnn_pred = (cnn_prob >= 0.5).astype(int)


cnn_results = {
    "Model": "1D CNN",
    "Accuracy": accuracy_score(y_test, cnn_pred),
    "Precision": precision_score(y_test, cnn_pred),
    "Recall": recall_score(y_test, cnn_pred),
    "F1 Score": f1_score(y_test, cnn_pred),
    "ROC-AUC": roc_auc_score(y_test, cnn_prob)
}


print("\nCNN Results:")
print(cnn_results)


# =========================================================
# 11. COMBINE ALL RESULTS
# =========================================================

dl_results_df = pd.DataFrame([
    ann_results,
    cnn_results
])


all_results = pd.concat(
    [ml_results_df, dl_results_df],
    ignore_index=True
)


all_results = all_results.sort_values(
    by="F1 Score",
    ascending=False
)


print("\n====================================")
print("ALL MODEL RESULTS")
print("====================================")

print(all_results)


# Save results
all_results.to_csv(
    "model_comparison.csv",
    index=False
)


# =========================================================
# 12. MODEL COMPARISON GRAPH
# =========================================================

plt.figure(figsize=(14, 7))

all_results.set_index("Model")[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
].plot(kind="bar")

plt.title("Machine Learning and Deep Learning Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("model_comparison.png")
plt.close()


# =========================================================
# 13. SELECT BEST ML MODEL
# =========================================================

ml_sorted = ml_results_df.sort_values(
    by="F1 Score",
    ascending=False
)

best_model_name = ml_sorted.iloc[0]["Model"]

print("\n====================================")
print("BEST MACHINE LEARNING MODEL")
print("====================================")

print(best_model_name)


# =========================================================
# 14. SAVE BEST ML MODEL
# =========================================================

best_ml_model = ml_models[best_model_name]

best_ml_model.fit(X_train, y_train)

joblib.dump(
    best_ml_model,
    "models/titanic_best_model.pkl"
)


print("\nBest model saved successfully!")


# =========================================================
# 15. CONFUSION MATRIX
# =========================================================

y_pred_best = best_ml_model.predict(X_test)


cm = confusion_matrix(
    y_test,
    y_pred_best
)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(
    "Confusion Matrix - " + best_model_name
)

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.close()


# =========================================================
# 16. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_best
    )
)


print("\n====================================")
print("PROJECT COMPLETED")
print("====================================")