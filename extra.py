
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("preprocessed_data.csv")

# Convert regression target to binary classification
median_price = df["MEDV"].median()
df["target"] = (df["MEDV"] >= median_price).astype(int)

X = df.drop(["MEDV", "target"], axis=1)
y = df["target"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# DECISION TREE
# ============================================================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("="*60)
print("Decision Tree Results")
print("="*60)

print("Accuracy:", accuracy_score(y_test, y_pred_dt))

print(classification_report(
    y_test,
    y_pred_dt,
    target_names=["Low Price","High Price"]
))

# ============================================================
# VISUALIZE TREE
# ============================================================

plt.figure(figsize=(18,10))

plot_tree(
    dt,
    feature_names=X.columns,
    class_names=["Low","High"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree")
plt.show()

# ============================================================
# OVERFITTING ANALYSIS
# ============================================================

train_acc = []
test_acc = []

depths = range(1,11)

for d in depths:

    model = DecisionTreeClassifier(
        max_depth=d,
        random_state=42
    )

    model.fit(X_train,y_train)

    train_acc.append(model.score(X_train,y_train))
    test_acc.append(model.score(X_test,y_test))

plt.figure(figsize=(7,5))

plt.plot(depths,train_acc,label="Training Accuracy")
plt.plot(depths,test_acc,label="Testing Accuracy")

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting Analysis")
plt.legend()
plt.grid()

plt.show()

# ============================================================
# GRID SEARCH
# ============================================================

params = {
    "max_depth":[2,3,4,5,6,None],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train,y_train)

best_tree = grid.best_estimator_

print("\nBest Decision Tree Parameters")
print(grid.best_params_)

# ============================================================
# RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)

print("\n")
print("="*60)
print("Random Forest Results")
print("="*60)

print("Accuracy:",accuracy_score(y_test,y_pred_rf))

print(classification_report(
    y_test,
    y_pred_rf,
    target_names=["Low Price","High Price"]
))

# ============================================================
# COMPARE ACCURACY
# ============================================================

print("\nDecision Tree Accuracy:",
      accuracy_score(y_test,y_pred_dt))

print("Random Forest Accuracy:",
      accuracy_score(y_test,y_pred_rf))

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")

print(importance)

plt.figure(figsize=(8,6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Random Forest Feature Importance")

plt.show()

# ============================================================
# CROSS VALIDATION
# ============================================================

cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Accuracy")

print(cv_scores)

print("Mean Accuracy:",cv_scores.mean())

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test,y_pred_rf)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Low Price","High Price"]
).plot(cmap="Blues")

plt.title("Random Forest Confusion Matrix")

plt.show()

print("\nMODEL TRAINING COMPLETE")