from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

for i, tree in enumerate(model.estimators_):
    plt.figure(figsize=(20, 10))
    plot_tree(tree, 
              feature_names=iris.feature_names, 
              class_names=iris.target_names, 
              filled=True,
              rounded=True)
    plt.title(f"Tree Number {i+1} in the Random Forest")
    plt.show()
