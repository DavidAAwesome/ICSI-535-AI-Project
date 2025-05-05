from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# dataset
from keras.datasets import mnist


# load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


def splitData(X, y, test_size=0.2, random_state=42):
    X_train

    return (X_train, X_test, y_train, y_test)
