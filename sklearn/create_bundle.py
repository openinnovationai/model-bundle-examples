from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

import joblib

if __name__ == "__main__":
    # Load the dataset and split it into training and testing sets.
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    print(X_train[0])
    print(y_train[0])

    rf = RandomForestRegressor(n_estimators=10, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    joblib.dump(rf, "data/model.joblib")
