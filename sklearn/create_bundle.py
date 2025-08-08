import joblib
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # Load the dataset and split it into training and testing sets.
    X, y = load_diabetes(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(X, y)

    rf = RandomForestRegressor(n_estimators=10, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    joblib.dump(rf, "model.joblib")
    print("Model weights saved into ./model.joblib")
