import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


# Load the dataset and split it into training and testing sets.
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create and train the XGBoost model.
params = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "reg:squarederror",
}
xgb_model = xgb.XGBRegressor(**params)
xgb_model.fit(X_train, y_train)

xgb_model.get_booster().save_model("data/booster.json")
