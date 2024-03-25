import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes


# Load the dataset and split it into training and testing sets.
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

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
