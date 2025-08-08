import numpy as np
import statsmodels.api as sm


if __name__ == "__main__":
    # Generate sample data with shape (-1, 10) for 10 features and a single output
    np.random.seed(42)
    num_samples = 100
    num_features = 10

    # Generating the response variable as a linear combination of the features with some noise
    X = np.random.rand(num_samples, num_features)
    true_coefficients = np.random.rand(num_features)
    noise = np.random.normal(loc=0, scale=0.1, size=num_samples)
    y = np.dot(X, true_coefficients) + noise

    # Add a constant term to the independent variable (intercept)
    X = sm.add_constant(X)

    # Create a linear regression model
    model = sm.OLS(y, X)

    # Fit the model to the data
    model_fit = model.fit()

    model_fit.save("model.pkl")
    print("Model weights are saved into ./model.pkl")
