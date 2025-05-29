import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------------------
# To showcase the effect of regularization (None, L1, L2, ElasticNet) in linear regression, a dataset needs the following:
# a. There are many features, but only a few are actually relevant
# b. Some features are highly correlated
# c. There's a bit of noise
# d. The regularization will:
#    1. Suppress irrelevant/noisy features
#    2. Handle multicollinearity
#    3. Shrink coefficients (L2) or set them to zero (L1)
#-----------------------------------------------------

# Create a regression dataset
X, y, coef = make_regression(
    n_samples=200,         # Enough data points
    n_features=50,         # High dimensionality
    n_informative=5,       # Only 5 real features matter
    noise=10.0,            # Add noise
    coef=True,
    random_state=42
)

# Add correlation: manually mix some features
X[:, 10] = X[:, 0] * 0.9 + np.random.normal(scale=0.1, size=200)  # Strongly correlated with feature 0
X[:, 20] = X[:, 1] * -0.8 + np.random.normal(scale=0.2, size=200)  # Strongly negatively correlated

# Put into DataFrame for easier exploration
feature_names = [f"X{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# Optionally visualize correlations
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# Output data preview
df.head()

# Save to CSV
df.to_csv("regularization_dataset.csv", index=False)
print("Dataset saved to 'regularization_dataset.csv'")