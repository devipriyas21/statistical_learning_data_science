#EXPERIMENT: Partial Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.api as sm

# Step 1: Generate synthetic data
np.random.seed(42)

n = 300
X = np.random.normal(0, 1, n) # Linear covariate
Z = np.random.uniform(0, 10, n) # Nonlinear covariate
beta_true = 2.0
g_true = np.sin(Z) # True nonlinear funcƟon
y = beta_true * X + g_true + np.random.normal(0, 0.5, n)

# Step 2: Estimate E(y|Z) using nonparametric regression
knn_y = KNeighborsRegressor(n_neighbors=20)
knn_y.fit(Z.reshape(-1, 1), y)
Ey_z = knn_y.predict(Z.reshape(-1, 1))

# Step 3: Estimate E(X|Z)
knn_x = KNeighborsRegressor(n_neighbors=20)
knn_x.fit(Z.reshape(-1, 1), X)
Ex_z = knn_x.predict(Z.reshape(-1, 1))

# Step 4: Residualize y and X
y_star = y - Ey_z
X_star = X - Ex_z

# Step 5: EsƟmate beta using OLS
X_star_mat = sm.add_constant(X_star)
plr_model = sm.OLS(y_star, X_star_mat).fit()
print("Estimated beta (Partial Linear Model):", plr_model.params[1])
print("True beta:", beta_true)

# Step 6: Estimate nonlinear component g(Z)
g_hat = Ey_z - plr_model.params[1] * Ex_z

# Step 7: Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(Z, g_true, alpha=0.5, label="True g(Z)")
plt.scatter(Z, g_hat, alpha=0.5, label="Estimated g(Z)")
plt.xlabel("Z")
plt.ylabel("g(Z)")
plt.title("Nonlinear Component Estimation")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label="Observed Data")
plt.plot(X, beta_true * X, color='black', label="True Linear Effect")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Component")
plt.legend()
plt.show()
