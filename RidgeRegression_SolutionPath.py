# Experiment: Ridge Regression Solution Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Step 1: Generate data
np.random.seed(42)
n, p = 100, 10

X = np.random.randn(n, p)
beta_true = np.random.randn(p)
y = X @ beta_true + np.random.normal(0, 2, n)

# Step 2: Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Ridge solution path
alphas = np.logspace(-3, 3, 50)
coefs = []

for a in alphas:
 ridge = Ridge(alpha=a)
 ridge.fit(X_scaled, y)
 coefs.append(ridge.coef_)
  
coefs = np.array(coefs)

# Step 4: Plot solution path
plt.figure(figsize=(8, 5))
for i in range(p):
 plt.semilogx(alphas, coefs[:, i])
  
plt.xlabel("Alpha (λ)")
plt.ylabel("Coefficient Value")
plt.title("Ridge Regression Solution Path")
plt.show()
