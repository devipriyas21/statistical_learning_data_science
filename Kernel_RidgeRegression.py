Experiment: Kernel Ridge Regression 
from sklearn.kernel_ridge import KernelRidge 
from sklearn.linear_model import Ridge 

# Step 1: Generate nonlinear data 

np.random.seed(42) 

X = np.linspace(0, 10, 100).reshape(-1, 1) 
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0]) 

# Step 2: Linear Ridge Regression

ridge = Ridge(alpha=1.0) 

ridge.fit(X, y) 
y_ridge = ridge.predict(X) 

# Step 3: Kernel Ridge Regression (RBF kernel) 

krr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.5) 

krr.fit(X, y) 
y_krr = krr.predict(X) 

# Step 4: Visualization

plt.figure(figsize=(8, 5)) 
plt.scatter(X, y, color='gray', alpha=0.5, label="Observed Data")
plt.plot(X, y_ridge, color='red', label="Linear Ridge") 
plt.plot(X, y_krr, color='blue', label="Kernel Ridge (RBF)") 
plt.xlabel("X") 
plt.ylabel("y") 
plt.title("Linear Ridge vs Kernel Ridge Regression")
plt.legend() 
plt.show()
