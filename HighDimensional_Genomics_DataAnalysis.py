# LAB EXPERIMENT: High-Dimensional Genomics Data Analysis 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposiƟon import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

# Step 1: Generate synthetic genomics data
np.random.seed(42) 
n_samples = 100 
n_genes = 10000 
n_informative = 40
X = np.random.normal(0, 1, (n_samples, n_genes)) 
true_genes = np.random.choice(n_genes, n_informative, replace=False)
beta = np.zeros(n_genes) 
beta[true_genes] = np.random.uniform(1.5, 3.0, n_informative)
y_score = X @ beta + np.random.normal(0, 5, n_samples) 
y = (y_score > np.median(y_score)).astype(int) 

# Step 2: Standardization
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
# Step 3: Feature screening (Variance-based) 
variances = np.var(X_scaled, axis=0) 
selected_genes = np.argsort(variances)[-2000:] 
X_screened = X_scaled[:, selected_genes] 
# Step 4: PCA 
pca = PCA(n_components=0.95) 
X_pca = pca.fit_transform(X_screened) 
# Step 5: Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split( 
 X_pca, y, test_size=0.25, random_state=42
)

# Step 6: Sparse Logistic Regression
model = LogisticRegressionCV(
 penalty='elasticnet',
 solver='saga', 
 l1_ratios=[0.5],
 cv=5, 
 max_iter=5000 
) 
model.fit(X_train, y_train) 

# Step 7: Evaluation 
y_pred = model.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
print("Classification Accuracy:", accuracy)
