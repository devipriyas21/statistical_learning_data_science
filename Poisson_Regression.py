#EXPERIMENT: Poisson Regression 

# Step 1: Generate count data 

np.random.seed(42) 
n = 200 

X = np.random.normal(0, 1, (n, 1)) 
beta_true = 0.8 
mu = np.exp(1 + beta_true * X.ravel()) 
y = np.random.poisson(mu) 
X_mat = sm.add_constant(X) 

# Step 2: Fit Poisson GLM 

glm_pois = sm.GLM( 
 y, X_mat, 
 family=sm.families.Poisson() 
).fit() 
print(glm_pois.summary())
