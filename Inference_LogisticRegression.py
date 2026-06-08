#EXPERIMENT: Inference in GLMs (Logistic Regression)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 1: Generate binary classification data
np.random.seed(42)

n = 300
X = np.random.normal(0, 1, (n, 2))
beta_true = np.array([1.5, -2.0])

linear_pred = X @ beta_true
prob = 1 / (1 + np.exp(-linear_pred))
y = np.random.binomial(1, prob) 
X_mat = sm.add_constant(X) 

# Step 2: Fit LogisƟ c GLM
glm_logit = sm.GLM( 
 y, X_mat, 
 family=sm.families.Binomial() 
).fit() 
print(glm_logit.summary()) 
Statistical Inference: Wald Test & Confidence Intervals
# Wald test (z-test) and confidence intervals 

print("\nConfidence Intervals:") 
print(glm_logit.conf_int()) 
print("\nWald Test (z-staƟ sƟ cs):")
print(glm_logit.tvalues) 
Likelihood Ratio Test

# Reduced model (intercept only) 
glm_null = sm.GLM( 
 y, 
 np.ones((n, 1)), 
 family=sm.families.Binomial() 
).fit() 
lr_stat = 2 * (glm_logit.llf - glm_null.llf) 
df = glm_logit.df_model - glm_null.df_model 
print("\nLikelihood Ratio Statistic:", lr_stat)
print("Degrees of Freedom:", df) 
