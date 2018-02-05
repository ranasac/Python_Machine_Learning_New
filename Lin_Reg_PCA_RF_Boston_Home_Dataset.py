# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:23:49 2018

@author: ranas
"""
# import libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import mean_squared_error

#import dataset
from sklearn.datasets import load_boston
boston = load_boston()
df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)

#fit a regular linear model
reg = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state = 4)
x_train_orig = x_train
x_test_orig = x_test
x_train_orig = np.array(x_train_orig)
x_test_orig = np.array(x_test_orig)

reg.fit(x_train, y_train)
lin_perf = reg.score(x_test, y_test)
y_pred = reg.predict(x_test)
lin_mse = mean_squared_error(y_test, y_pred) # calculate mse

# perform pca and extract only 8 principal components
pca = PCA(n_components=8, whiten='True')
x = pca.fit(df_x).transform(df_x)
x_train = pca.transform(x_train_orig)
x_test = pca.transform(x_test_orig)
pca_var = pca.explained_variance_

reg = linear_model.LinearRegression()
#x_train, x_test, y_train, y_test = train_test_split(x, df_y, test_size=0.2, random_state = 4)
reg.fit(x_train, y_train)
lin_perf_pca = reg.score(x_test, y_test)
y_pred = reg.predict(x_test)
lin_mse_pca = mean_squared_error(y_test, y_pred)  # calculate mse

#perform SVD as opposed to PCA and retain only 8 components
svd = TruncatedSVD(n_components = 8)
x = svd.fit(df_x).transform(df_x)
x_train = svd.transform(x_train_orig)
x_test = svd.transform(x_test_orig)
reg = linear_model.LinearRegression()
#x_train, x_test, y_train, y_test = train_test_split(x, df_y, test_size=0.2, random_state = 4)
reg.fit(x_train, y_train)
lin_perf_svd = reg.score(x_test, y_test)
y_pred = reg.predict(x_test)
lin_mse_svd = mean_squared_error(y_test, y_pred) #calculate mse

# print the correlation matrix of input data
corr_mat = df_x.corr()
plt.imshow(corr_mat)
plt.show()

# plot the bar chart of scores of linear regression model
objects = ('Lin_reg_all','Lin_reg_pca','Lin_reg_svd')
y_pos = np.arange(len(objects))
performance = [lin_perf, lin_perf_pca,lin_perf_svd]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Performance')
plt.title('Performance of different models')
plt.show()

# fit random forest regression model and use original input features
rf = RandomForestRegressor(n_estimators = 100)
rf.fit(x_train_orig, y_train)
y_pred = rf.predict(x_test_orig)
lin_mse_rf = mean_squared_error(y_test, y_pred)
print(rf.feature_importances_) #print the importance of input parameters
 # bar chart of importance of input parameters
plt.bar(np.arange(len(x_train_orig[0,0:])), rf.feature_importances_,  align='center', alpha=0.5)
plt.xticks(np.arange(len(x_train_orig[0,0:])), np.arange(len(x_train_orig[0,0:])))
plt.ylabel('Relative Importance')
plt.title('Importance of different input parameters')
plt.show()

# bar chart of mean squared error of all the fitted models
objects = ('Lin_mse_all','Lin_mse_pca','Lin_mse_svd','Lin_mse_rf')
y_pos = np.arange(len(objects))
performance = [lin_mse, lin_mse_pca,lin_mse_svd, lin_mse_rf]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Mean Squared Error')
plt.title('Performance of different models')
plt.show()
