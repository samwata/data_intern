# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:46:54 2020

@author: intern
"""
import pandas as pd
# save filepath to variable for easier access

melbourne_file_path = 'C:/Users/intern/Desktop/Samwata Python Projects/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
melbourne_data.describe()

melbourne_data.columns

y = melbourne_data.Price
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
#selecting multiple features(columns we will use to predict) to use in our model
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
X.describe() # to see the summary of data
X.head()  #to view the data


#Specify and fit the model
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1) #You use any number, 
#and model quality won't depend meaningfully on exactly what value you choose.

# Fit model
melbourne_model.fit(X, y)
#make predictions for the first few rows of the training data to see how the predict function works.
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")

#Predict values
predictions = melbourne_model.predict(X.head())
print(predictions)

#Calculate Mean Absolute Error (MEA)
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

#Validating the data in our model
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#We can use a utility function to help compare MAE scores from different 
#values for max_leaf_nodes:
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
    
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    
    
    
    
    
    







