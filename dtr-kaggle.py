# Code you have previously used to load data 
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split

# Path of the file to read 
train_data=pd.read_csv('../input/home-data-for-ml-course/train.csv') 

# print the list of columns in the dataset to find the name of the prediction train_data.columns 
train_data.shape 
train_data.describe() 
train_data.head() 



# dropna drops missing values (think of na as "not available") 
filtered_train_data = train_data.dropna(axis=0) 

filtered_train_data.shape 
filtered_train_data.describe()
filtered_train_data.head() 

# target 
y = filtered_train_data.SalePrice 


# Create the list of features below 
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd'] 

# Select data corresponding to features in feature_names 
X = filtered_train_data[feature_names] 

X.shape 

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# a utility function to help compare 
# MAE scores from different values for max_leaf_nodes:

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    dtr_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    dtr_model.fit(train_X, train_y)
    val_predictions = dtr_model.predict(val_X)
    mae = mean_absolute_error(val_y, val_predictions)
    return(mae)



for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# read test data file using pandas 
test_data = pd.read_csv('../input/test.csv') 
test_data.shape 

# dropna drops missing values (think of na as "not available") 
test_data = test_data.dropna(axis=0)  
test_data.shape 

# create test_X which comes from test_data but includes only the columns you used for prediction. The list of columns is stored in a variable called features. 
test_X = test_data[feature_names] 


# make predictions which we will submit  
test_preds = dtr_model.predict(test_X) 
test_preds.shape 

# Run the code to save predictions in the format used for competition scoring  
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds}) 
output.to_csv('submission.csv', index=False) 