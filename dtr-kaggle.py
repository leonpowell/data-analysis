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

# specify the model. #For model reproducibility, set a numeric value for random_state when specifying the model # 
dtr_model = DecisionTreeRegressor(random_state=10)  

# Fit the model 
dtr_model.fit(train_X, train_y) 

# make predictions 
val_predictions = dtr_model.predict(val_X) 

print(val_predictions) 

print(mean_absolute_error(val_y, val_predictions))


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