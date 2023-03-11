
import pandas as pd
import pickle

crop_data=pd.read_csv("crop_production.csv")
crop_data = crop_data.dropna()
crop_data['Yield'] = (crop_data['Production'] / crop_data['Area'])
data = crop_data.drop(['State_Name'], axis = 1)
dummy = pd.get_dummies(data)
from sklearn.model_selection import train_test_split

x = dummy.drop(["Production","Yield"], axis=1)
y = dummy["Production"]

# Splitting data set - 25% test dataset and 75% 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=5)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 5)
regressor.fit(x_train,y_train)


pickle.dump(regressor, open('crop-yield.pkl','wb'))
