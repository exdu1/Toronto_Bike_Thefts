# import modules
import pandas as pd
import os
import matplotlib.pyplot as pt

# Import data from csv file
path = "C:/Users/Eric/Desktop/COMP_309_Data_Warehousing/Toronto_Bike_Thefts"
filename = 'bike_thefts.csv'
fullpath = os.path.join(path, filename)

# Read data
data = pd.read_csv(fullpath)
print(data.head(10))
print(data.columns.values)
data.dtypes
data.describe()
data.info()

dummy_month  = pd.get_dummies((data['OCC_MONTH']))
