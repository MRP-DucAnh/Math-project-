import pandas
import numpy as np
from sklearn import linear_model

#Cleaning Empty cells
def CEC(dataf):
    newdataframe=dataf.dropna() #Delete rows containing NaN . values
    return newdataframe
#Dataframe to numpy Array
def toArr(dataf):
    dataf=dataf.dropna()
    newdataframe=dataf.to_numpy() #convert dataframe to numpy array
    return newdataframe

def Predictor(Z_array, regr,random_error):
    
    print("Linear Regression function: ")
    print("Y=",regr.coef_[0],"*X1 + ",regr.coef_[1],"*X2 + ",regr.coef_[2],"*X3 + ",regr.coef_[3],"*X4"," + ",random_error)
    print("____________________________________Linear Regression____________________________________")
    k=0
    for i in Z_array:
        print("=> We have the result Y[",k,"] = ",regr.predict([i])+random_error,"With the parameters ",i,)
        k+=1

#Create a dataframe to store data from excel file
dataf = pandas.read_excel(r'E:\fpt\Ki123\Python\z_MAI_Project\Health.xlsx')

#Split the above dataframe for easy manipulation
#Independent Values
X = dataf[['X2', 'X3','X4','X5']]
print("Independent Values:")
print(CEC(X),"\n")

#Dependent Values
y = dataf['X1']
print("Dependent Values:")
print(CEC(y),"\n")
#X and y will use like a sample to create Linear Regression function.

#'X7', 'X8','X9','X10' has the same meaning as 'X2', 'X3','X4','X5'
#They are input for future predictions
Z = dataf[['X7', 'X8','X9','X10']]
Z=CEC(Z)
print("Test Value: ")
print(Z,"\n")

regr = linear_model.LinearRegression()
regr.fit(X, y)

Z_array=toArr(Z) #change the data type of Z (dataframe type to array type)

Predictor(Z_array,regr,1) #random error=1 (%)

#print("Test:",regr.rank_)#Rank
print("Test:",regr.rank_)#Rank





"""
The data (X1, X2, X3, X4, X5) are by city.
X1 = death rate per 1000 residents
X2 = doctor availability per 100,000 residents
X3 = hospital availability per 100,000 residents
X4 = annual per capita income in thousands of dollars
X5 = population density people per square mile
Reference: Life In America's Small Cities, by G.S. Thomas
"""






"""
#print(Z_array)
#k=[[71,345,9.199,50],[118,463,7.800000191,35],[121,728,8.199999809,86],[68,383,7.400000095,57],[112,316,10.39999962,57]]
#print(k)

predictedHealth = regr.predict([[71,345,9.199,50],[118,463,7.800000191,35],[121,728,8.199999809,86],[68,383,7.400000095,57],[112,316,10.39999962,57]])
print("X6=", predictedHealth)
print(regr.coef_)
print(regr.coef_[0],regr.coef_[1],regr.coef_[2],regr.coef_[3])
Z_array=Z.values
#Z_array=Z_array.reshape(-1,1)
print("all:")
for i in Z_array:
    print("Predicted test: ",regr.predict([i]))
    
    delete this
"""



