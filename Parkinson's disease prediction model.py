#importing the necessary packages
import pandas as pd
import numpy as np
import seaborn as sns

#loading the data set
data = pd.read_csv(r'C:\Users\dell\Downloads\parkinsons.csv') #use your file path here
print(data)

#checking for the null values
print(data.isnull().sum())

#dropping the unnecessary columns
data.drop(['name'],axis = 1 ,inplace = True)

#visualizing of the dataset
sns.pairplot(data)

#dividing the data into x and y
x = data.drop(['status'],axis = 1)
y = data['status']

#convert the x into array values for fast accessing
x = np.array(x)

#split the data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3) #test_size refers 30% of data is for testing

#Decision Tree Classifier
from sklearn import tree

#create an instance for it
dt = tree.DecisionTreeClassifier()

#train it using fit method
dt.fit(x_train,y_train)

#predicting the data
prediction = dt.predict(x_test)
print(prediction)

#checking the accuracy
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test,prediction)
print(acc)

#predicting with custom values
categories = ['Not Suffering from PD','Suffering from PD']

custom_data = [['170.756','450.247','79.032','0.00555','0.00003','0.00244','0.00261','0.00731','0.01725','0.175','0.00757','0.00971','0.01652','0.0227','0.01802','25.69','0.486738','0.676023','-4.597834','0.372114','2.975889','0.28278']]

print(categories[int(dt.predict(custom_data))])

o/p = 'Suffering from PD'

