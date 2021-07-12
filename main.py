#import necessary libraries
import pandas as pd 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

#Start work on the main arrays
##Import the main observation data from the spreadsheet
data = pd.read_csv('LDAWeek9RawData.csv')
## use this to check if it works (it does) print(data)
### Read the data for the second dataframe, mouseID, in manually, since doing it through reading a CSV creates a dataframe, which throws up a joining error on line 16 for some reason. The codes needed for each week are listed in the LDAcodes.txt file. 
mouseID = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,]
exCat = ["No-Exercise","Exercise"]
mouseClass = pd.Categorical.from_codes(mouseID, exCat)
mouseDataFrame = data.join(pd.Series(mouseClass, name='class'))

#Begin manually solving the LDA in order to plot the points in pyplot (mouseDataFrame is needed for this)
#Step 1: Creating a set of vectors w/ the means for the observation parameters
classMeans = pd.DataFrame(columns=exCat)

#The next 3 lines of code serve to take the mean for every column in the csv via a for loop. C represents the number of distinct classes mathematically. 
for c, rows in mouseDataFrame.groupby('class'):
    classMeans[c] = rows.mean()

#With the means obtained, we can now find the in-class scatter matrix for plotting
##This next line just creates an array, 10x10 in shape, filled entirely with zeroes (Ten rows, 10 columns, 10 input variables).  
scatterMatrix = np.zeros((10,10))
for c, rows in mouseDataFrame.groupby('class'):
  # Line 34 creates a new array with just the data, that drops the Exercise Status (the .drop method is a pandas method, and setting the axis to 1 drops the column)
  rows = rows.drop(['class'], axis=1)
  #This next row creates a second 10x10 array of zeroes  
  s = np.zeros((10,10))
  #Next, we iterate over the array and plug in the mean to obtain the in-class scatter. This is done using the iterrows method of panda, which iterates over the rows in a dataframe. 
  for index, row in rows.iterrows():
    x, mc = row.values.reshape(10,1),classMeans[c].values.reshape(10,1)
  s += (x - mc).dot((x - mc).T)
  scatterMatrix += s

#Step 2: Obtain Between-Class Scatter matrix. Start by finding the means and creating a new matrix of zeroes. 
feature_means = mouseDataFrame.mean()
bcScatterMatrix = np.zeros((10,10))

#Iterate through the classMeans to create a new array and reshape the feature means dataframe 
for c in classMeans:    
    n = len(mouseDataFrame.loc[mouseDataFrame['class'] == c].index)
    
    mc, m = classMeans[c].values.reshape(10,1), feature_means.values.reshape(10,1)
    
    bcScatterMatrix += n * (mc - m).dot((mc - m).T)

#now use numpy and pandas to multiple the in-class and between-class scatter matrices to find the linear discriminants of the problem (in other words, the eigenvalues)  
eigenValues, eigenVectors = np.linalg.eig(np.linalg.inv(scatterMatrix).dot(bcScatterMatrix))
#Now get those eigenValues printed out 
print ("The eigenvalues are as follows:")
print(eigenValues)

#Now order the eigenvalues from least to greatest, since the greatest eigenvalues show the most variance. Store those ordered eigenPairs in a temporary array. 
eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]
eigenPairs = sorted(eigenPairs, key=lambda x: x[0], reverse=True)
for eigenPair in eigenPairs:
    print("Ordered Eigenvalues: ", eigenPair[0])
#To better represent the variance in the data, restructure the eigenvalues into percentages of the total sum of the eigenValues. 
eigenSum = sum(eigenValues)

#print('Variance explained via percentages')
for i, eigenPair in enumerate (eigenPairs):
  print('Eigenvector {}: {}'.format(i, (eigenPair[0]/eigenSum).real))
w_matrix = np.hstack((eigenPairs[0][1].reshape(10,1), eigenPairs[1][1].reshape(10,1))).real



