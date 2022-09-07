# For data wrangling 
import numpy as np
import pandas as pd

# For visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
pd.options.display.max_rows = None #to show all rows
pd.options.display.max_columns = None #to show all columns

# For encoding categorical data
from sklearn.preprocessing import OneHotEncoder

# For scaling
from sklearn.preprocessing import RobustScaler

# For splitting data
from sklearn.model_selection import train_test_split

# For modelling
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures

# For evaluation
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_auc_score, roc_curve



# Aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
mpl_color = sns.color_palette('Set2')

# Plot features universal settings
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('font', size=13)



df = pd.read_csv('creditcard.csv')
df.head()
df.info()
df1 = df.copy() # Just in case
df.drop('Time', axis = 1, inplace = True)
print(df.head())


dup = df[df.duplicated()]
print("Number of duplicated records total:", len(dup))
print("Number of duplicated records in Fraud cases:", len(dup[dup["Class"]==1]))
print("Number of duplicated records in No Fraud cases:", len(dup[dup["Class"]==0]))


total = len(df)
df.drop_duplicates(inplace=True)
print(total - len(df), " duplicated records removed")
print("Total records left:", len(df))

total = len(df)
df.drop_duplicates(inplace=True)
print(total - len(df), " duplicated records removed")
print("Total records left:", len(df))
