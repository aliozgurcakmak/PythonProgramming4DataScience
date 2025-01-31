"""### NUMPY ###
#############"""

# Why NumPy?
# Creating NumPy Array
# Attributes of NumPy Array
# ReShaping
# Index Selection
# Slicing
# Fancy Index
# Conditions on NumPy
# Math Operations


#region Why NumPy?
"""NumPy is an ideal library for data processing due to its speed, high-performance vectorized operations, robust mathematical functions,  low-level optimizations."""

import numpy as np
a = [1,2,3,4]
b = [2,3,4,5]

ab = []
for i in range(len(a)):
    ab.append(a[i]*b[i])
# It was classical Pythonic way. Now Let's use NumPy.

a = np.array(a)
b = np.array(b)
a * b
#endregion

#region Creating NumPy Array
import numpy as np

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))
np.zeros(10, dtype=int)
np.zeros(10)
np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3,4))


#endregion

#region Attributes of NumPy Array
import numpy as np
# ndim: dimension count
# shape: dimension information
# size: total element count
# dtype: array data type

a = np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype
#endregion

#region ReShaping
import numpy as np

np.random.randint(1,10,size=9)
np.random.randint(1,10,size=9).reshape(3,3)

ar = np.random.randint(1,10,size=9)
ar = ar.reshape(3,3)

#endregion

#region Index Selection & Slicing
import numpy as np

a = np.random.randint(10,size=9)
a[0]
a[0:5]
a[0] = 999

m = np.random.randint(10,size=(3,5))
m[0,0]
m[1,1]
m[2,3] = 65+65
m[2,3] = 2.99
m[:, 0]
m[1, :]
m[0:2, 0:3]
#endregion

#region Fancy Index

"""Fancy indexing is a type of indexing in NumPy that allows selecting multiple elements of an array at once using lists or arrays as indices."""

v = np.arange(0,30,3)

v[1]
v[4]

catch = [1,2,3]
v[catch]
#endregion

#region Conditions on NumPy

import numpy as np
v = np.array([1,2,3,4,5])

"""Let's find the elements which are smaller than 3. Firstly we gonna use classical Pythonic way."""
ab = []
for i in range(len(v)):
    if v[i] < 3:
        ab.append(v[i])

"""Now let's we use NumPy."""

v = [v < 3]
v = [v > 3]
v = [v == 3]
v = [v != 3]
v = [v <= 3]
v = [v >= 3]
#endregion

#region Math Operations

import numpy as np
v = np.array([1,2,3,4,5])

v / 5
v * 5 / 10
v ** 2
v + 5
v - 5

np.subtract(v,1)
np.add(v,1)
np.multiply(v,2)
np.divide(v,0.5)
np.power(v,2)
np.mean(v)
np.sum(v)
np.max(v)
np.min(v)
np.var(v)
v = np.subtract(v,1)

"""Solution of a equation of two equations with two variables."""
# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5,1], [1,3]])
b = np.array([12,10])

np.linalg.solve(a, b)
#endregion


"""
##############
### PANDAS ###
##############
"""

# Pandas Series
# Reading Data
# Quick Look at Data
# Selection in Pandas
# Aggregation and Grouping
# Apply and Lambda
# Join Operations

#region Pandas Series
import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)

s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
"""When the `values` attribute is applied to a Pandas Series and you want to access its values, it returns a NumPy array, as the index information is disregarded."""
s.head()
s.head(3)
s.tail()
s.tail(3)
#endregion

#region Reading Data
import pandas as pd

df = pd.read_csv('datasets/access-code-password-recovery-code.csv')
df.head()

"""If you click on "pd", you can access the documentation for Pandas functions."""
"""You can access to most popular functions of Pandas with Pancas Cheatsheet"""
#endregion

#region Quick Look at Data
import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')
df.head()
df.tail()
a = df.shape
df.info()
b = df.columns
c = df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].head( )
df["sex"].value_counts()
#endregion

#region Selection in Pandas
import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')
df.head()

df.index
df[0:13]
df.drop(axis=0, index=0).head()
delete_indexes = [1,3,5,7]
df.drop(axis=0, index=delete_indexes).head()

"""to permanently delete:
-- df = df.drop(axis=0, index=delete_indexes)
-- df = df.drop(axis=0, index=delete_indexes, inplace=True)"""

"""To convert variable to index:"""
df["age"].head()
df.age.head()

df.index = df["age"]
df.drop("age", axis=1).head()
df.drop("age", axis=1, inplace=True)
df.head()

"""To convert index to variable:"""
df.index

df["age"] = df.index
df.head()
df.drop("age", axis=1, inplace=True)

df.reset_index().head()
df = df.reset_index().head()
#endregion

#region Operations on Variables
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset('titanic')
df.head()

"age" in df
df["age"].head()
df.age.head()

df["age"].head()
type(df["age"].head())
"""type(df["age"].head())
<class 'pandas.core.series.Series'>"""

"""If you want the result as dataframe when you select a variable, then you need use [[]] instead of []"""
df[["age"]].head()
type(df[["age"]].head())
"""type(df[["age"]].head())
<class 'pandas.core.frame.DataFrame'>"""

df[["age", "alive"]]

col_Names = ["age", "alive", "adult_male"]
df[col_Names]

df["age2"] = df["age"]**2
df["age3"] = df["age"] / df["age2"]
df.drop(col_Names, axis=1).head()

df.loc[:, ~ df.columns.str.contains('age')].head()
#endregion

#region iLoc & Loc
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset('titanic')
df.head()

"""iLoc: integer-based selection"""
df.iloc[0:3]
df.iloc[0:3,0:3]

"""Loc: label-based selection"""
df.loc[0:3]
df.loc[0:3]

df.iloc[0:3, "age"]
# """iLoc needs an integer input"""
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]
#endregion

#region Conditional Selection
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset('titanic')
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, ["age", "class"]].head()
df.loc[(df["age"] > 50) & (df["sex"]=="male"), ["age", "class"]].count()

df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50)
       & (df["sex"]=="male")
       & ((df["embark_town"]== "Cherbourg") | (df["embark_town"]=="Southampton")),
       ["age", "class", "embark_town"]]

df_new["embark_town"].value_counts()
#endregion

#region Aggregation & Grouping

# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset('titanic')
df.head()

df["age"].mean()

df.groupby("sex")["age"].mean()

df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town"]).agg({"age": ["mean"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],
                       "survived": "mean"})


df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean"],
    "survived": "mean",
    "sex": "count"})
#endregion

#region Pivot Table
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset('titanic')
df.head()


df.pivot_table("survived", "sex", "embarked")

df.pivot_table("survived", "sex", ["embarked","class"])

df["new_age"] = pd.cut(df["age"], [0,10,18,25,40,90])
df.head()

df.pivot_table("survived", "sex",["new_age", "class"] )

df.pivot_table("survived",
               ["sex", "new_age"],
               ["embarked","class"])

pd.set_option("display.width", 500)
#endregion

#region Apply & Lambda
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset('titanic')
df.head()

"""Apply: It provides the possibility to run functions automatically in rows or columns."""

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

(df["age"]/10).head()
df["age2"]/10
df["age3"]/10

for col in df.columns:
    if "age" in col:
        print(col)


for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10

df[["age", "age2", "age3"]].apply(lambda x: x/10).head()


df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x-x.mean()) / x.std()).head()

def standard_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standard_scaler).head()

"""df.loc[:, ["age", "age2", "age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standard_scaler).head()"""
df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standard_scaler).head()

df.head()

#endregion

#region Join Operations
import pandas as pd
import seaborn as sns
import numpy as np
m = np.random.randint(1,30,size=(5,3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2])

pd.concat([df1, df2], ignore_index=True)

## Join Operations with Merge

df1 = pd.DataFrame({'employees': ["John", "Dennis","Mark", "Maria"],
                    "group": ["Accounting", "Engineering", "Engineering", "HR"]})

df2 = pd.DataFrame({'employees': ["Mark", "John", "Dennis","Maria"],
                    "start_Date": [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
pd.merge(df1, df2, on="employees")

"""Mission: Access the manager information of all employees."""

df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({"group": ["Accounting", "Engineering", "HR"],
                    "manager": ["Caner", "Mustafa", "Berkcan"]})


df5 = pd.merge(df3, df4, on="group")
#endregion


"""
##############
### DATA VISUALIZATION: MATPLOTLIB & SEABORN ###
##############
"""

# MatPlotLib
### If you have a categoric variable, you need column graphic: countplot, barplot.
### If you have a numeric variable, you need histogram or boxplot.

#region Categoric Variable Visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset('titanic')
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show()
#endregion

#region Numeric Variable Visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset('titanic')
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()
#endregion

#region Attributes of Matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

## plot
x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o")
plt.show()


x = np.array([2,4,6,8,10])
y = np.array([1,3,5,7,9])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o")
plt.show()

plt.plot(x, y, "o-")
plt.show()

## marker

y = np.array([13, 28, 11, 100])
plt.plot(y, marker="o")
plt.show()

y = np.array([13, 28, 11, 100])
plt.plot(y, marker="*")
plt.show()

## line

y = np.array([13, 28, 11, 100])
plt.plot(y)
plt.show()

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dashed")
plt.show()

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dotted")
plt.show()


y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dashdot", color="red")
plt.show()


## multiple lines

x = np.array([23, 8, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()


## labels

x = np.array([15,30,45,60,75])
y = np.array([250,260,270,280,290])
plt.plot(x, y)
plt.title("Big Title")
plt.xlabel("Axis X")
plt.ylabel("Axis Y")
plt.grid()
plt.show()

## subplots

# plot 1
x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([240,250,260,270,280,290,300,310,320,330])
plt.subplot(1,2,1)
plt.title("Plot 1")
plt.plot(x, y)

# plot 2

x = np.array([91,57,34,21,13,8,5,3,2,1])
y = np.array([200,210,220,230,240,250,260,270,280,290])
plt.subplot(1,2,2)
plt.plot(x, y)
plt.title("Plot 2")
plt.show()


## Defining 3 graphs as one row and two columns

# plot 1

x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([240,250,260,270,280,290,300,310,320,330])
plt.subplot(1,3,1)
plt.title("Plot 1")
plt.plot(x, y)

# plot 2

x = np.array([91,57,34,21,13,8,5,3,2,1])
y = np.array([200,210,220,230,240,250,260,270,280,290])
plt.subplot(1,3,2)
plt.plot(x, y)
plt.title("Plot 2")

# plot 3

x = np.array([101,107,113,119,125,131,137,143,149,155])
y = np.array([200,210,220,230,240,250,260,270,280,290])
plt.subplot(1,3,3)
plt.plot(x, y)
plt.title("Plot 3")
plt.show()
#endregion

#region Attributes of Seaborn
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset('tips')
df.head()

df["sex"].value_counts()
sns.countplot(x= df["sex"], data=df)
plt.show()

df["sex"].value_counts().plot(kind="bar")
plt.show()

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()
#endregion


"""
##############################
### ADVANCE FUNCTIONAL EDA ###
##############################
"""

# Overall View
# Analysis of Categorical Variables
# Analysis of Numerical Varaibles
# Analysis of Target Varaible
# Analysis of Corelation

#region Overall View
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset('titanic')
df.head()

df.head()
df.info()
df.describe()
df.tail()
df.shape
df.columns
df.index
df.isnull().values.any()
df.isnull().sum()


def check_df(dataframe, head=5):
    print("################## Shape ##################")
    print(dataframe.shape)
    print("################## Types ##################")
    print(dataframe.dtypes)
    print("################## Head ##################")
    print(dataframe.head(head))
    print("################## Tail ##################")
    print(dataframe.tail(head))
    print("################## NA ##################")
    print(dataframe.isnull().sum())
    print("################## Quantiles ##################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1 ]).T)


check_df(df)

df = sns.load_dataset('tips')
check_df(df)

df = sns.load_dataset('flights')
check_df(df)

#endregion

#region Analysis of Categorical Variables
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset('titanic')
df.head()

df["embarked"].value_counts()
df["sex"].unique()
df["sex"].nunique()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]

num_but_bat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["object", "category", "bool"]]

cat_cols = cat_cols + num_but_bat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols]
df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]



def cat_summary (dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
    "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("                                      ")
    print("######################################")
    print("                                      ")


cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)

def cat_summary (dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
    "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("                                      ")
    print("######################################")
    print("                                      ")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


cat_summary(df, "sex", plot=True)

for col in cat_cols:

        cat_summary(df, col, plot=True)

df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df, col, plot=True)



def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("                                      ")
        print("######################################")
        print("                                      ")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("                                      ")
        print("######################################")
        print("                                      ")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

cat_summary(df, "sex", plot=True)

#endregion

#region Analysis of Numerical Variables
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset('titanic')
df.head()

df[["age", "fare"]].describe().T



cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]
num_but_bat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["object", "category", "bool"]]
cat_cols = cat_cols + num_but_bat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col_name):
    df[["age", "fare"]].describe().T
    quantiles = [0, 0.05, 0.50, 0.95, 0.99, 1]
    print(df[numerical_col_name].describe(quantiles).T)
    print("                                      ")
    print("######################################")
    print("                                      ")


num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)

def num_summary(dataframe, numerical_col_name, plot=False):
    df[["age", "fare"]].describe().T
    quantiles = [0, 0.05, 0.50, 0.95, 0.99, 1]
    print(df[numerical_col_name].describe(quantiles).T)
    print("                                      ")
    print("######################################")
    print("                                     ")

    if plot:
        dataframe[numerical_col_name].hist()
        plt.xlabel(numerical_col_name)
        plt.title(numerical_col_name)
        plt.show(block=True)

num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

#endregion

#region Catching Variables and Generalization of Operations

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset('titanic')
df.head()
df.info()

#docstring
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives names of categoric, numeric and categoric but cardinal variables of a dataset.

    Parameters
    ----------
    dataframe: dataframe
        The dataframe whose variable names are to be written is this one.
    cat_th: int, float
        A threshold for numeric but categoric variables.
    car_th: int, float
        A threshold for categoric but cardinal variables.

    Returns
    -------
    cat_cols: list
        Categoric variable names.
    num_cols: list
        Numeric variable names.
    cat_but_car: list
        Categoric looking but cardinal variable names.

    Notes
    ------
    cat_cols + num_cols + cat_but_car = total variable number
    num_but_vat is inside of cat_cols



    """

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["object", "category", "bool"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categoric Variables: {len(cat_cols)}")
    print(f"Numeric Variables: {len(num_cols)}")
    print(f"Categoric looking but cardinal Variables: {len(cat_but_car)}")
    print(f"Numeric but categoric Variables: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)



def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("                                      ")
    print("#######################################")


cat_summary(df,"sex")

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col_name, plot=False):
    df[["age", "fare"]].describe().T
    quantiles = [0, 0.05, 0.50, 0.95, 0.99, 1]
    print(df[numerical_col_name].describe(quantiles).T)
    print("                                      ")
    print("######################################")
    print("                                     ")

    if plot:
        dataframe[numerical_col_name].hist()
        plt.xlabel(numerical_col_name)
        plt.title(numerical_col_name)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


#bonus
df = sns.load_dataset('titanic')
df.info()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)




cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("                                      ")
    print("#######################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

#endregion

#region Analysis of Target Variables

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset('titanic')

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives names of categoric, numeric and categoric but cardinal variables of a dataset.

    Parameters
    ----------
    dataframe: dataframe
        The dataframe whose variable names are to be written is this one.
    cat_th: int, float
        A threshold for numeric but categoric variables.
    car_th: int, float
        A threshold for categoric but cardinal variables.

    Returns
    -------
    cat_cols: list
        Categoric variable names.
    num_cols: list
        Numeric variable names.
    cat_but_car: list
        Categoric looking but cardinal variable names.

    Notes
    ------
    cat_cols + num_cols + cat_but_car = total variable number
    num_but_vat is inside of cat_cols



    """

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["object", "category", "bool"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categoric Variables: {len(cat_cols)}")
    print(f"Numeric Variables: {len(num_cols)}")
    print(f"Categoric looking but cardinal Variables: {len(cat_but_car)}")
    print(f"Numeric but categoric Variables: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("                                      ")
    print("#######################################")

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

df["survived"].value_counts()
cat_summary(df, "survived")

######################################################
# Analysis of Target Variable with Categoric Variables
######################################################

df.groupby("sex")["survived"].mean()
"""
output:
sex
female    0.742038
male      0.188908

An Inference: Being woman is a important factor for survival.
"""

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),}))

target_summary_with_cat(df, "survived", "sex")

target_summary_with_cat(df, "survived", "pclass")
"""
        TARGET_MEAN
pclass             
1          0.629630
2          0.472826
3          0.242363

An Inference: Being in a high class is an important factor for survival
"""


for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

    """       TARGET_MEAN
sex                
female     0.742038
male       0.188908
          TARGET_MEAN
embarked             
C            0.553571
Q            0.389610
S            0.336957
        TARGET_MEAN
class              
First      0.629630
Second     0.472826
Third      0.242363
       TARGET_MEAN
who               
child     0.590361
man       0.163873
woman     0.756458
      TARGET_MEAN
deck             
A        0.466667
B        0.744681
C        0.593220
D        0.757576
E        0.750000
F        0.615385
G        0.500000
             TARGET_MEAN
embark_town             
Cherbourg       0.553571
Queenstown      0.389610
Southampton     0.336957
       TARGET_MEAN
alive             
no             0.0
yes            1.0
          TARGET_MEAN
survived             
0                 0.0
1                 1.0
        TARGET_MEAN
pclass             
1          0.629630
2          0.472826
3          0.242363
       TARGET_MEAN
sibsp             
0         0.345395
1         0.535885
2         0.464286
3         0.250000
4         0.166667
5         0.000000
8         0.000000
       TARGET_MEAN
parch             
0         0.343658
1         0.550847
2         0.500000
3         0.600000
4         0.000000
5         0.200000
6         0.000000"""

######################################################
# Analysis of Target Variable with Numeric Variables
######################################################


df.groupby("survived")["age"].mean()
df.groupby("survived").agg(
    {"age": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end = "\n\n\n")

target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)

#endregion

#region Analysis of Correlation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
file_path = r"C:\Users\alioz\OneDrive\Belgeler\datasets\data.csv"
df = pd.read_csv(file_path)
df = df.iloc [:, 1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int,float]]

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

########################################
# Removing of High-Correlation Variables
########################################

corr_matrix = df[num_cols].corr().abs()

upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
corr_matrix.drop(drop_list)
df.drop(drop_list, axis=1)

def high_correlated_cols(dataframe, plot=False, corr_th= 0.90):
    corr = dataframe.corr()
    corr_matrix = dataframe[num_cols].corr().abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

    drop_list = [col for col in upper_triangle_matrix if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize':(12,12)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


drop_list =  high_correlated_cols(df[num_cols], plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

# Let's try it 600 mb sized data set which has more 300 variables.

file_path = r"C:\Users\alioz\OneDrive\Belgeler\datasets\train_transaction.csv"
df = pd.read_csv(file_path)
len(df.columns)
df.head()

drop_list =  high_correlated_cols(df, plot=True)
len(df.drop(drop_list, axis=1).columns)
#endregion