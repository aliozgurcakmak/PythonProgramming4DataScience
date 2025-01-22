##################################################
# Functions, Conditions, Loops, Comprehensions
##################################################
# - Functions
# - Conditions
# - Loops
# - Comprehensions


"""FUNCTIONS"""
from envs.myEnv.Lib.symbol import continue_stmt

#region Functions

# Functions
print("a", "b", sep="__")
# You can query the properties of a function using help(functionName) or ?functionName

# Defining a Function
def calculate(x):
    print(x*2)

calculate(2)

# Defining a Function with Two Arguments/Parameters
def summer(arg1, arg2):
    print(arg1 + arg2)

summer(4,5)
#endregion

#region Docstring

# Docstring

def summer(arg1, arg2):
    """
    Sum of two numbers

    Args:
        arg1: int, float
        arg2: int, float

    Returns:
        int, float

    Examples:

    Notes:


    """
    print(arg1 + arg2)




#endregion

#region Body Part of Functions

# Body Part of Functions

# def functionName(parameters/arguments):
#   statements (function body)

def say_hi(string):
    print(string)
    print("HRU?")
    print("Hi!")
say_hi("Miuul")


def multiplication(x, y):
    z = x * y
    print(z)

multiplication(10, 8)


# Function to store the entered values in a list

list_store = []
def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)

add_element(9, 25)
add_element(9, 25)
add_element(2, 13)
# now, list stored 225, 225 and 26





#endregion

#region Default Parameters/Arguments
# Default Parameters/Arguments

def divide(a, b):
    print(a / b)
divide(1, 2)

def divide(a, b=3):
    #If we do not use ‘3’ as a predefinition here, an error message will be returned when the user does not enter an argument.
    print(a / b)
divide(10)


def say_hello(string="Hello!"):
    #If we do not use ‘hello’ as a predefinition here, an error message will be returned when the user does not enter an argument.
    print(string)
    print("HRU?")
    print("Hi!")
say_hi()

#endregion

#region When will we need to define functions?
# When will we need to define functions?

""" Assume that we are a municipality emplyoee, and we need to do some operations with datas obtained from street lamps. """
""" The datas are warm, moisture and charge."""
""" For example, warm + moisture / charge will give us a score"""

"""(56 + 15) / 80
(17 + 45) / 70
(52 + 45) / 80"""

# Don't Repeat Yourself (DRY)  principe says, if you have some self-repeat missions then you need define some functions.

def calculate(varm, moisture, charge):
    print("Score of this lamp is: ", (varm + moisture) / charge)
calculate(98, 62, 25)

#endregion

#region Using function outputs as input
# Using function outputs as input

def calculate(varm, moisture, charge):
    print("Score of this lamp is: ", (varm + moisture) / charge)
calculate(98, 62, 25) * 10

"""In here we will get "TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'" error because of NoneType. Then we need use "return". """


def calculate(varm, moisture, charge):
    return (varm + moisture) / charge
calculate(98, 62, 25) * 10

"""Now we want 4 output from process"""
def calculate(varm, moisture, charge):
    varm = varm*2
    moisture = moisture*2
    charge = charge*2
    output =  (varm + moisture) / charge
    return varm, moisture, charge, output

calculate(98, 12, 78)
"""Type of this output is tuple."""
varm, moisture, charge, output = calculate(98, 12, 78)
#endregion

#region Call a Function from within a Function

def calculate (varm, moisture, charge):
    return int((varm + moisture) / charge)
calculate(90, 12 ,12) * 10

def standardization(a, p):
    return a * 10 / 100 * p * p

standardization(45, 1)


"""def all_calculation(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b*10)

    all_calculation(1, 6 , 8, 12)"""

def all_calculation(varm, moisture, charge, a, p):
    print(calculate(varm, moisture, charge))
    b = standardization(a, p)
    print(b * 10)

all_calculation(1, 6, 8, 19 ,12)

#endregion

#region Local and Global Variables

list_store = [1, 2]
type(list_store)

def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)
    """We are on the local area. Since the Append method acts on the global domain, it is called ‘local to global domain’."""

add_element(1, 9)
"""Variable 'C' is not in global area. It is just for add_element's act area."""

#endregion


"""CONDITIONS"""

#region Conditions & If

# Let's remember True-False.
"""1 == 1
2 == 1"""

# If
if 1 == 1:
    print ("something")

if 1 == 4:
    print("something")


number = 11
if number < 8:
    print ("Number is smaller than 8!")

number = 7
if number < 8:
    print("Number is smaller than 8!")

"""Let's create a function to comply with the DRY principle"""
def numberCheck(number):
    if number < 8:
        print("Number is smaller than 8!")
numberCheck(7)

#endregion

#region Else

def numberCheck (number):
    if number == 10:
        print("Number is 10")
    else:
        print("Number is not 10")
numberCheck(12)

#endregion

#region ElIf

def numberCheck (number):
    if number > 10:
        print("Number is greater than 10")
    elif number < 10:
        print("Number is less than 10")
    else:
        print("Number is 10")

numberCheck(7)

#endregion

"""LOOPS"""

#region For Loop

students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]
students[3]

for student in students:
    print(student)

for student in students:
    print(student.upper())


salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

"""Now let's raise these salaries."""
for salary in salaries:
     print(int(salary*0.20 + salary))

for salary in salaries:
     print(int(salary*0.30 + salary))

for salary in salaries:
     print(int(salary*0.50 + salary))

"""DRY PRINCIPE!"""

def new_salary (salary, increaseRate):
    return int(salary*increaseRate/100 + salary)
new_salary(1500, 10)

"""Now apply it on salaries list"""

for salary in salaries:
    print(new_salary(salary, 20))

"""Also we apply it another list."""

salaries2 = [10700, 25000, 30400, 40300, 50200]

for salary in salaries2:
    print(new_salary(salary, 15))

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))







#endregion

"""Application - An Interview Question"""

#region Application
# We want to write a function that changes the string as follows.

# Before: Hi my name is John and i am learning python.
# After: Hi mY NaMe iS JoHn i aM LeArNiNg pYtHoN

range(len("Miuul"))
range(0,5)

for i in range(len("Miuul")):
    print(i)

def alternating(string):
    newString = ""
    # Navigate the indexes of the entered string.
    for stringIndex in range(len(string)):
        # If the index is even, upper this letter.
          if stringIndex % 2 == 0 :
              newString += string[stringIndex].upper()
        # If the index is odd, lower this letter.
          else:
               newString += string[stringIndex].lower()
    # Finally print it.
    print(newString)

alternating("Benim adım Ali Özgür Çakmak")

#endregion

"""BREAK, CONTINUE, WHILE & others"""

#region

salaries [1000, 2000, 3000, 4000, 5000]

#region Break
for salary in salaries:
    if salary == 3000:
        break
    print(salary)
#endregion

# region Continue
for salary in salaries:
    if salary == 3000:
        continue
    print(salary)
#endregion

#region While

number = 1
while number < 5:
    print(number)
    number += 1
    
#endregion

#region Enumerate

students = ["John", "Mark", "Venessa", "Mariam"]

"""Enumerate brings also index with every element."""
for student in students:
    print(student)

for index, student in enumerate(students):
    print(index+1, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)
print(A, B)


#endregion

"""Application - An Interview Question | 2"""

#region Application 2
# Define the divide_students
# Keep a list of students with double indexes.
# Move single-indexed students to another list.
# But these 2 lists have to return as just one list.

students = ["Özgür", "Berkan", "Aras", "İlke"]

def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups

st = divide_students(students)
#endregion

#region Define 'alternating' Function with Enumerate

def alternatingWithEnumerate (string):
    new_string = ""
    for i, letter in enumerate(string):
        if i %2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

alternatingWithEnumerate("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas finibus fringilla massa ut viverra. Maecenas malesuada nisi ac magna congue ullamcorper.")

#endregion

"""---"""

#region Zip

departments = ["Engineering", "Mathematics", "Physics", "Literature", "Biology"]
students = ["Ahmet", "Mehmet", "Elif", "Ayşe", "Fatma"]  
ages = [22, 25, 36, 27, 33]

list(zip(students, departments, ages))

#endregion

"""lambda, map, filter, reduce"""

# Map: The map function applies a given function to all items in an input list (or any iterable).
# Filter: The filter function applies a given function to an iterable to filter out elements that do not satisfy the condition.
# Reduce: The reduce function applies a rolling computation to sequential pairs of values in an iterable (from the functools module).

#region Lambda

"""Lambda: Lambda allows the creation of small anonymous functions. These functions are defined using the 'lambda' keyword."""

def summer(a, b):
    return a + b
summer(1, 3) * 9

newSum = lambda a, b: a + b
newSum(4 ,5)

#endregion

#region Map

salaries = [1000, 2000, 3000, 4000, 5000]

def newSalary(x):
     return x * 20 / 100 + x

newSalary(500)

for salary in salaries:
    print(newSalary(salary))


list(map(lambda x: x * 20 / 100 + x, salaries))
# del newSum

list(map(lambda x: x**2, salaries))

#endregion

#region Filter

list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))
#endregion

#region Reduce

from functools import reduce
list_store = [1, 2, 3, 4]

reduce(lambda a, b: a + b, list_store)

#endregion

"""COMPREHENSIONS"""

#region List Comprehension

salaries = [1000, 2000, 3000, 4000, 5000]
def newSalary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(newSalary(salary))

nullList = []

for salary in salaries:
 if salary > 3000:
        nullList.append(newSalary(salary))
 else:
        nullList.append(newSalary(salary*2))

[newSalary(salary*2) if salary < 3000 else newSalary(salary) for salary in salaries]

[salary * 2 for salary in salaries]

[salary * 2 for salary in salaries if salary < 3000]

[salary * 2 if salary < 3000 else salary * 0 for salary in salaries ]

[newSalary(salary * 2) if salary < 3000 else newSalary(salary * 0.2)  for salary in salaries]

students = ["John", "Mark", "Venessa", "Mariam"]
studentsNo = ["John", "Venessa"]

[student.lower() if student in studentsNo else student.upper() for student in students]

#endregion

#region Dict Comprehension

dictionary = {"a": 1,
              "b": 2,
              "c": 3,
              "d": 4}

dictionary.keys()
dictionary.values()
dictionary.items()

{k: v ** 2 for (k,v) in dictionary.items()}
{k.upper(): v ** 2 for (k,v) in dictionary.items()}

#endregion

"""Application - An Interview Question | 3"""

#region Application 3

# You need to take squares of even numbers and add to dictionary them.
# Keys will be original values and values will be changed.


numbers = range(10)
newDict = {}

for n in numbers:
    if n % 2 == 0:
        newDict[n] = n ** 2

{n: n ** 2 for n in numbers if n % 2 == 0}

#endregion

"""Some Problems"""

#region Comprehension Problems

##region Change the variable names of a data set.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

df.columns = [col.upper() for col in df.columns]

##endregion

##region We want to add "FLAG" as a prefix to variables that contain "INS" in their names and "NOFLAG" as a prefix to all other variables.

[col for col in df.columns if "INS" in col]

["FLAG_"  + col for col in df.columns if "INS" in col]

["FLAG_" + col if "INS" in col else "UNFLAG_" + col for col in df.columns]

df.columns = ["FLAG_" + col if "INS" in col else "UNFLAG_" + col for col in df.columns]

##endregion

##region To create a dictionary where the key is a string and the value is a list as shown below.

"""Output:
{"total": ["mean", "min", "max", "var"]}
{"speeding": ["mean", "min", "max", "var"]}
{"alcohol": ["mean", "min", "max", "var"]}
{"not_distracted": ["mean", "min", "max", "var"]}
{"no_previous": ["mean", "min", "max", "var"]}
{"ins_premium": ["mean", "min", "max", "var"]}
{"ins_losses": ["mean", "min", "max", "var"]}"""

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

num_cols = [col for col in df.columns if df[col].dtype != "O"]
dict1 = {}
agg_list = ["mean", "min", "max", "sum"]

for col in num_cols:
    dict1[col] = agg_list

#### Shortcut
newDict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(newDict)

##endregion

#endregion










