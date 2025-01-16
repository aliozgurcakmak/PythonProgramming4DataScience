#######################################################
# DATA STRUCTURES
#######################################################
# - Introduction to Data Structures
# - Numbers
# - Strings
# - Boolean
# - List
# - Dictionary
# - Tuple
# - Set


#region Introduction to Data Structures

# - Introduction to Data Structures


# Numbers: integer
x = 46
type(x)

# Numbers: float
x = 10.9
type(x)

# Numbers: complex
x = 2j+9
type(x)


# String
x = "Hello AI Era"
type(x)


# Boolean
True
False
type(True)
5 == 4
3 == 2
1 == 1
type(3==2)


# List
x = ["try", "usd", "eur"]
type(x)
print(x)


# Dictionary
x = {"Name": "Ozgur", "Age": 17}
type(x)


# Tuple
x =("python", "java", "csharp")
type(x)


# Set
x = {"python", "java", "csharp"}
type(x)

## NOTE = List, Tuple, Set and Dictionary structures called Python Collections(Arrays).
#endregion

#region Numbers

# Numbers


a = 5
b = 10.5

a * 3
a / 7
a * b / 10
a ** 2


# Changing the types
int(b)
float(a)
int(a * b / 10)
c = a * b / 10
int(c)



#endregion

#region Strings

# Strings

print("John")
print('John')


longStr = """Uyanın, vurgun var!
Görmüyor musun ahali, çarpıyor iki kalp!
Ama mı oldu zehir gözleriniz?
Hiç mi atmadı bu denli nabzınız?
Hiç mi sevmedi böyle şahsınız?"""

name = "Chris"
name[0]
name[2]

# Slicing
name[0:2]
longStr[0:28]


# Querying a charachter
"uyanın" in longStr
"Uyanın" in longStr
            #-- Python is case sensitive. It pays attention to uppercase and lowercase letters


# String Methods
dir(int)
dir(str)

# len function: returns the size of the expression entered into it.
name = "John"
type(len)
len(name)
len("aliozgurcakmak")

# upper, lower: small-large charachter transformations
"miuul".upper()
"MIUUL".lower()

# replace: character changes
hi = "Hello AI Era"
hi.replace("l","p")

# split: splitting process
"Hello AI Era".split()
"Arkadaş, yurdumu alçaklara uğratma sakın.".split("a")

# strip: cropping process
" Hello AI Era ".strip()
"Arkadaş, yurdumu alçaklara uğratma sakın. A".strip("A")

# capitalize: enlarges the first letter
"hello".capitalize()

# startswith: checks if it starts with the entered argument
"Clementine".startswith("C")


#endregion

#region Lists
#######################################################
# Lists
#######################################################

# - They can be changed.
# - They are sorted.
# - They are inclusive.

notes = [1,2,3,4]
type(notes)

strings = ["a","b","c","d"]

not_nam = [1,2,3,"a","b","c",True,[1,2,3]]
not_nam[6]
not_nam[7][1]
notes[0]
notes[0] = 99
not_nam[0:4]



# List Methods
dir(notes)
len(notes)

# append: adds element
notes.append(100)
notes

# pop: removes by index
notes.pop(0)
notes

# insert = adds to index
notes.insert(0,100)
notes


#endregion

#region Dictionaries
#######################################################
# - Dictionary
#######################################################

# They can be changed.
# They aren't sorted. (After 3.7, they can be sorted.)
# They are inclusive.
# key-value

dictionary = {"Name": "Ozgur",
              "Age": 17,
              "Department": "Management Information Systems"}
dictionary["Department"]


dictionary = {"Name": ["Ali", "Ozgur"],
              "Age": 17,
              "Department": ["Software Engineering", "Management Information Systems"]}
dictionary["Department"][0]


# Querying a key
"Name" in dictionary
"Surname" in dictionary

# Accessing Value by Key
dictionary["Name"]
dictionary.get("Name")[1]

# Value Change
dictionary["Name"] = ["Mustafa", "Kemal"]

# Accessing all keys
dictionary.keys()

# Accessing all values
dictionary.values()

# Convert all pairs to a list in tuple format
dictionary.items()

# Updating key-value values
dictionary.update({"Name": ["Ata"]})

# Adding new key-value values
dictionary.update({"Surname": "Atadan"})



#endregion

#region Tuples

# Tuple

# - They can not be changed.
# - They are sorted.
# - They are inclusive.

t = ("john", "mark", "2", True)
type(t)
t[0]
t[0:2]
t[0] = 99
# Tuple can not be changed.

t = list(t)
t[0] = 99
t = tuple(t)
# In this way, tuple elements can be changed by first converting them to list format. But it is generally not preferred.


#endregion

#region Sets

# Set

# - They can be changed.
# - They are unique and not sorted.
# - They are inclusive.

# difference(): difference of two sets
set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

#in set1 but not in set2
set1.difference(set2)
set1 - set2

#in set2 but not in set1
set2.difference(set1)
set2 - set1

# symmetric_difference(): not relative to each other in both sets
set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

# intersection(): intersection of two sets
set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

set1.intersection(set2)
set2.intersection(set1)
set1 & set2

# union(): union of two sets
set1.union(set2)
set2.union(set1)

# isdisjoint(): is intersection of two sets null?
set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])

set1.isdisjoint(set2)

# isdissubset(): is one set a subset of another set?
set1.issubset(set2)
set2.issubset(set1)

# issuperset(): is one set a superset of another set?
set1.issubset(set2)
set2.issubset(set1)

#endregion

