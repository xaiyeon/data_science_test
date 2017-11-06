# Following Tutorial Online~
# By Royce A.

import numpy
import sklearn
from sklearn.datasets import make_moons
from sklearn import tree

## [height weight and shoe size]

X = [[155, 100, 7], [200, 160, 10], [165, 120, 9], [177, 150, 9], [180, 190, 8], [150, 172, 9]]

Y = ['female', 'male', 'female', 'male', 'male', 'male']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, Y)

person = [134, 111, 7]
person2 = [170, 189, 9]

# don't include too much brackets...
# Here we predict based on our given data from X and Y
prediction = clf.predict([person])

prediction2 = clf.predict([person2])

print("My first lesson to AI and deep learning...")
print("The gender for {0} is more likely a {1}".format(person, prediction))
print("The gender for {0} is more likely a {1}".format(person2, prediction2))
print("Yay!")
