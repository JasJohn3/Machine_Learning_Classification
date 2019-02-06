from sklearn import tree


#classification of Minivans vs sportscars
#syntax for features
#Features =Engine Horse power, Number of seats
features = [[440,2],[500,2],[190,9],[150,8]]
#Labeled Training Data
#labels = ["Sports-Car","Sports-Car","Minivan","Minivan"]
#0=Sportscar, 1=Minivan
lables=[0,0,1,1]

#create your classifier for the training data
#sci-kit learn Documentation https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
clf =tree.DecisionTreeClassifier()

#Fit is finding patterns in the data
clf=clf.fit(features,lables)

#input unknown data
#HP=160, Number of Seats =7
#Prediction should be a minivan [0]
print(clf.predict([[160,7]]))

#HP=600, number of seats =2
#Prediction should be a sports car [0]
print(clf.predict([[600,2]]))