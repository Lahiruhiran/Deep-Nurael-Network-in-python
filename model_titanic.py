#!/usr/bin/env python
# coding: utf-8

# In[109]:


import pandas as pd
import matplotlib.pyplot as plt


# In[110]:


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# In[111]:


print(train_df.columns)


# In[112]:


train_df.head(800)


# In[ ]:


# In[113]:


train_df.isna().sum()


# In[114]:


median = train_df['Age'].median()
train_df['Age'].fillna(median, inplace=True)


# In[115]:


train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1}).astype(int)


# In[116]:


train_df['Embarked'].fillna("S", inplace=True)


# In[117]:


train_df.isna().sum()


# In[118]:


train_df.head(800)


# In[119]:


train_df['Embarked'] = train_df['Embarked'].map(
    {'S': 0, 'C': 1, 'Q': 2}).astype(int)


# In[120]:


train_df.head(800)


# In[121]:


test_df['Embarked'].fillna("S", inplace=True)


# In[122]:


test_df['Embarked'] = test_df['Embarked'].map(
    {'S': 0, 'C': 1, 'Q': 2}).astype(int)


# In[123]:


test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1}).astype(int)


# In[124]:


median = test_df['Age'].median()
test_df['Age'].fillna(median, inplace=True)


# In[125]:


train_df['Age'].value_counts()


# In[126]:


test_df.head(800)


# In[127]:


test_df.isna().sum()


# In[128]:


train_df.dtypes


# In[129]:


print(train_df.shape)


# In[130]:


train_df.describe()


# In[131]:


print(train_df['Age'].unique())


# In[132]:


print(train_df.groupby('Age').size())


# In[133]:


train_df.drop('Pclass', axis=1).plot(kind='box', subplots=True, layout=(6, 6), sharex=False, sharey=False, figsize=(20, 20),
                                     title='Box Plot for each input variable')
plt.savefig('titanic_box')
plt.show()


# In[134]:


import pylab as pl
train_df.drop('Pclass', axis=1).hist(bins=30, figsize=(9, 9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('Pclass_histogramme')
plt.show()


# In[135]:


train_df[['Pclass', 'Survived']].groupby(
    ['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[136]:


train_df[['Sex', 'Survived']].groupby(
    ['Sex'], as_index=False).mean().sort_values(by='Sex', ascending=False)


# In[137]:


train_df[['Age', 'Survived']].groupby(
    ['Age'], as_index=False).mean().sort_values(by='Age', ascending=False)


# In[138]:


train_df[['Parch', 'Survived']].groupby(
    ['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=False)


# In[139]:


train_df[['SibSp', 'Survived']].groupby(
    ['SibSp'], as_index=False).mean().sort_values(by='SibSp', ascending=False)


# In[140]:


# In[ ]:


# In[166]:


feature_names = ['Age', 'Sex', 'SibSp', 'Parch', 'Pclass']
X = train_df[feature_names]
y = train_df['Survived']


# In[167]:


print(len(y))


# In[168]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[169]:


print(len(X_test))
print(len(y_test))
print(len(X_train))


# Logisstics reggression

# In[170]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
      .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
      .format(logreg.score(X_test, y_test)))


# In[171]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))


# #K-Nearest Neighbors

# In[172]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
      .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
      .format(knn.score(X_test, y_test)))


# Linear Discriminant Analysis

# In[173]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
      .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
      .format(lda.score(X_test, y_test)))


# Gaussian Naive Bayes

# In[174]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
      .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
      .format(gnb.score(X_test, y_test)))


# Support Vector Machine

# In[175]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)

print('Accuracy of SVM classifier on training set: {:.2f}'
      .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
      .format(svm.score(X_test, y_test)))
print(X_test)


# In[176]:


import pickle

# Save the trained model as a pickle string.
saved_model = pickle.dumps(knn)

# Load the pickled model
knn_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions
pred = knn_from_pickle.predict(X_test)
print(pred)


# In[179]:


submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"][:223],
    "Survived": pred
})
submission.to_csv('submission.csv', index=False)
