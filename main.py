import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# extract the data and get the bare/ folder data and put it in a dataset/ folder
RES_DIR = os.path.join(os.path.dirname(__name__),'dataset')
# create a dict to load the mails
mails = {'msg':[],'label':[]}
for mail in os.listdir(RES_DIR):
    # get the body of the message
    message = open(os.path.join(RES_DIR, mail)).read().splitlines()
    # get the label
    # let's use -1 for spam and +1 to ham
    label = -1 if 'spmsg' in mail else 1
    # append it to the dict
    mails['msg'].append(message)
    mails['label'].append(label)

# let's build a big dataframe of the data
dataset = pd.DataFrame(mails)
# let's view it
print(dataset.label.value_counts())
dataset.head()


def clean_str(text):
    new_text = []
    for word in text.split():
        if word.isalpha():
            new_text.append(word)
    new_text = ' '.join(new_text)
    return new_text

vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(dataset['msg'])

message = dataset['msg'][0]
print("mail befor :", message)
print ("---------------")
features = vectorizer.transform([message])
print("mail after:", features)


features_set = vectorizer.transform(dataset['msg'])

pd.DataFrame(features_set.toarray()).head()

x_train, x_test, y_train, y_test = train_test_split(features_set.toarray(), dataset['label'].values, test_size=.25)

clfs = {
    'Naive_bayes': GaussianNB(),
    'SVM': SVC(),
    'Decision_tree': DecisionTreeClassifier(),
    'gradient_descent': SGDClassifier()
}

for clf_name in clfs.keys():
    print("now traing",clf_name,"classifier")
    clf = clfs[clf_name]
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    print(classification_report(y_test, y_predict))
    print()
