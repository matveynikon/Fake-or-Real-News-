import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("news.csv")
#print(data.head())

x = np.array(data["title"])
y = np.array(data["label"])

cv = CountVectorizer()
x = cv.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(xtrain, ytrain)
#print(model.score(xtest, ytest))

news_headline = "Shrek starts WW3"
data = cv.transform([news_headline]).toarray()
print("\n\n\n\n\n\n\n" + news_headline)
print(model.predict(data))
print("\n\n\n")
#print(model.score(model.predict(data), news_headline))