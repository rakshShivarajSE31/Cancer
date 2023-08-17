from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

news = fetch_20newsgroups()

print(news.target_names)

cat = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

train = fetch_20newsgroups(subset='train', categories=cat)
test = fetch_20newsgroups(subset='test', categories=cat)

model = make_pipeline( TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
pred = model.predict(test.data)
output1 = model.score(test.data, test.target)

from sklearn.metrics import accuracy_score
print(output1)
print(model.predict(['space']))
results=int(model.predict(['space']))
print(cat[results])