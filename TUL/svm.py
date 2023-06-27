from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score , f1_score , precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
def lda(file_path,lda_type):
  df = pd.read_csv(file_path)
  input_seq = np.array(df['poi'])
  vectorizer = CountVectorizer()
  bag = vectorizer.fit_transform(input_seq)
  X_train, X_test, y_train, y_test = train_test_split(bag, df['user'], test_size=0.2,stratify=df['user'])
  clf = SVC(probability=True,kernel='linear')
  clf.fit(X_train.toarray(),y_train )
  #clf.predict_proba(
  top1 = top_k_accuracy_score(y_test,clf.predict_proba(X_test.toarray()), k=1)
  top5 = top_k_accuracy_score(y_test,clf.predict_proba(X_test.toarray()), k=5)
  results = clf.predict(X_test.toarray())
  f1_score_macro = f1_score(y_test, results, average='macro')
  r_score = recall_score(y_test, results, average='macro')
  p_score = precision_score(y_test, results, average='macro')
  print(file_path)
  print(lda_type)
  print(f"top@1 = {top1} | top@5 = {top5} | macro-P={p_score} | macro-R{r_score} |f1_score_macro={f1_score_macro}" )

files = ['dataset/nyc_108.csv','dataset/nyc_209.csv','dataset/nyc_full_set.csv','dataset/tky_108.csv','dataset/tky_209.csv','dataset/tky_full_set.csv']
#
types = ['svd']

for i in files:
  for j in types:
    lda(i,j)