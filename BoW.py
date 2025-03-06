from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
def BoW(text,desired_word):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text)
    features= vectorizer.get_feature_names_out()
    if desired_word in features:
        word_index = np.where(features == desired_word)[0][0]  
        word_scores = X[:, word_index].toarray().ravel()
        print('word_scores:',word_scores)
    return X

a=BoW(text=["this is text, not file or filename file"],desired_word="file")
print(a)
