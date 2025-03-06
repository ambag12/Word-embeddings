import numpy as np
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

def extract_emotional_incentives(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

text1 = """A corollary argument is that exposure to insensitive speech can lead people to question their prior beliefs and biases.
A corollary argument is that exposure to insensitive speech can lead people to question their prior beliefs and biases. Depending on the beliefs questioned, this can be judged good or bad, but it is certain that the capacity to question one’s biases is a necessary condition for distinguishing what may need to be changed and what should be conserved in society."""
text2 = """Comedy was long a favorite way to challenge popular beliefs, perhaps starting with Aristophanes’s “Lysistrata,” which imagined a women’s sex strike to stop wars.
Responding to the leader of the chorus of old men, the chorus of women is pretty insensitive."""

corpus1 = preprocess_text(text1)
corpus2 = preprocess_text(text2)

w2v_model = Word2Vec(sentences=[corpus1, corpus2], vector_size=100, window=5, min_count=1, workers=4, sg=1)

def vectorize_text(text, model, vector_size):
    vectors = [model.wv[word] for word in text if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)  
    return np.mean(vectors, axis=0)

corpus_vector1 = vectorize_text(corpus1, w2v_model, 100)
corpus_vector2 = vectorize_text(corpus2, w2v_model, 100)

X = np.array([corpus_vector1, corpus_vector2])  
y = np.array([1, 0]) 
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X, y)

y_pred = classifier.predict(X)

sentiment1 = extract_emotional_incentives(text1)
sentiment2 = extract_emotional_incentives(text2)

# Print Results
print("Predicted labels:", y_pred)
print("\nEmotional Incentives:")
print("Text 1 Sentiment:", sentiment1)
print("Text 2 Sentiment:", sentiment2)
