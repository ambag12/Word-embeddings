
from nltk import ngrams
sentence = 'I reside in Bengaluru.'
n = 1
unigrams = ngrams(sentence.split(), n)
for grams in unigrams:
  print(grams)