import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import math

nltk.download('punkt')
nltk.download('punkt_tab')

file = open("data.txt", "r")
text = file.read().lower()
file.close()


#  Tokenization 

tokens = word_tokenize(text)
print("Tokens:\n", tokens)


#  Bigrams 

bigrams = []
for i in range(len(tokens) - 1):
    bigrams.append((tokens[i], tokens[i+1]))

print("\nBigrams:\n", bigrams)


#  MLE Model

bigram_counts = Counter(bigrams)
unigram_counts = Counter([w[0] for w in bigrams])

mle_model = {}

for (w1, w2) in bigram_counts:
    prob = bigram_counts[(w1, w2)] / unigram_counts[w1]
    mle_model[(w1, w2)] = prob

print("\nMLE Model:")
for k, v in mle_model.items():
    print(k, ":", v)


# Laplace Smoothing

vocab = set(tokens)
V = len(vocab)

laplace_model = {}

for (w1, w2) in bigram_counts:
    prob = (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + V)
    laplace_model[(w1, w2)] = prob

print("\nLaplace Model:")
for k, v in laplace_model.items():
    print(k, ":", v)


#  Kneser-Ney 

continuation_counts = Counter([w[1] for w in bigrams])
total_bigrams = len(bigrams)

kneser_model = {}

for (w1, w2) in bigram_counts:
    d = 0.75
    first_part = max(bigram_counts[(w1, w2)] - d, 0) / unigram_counts[w1]
    lambda_w1 = d / unigram_counts[w1]
    cont_prob = continuation_counts[w2] / total_bigrams

    kneser_model[(w1, w2)] = first_part + lambda_w1 * cont_prob

print("\nKneser-Ney Model:")
for k, v in kneser_model.items():
    print(k, ":", v)


# Perplexity

def calculate_perplexity(model, bigrams):
    log_prob = 0
    N = len(bigrams)

    for bg in bigrams:
        prob = model.get(bg, 1e-6)
        log_prob += math.log(prob, 2)

    return 2 ** (-log_prob / N)

print("\nPerplexity:")
print("MLE:", calculate_perplexity(mle_model, bigrams))
print("Laplace:", calculate_perplexity(laplace_model, bigrams))
print("Kneser:", calculate_perplexity(kneser_model, bigrams))


#  Prediction

def predict_next(word, model):
    candidates = {}

    for (w1, w2) in model:
        if w1 == word:
            candidates[w2] = model[(w1, w2)]

    if candidates:
        return max(candidates, key=candidates.get)
    else:
        return "No prediction"

print("\nNext word after 'i':", predict_next("i", mle_model))



#  Prediction

def predict_next(word, model):
    word = word.lower()
    candidates = {}

    for (w1, w2) in model:
        if w1 == word:
            candidates[w2] = model[(w1, w2)]

    if candidates:
        return max(candidates, key=candidates.get)
    else:
        return "No prediction"

