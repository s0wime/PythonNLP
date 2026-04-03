import re
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from gensim.models import word2vec

with open("doc1.txt", "r", encoding="utf-8") as f:
    raw_corpus = [line.strip() for line in f.readlines() if line.strip()]

print("КОРПУС ДОКУМЕНТІВ:")
for i, doc in enumerate(raw_corpus):
    print(f"  Doc {i+1}: {doc}")

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def preprocess(doc):
    doc = doc.lower()
    doc = re.sub(r'[^a-z\s]', '', doc)
    tokens = wpt.tokenize(doc)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

corpus = [preprocess(doc) for doc in raw_corpus]

print("\nПОПЕРЕДНЬО ОБРОБЛЕНИЙ КОРПУС:")
for i, doc in enumerate(corpus):
    print(f"  Doc {i+1}: {doc}")

# ===================================================================
# 1) Bag of Words
# ===================================================================
print("\n" + "=" * 60)
print("1) МОДЕЛЬ «СУМКА СЛІВ» (Bag of Words)")
print("=" * 60)

cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(corpus)
vocab = cv.get_feature_names_out()

bow_df = pd.DataFrame(cv_matrix.toarray(), columns=vocab,
                       index=[f"Doc {i+1}" for i in range(len(corpus))])
print("\nТаблиця Bag of Words:")
print(bow_df.to_string())

if 'film' in vocab:
    film_idx = list(vocab).index('film')
    film_vector = cv_matrix.toarray()[:, film_idx]
    print(f"\nВектор для слова 'film': {film_vector}")
else:
    print("\nСлово 'film' відсутнє у словнику після попередньої обробки.")

# ===================================================================
# 2) TF-IDF + ієрархічна кластеризація
# ===================================================================
print("\n" + "=" * 60)
print("2) МОДЕЛЬ TF-IDF + КЛАСТЕРИЗАЦІЯ")
print("=" * 60)

tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2',
                     use_idf=True, smooth_idf=True)
tv_matrix = tv.fit_transform(corpus)
tfidf_vocab = tv.get_feature_names_out()

tfidf_df = pd.DataFrame(tv_matrix.toarray(), columns=tfidf_vocab,
                         index=[f"Doc {i+1}" for i in range(len(corpus))])
print("\nТаблиця TF-IDF:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
print(tfidf_df.to_string())

ag = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
ag.fit(tv_matrix.toarray())

print("\nРезультати ієрархічної агломераційної кластеризації (3 кластери):")
for i, label in enumerate(ag.labels_):
    print(f"  Doc {i+1} -> Кластер {label + 1}: {raw_corpus[i][:70]}...")

# ===================================================================
# 3) Word2Vec
# ===================================================================
print("\n" + "=" * 60)
print("3) МОДЕЛЬ Word2Vec")
print("=" * 60)

tokenized_corpus = [wpt.tokenize(doc.lower()) for doc in raw_corpus]
tokenized_corpus = [[re.sub(r'[^a-z]', '', t) for t in doc if re.sub(r'[^a-z]', '', t)]
                     for doc in tokenized_corpus]

w2v_model = word2vec.Word2Vec(
    tokenized_corpus,
    vector_size=50,
    window=10,
    min_count=1,
    sample=1e-3,
    epochs=500
)

for target_word in ['shrimp', 'economy']:
    if target_word in w2v_model.wv:
        vec = w2v_model.wv[target_word]
        print(f"\nВектор слова '{target_word}' (перші 10 значень): {vec[:10]}")
        similar = w2v_model.wv.most_similar(target_word, topn=5)
        print(f"Подібні слова до '{target_word}':")
        for word, score in similar:
            print(f"  {word:20s} (подібність: {score:.4f})")
    else:
        print(f"\nСлово '{target_word}' відсутнє у словнику моделі.")
