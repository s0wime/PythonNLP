import re
import warnings
import numpy as np
import pandas as pd
import nltk
import gensim
from gensim import corpora
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

warnings.filterwarnings('ignore')

# =====================================================================
# ЗАВДАННЯ 1: LSI моделювання тем
# =====================================================================

df = pd.read_csv('news3.csv', index_col=0)
print(f"Розмір датасету: {df.shape}")
print(f"Кількість міток: {df['label'].nunique()}\n")

wpt = nltk.WordPunctTokenizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = wpt.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens

df['tokens'] = df['text'].apply(preprocess)
corpus_tokens = df['tokens'].tolist()

dictionary = corpora.Dictionary(corpus_tokens)
dictionary.filter_extremes(no_below=5, no_above=0.7)
bow_corpus = [dictionary.doc2bow(doc) for doc in corpus_tokens]

print(f"Розмір словника: {len(dictionary)}")
print(f"Кількість документів: {len(bow_corpus)}\n")

TOTAL_TOPICS = 10

print("=" * 60)
print("1) ПРИХОВАНЕ СЕМАНТИЧНЕ ІНДЕКСУВАННЯ (LSI)")
print("=" * 60)

lsi_model = gensim.models.LsiModel(
    bow_corpus, id2word=dictionary,
    num_topics=TOTAL_TOPICS, onepass=True,
    chunksize=len(bow_corpus), power_iters=1000
)

# а) Терми, що описують теми
print("\nа) Терми, що описують теми:")
for idx, topic in lsi_model.print_topics(num_topics=TOTAL_TOPICS, num_words=8):
    print(f"  Тема {idx}: {topic}")

# б) Документи з найбільшим вкладом у теми
print("\nб) Документи з найбільшим вкладом у кожну тему:")
doc_topics = [lsi_model[bow] for bow in bow_corpus]

for topic_id in range(TOTAL_TOPICS):
    scores = []
    for doc_idx, doc_topic in enumerate(doc_topics):
        topic_dict = dict(doc_topic)
        score = abs(topic_dict.get(topic_id, 0.0))
        scores.append((doc_idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_doc_idx = scores[0][0]
    top_score = scores[0][1]
    text_preview = df['text'].iloc[top_doc_idx][:100]
    print(f"  Тема {topic_id} -> Doc {top_doc_idx} (score: {top_score:.4f}): {text_preview}...")

# в) Три нових документи
print("\nв) Визначення тем для нових документів:")
new_docs = [
    "Apple announced a new iPhone with improved camera and battery life, available next month worldwide.",
    "The Federal Reserve raised interest rates by 0.5 percent amid rising inflation and economic concerns.",
    "Scientists discovered a new species of deep-sea fish near the Mariana Trench using advanced submersibles."
]

for i, doc in enumerate(new_docs):
    tokens = preprocess(doc)
    bow = dictionary.doc2bow(tokens)
    topics = lsi_model[bow]
    topics_sorted = sorted(topics, key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Новий документ {i+1}: \"{doc[:80]}...\"")
    print(f"  Топ-3 теми:")
    for topic_id, score in topics_sorted[:3]:
        print(f"    Тема {topic_id} (score: {score:.4f})")

# =====================================================================
# ЗАВДАННЯ 2: Ключові біграми з austen-persuasion.txt
# =====================================================================

print("\n" + "=" * 60)
print("2) КЛЮЧОВІ БІГРАМИ — austen-persuasion.txt")
print("=" * 60)

raw_text = nltk.corpus.gutenberg.raw('austen-persuasion.txt')
print(f"\nДовжина тексту: {len(raw_text)} символів")

tokens = wpt.tokenize(raw_text.lower())
tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]
print(f"Кількість токенів після обробки: {len(tokens)}")

bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(tokens)
finder.apply_freq_filter(5)

print("\nТоп-20 біграм за поточковою взаємною інформацією (PMI):")
for bigram in finder.nbest(bigram_measures.pmi, 20):
    print(f"  {bigram[0]} {bigram[1]}")

print("\nТоп-20 біграм за частотою:")
for bigram, freq in finder.ngram_fd.most_common(20):
    print(f"  {bigram[0]} {bigram[1]} — {freq} разів")

print("\nТоп-20 біграм за хі-квадрат:")
for bigram in finder.nbest(bigram_measures.chi_sq, 20):
    print(f"  {bigram[0]} {bigram[1]}")
