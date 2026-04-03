import re
import warnings
import numpy as np
import pandas as pd
import nltk
from gensim.models import word2vec
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

df = pd.read_csv('bbc-news-data1.csv', index_col=0)
print(f"Розмір датасету: {df.shape}")
print(f"Категорії:\n{df['category'].value_counts()}\n")

wpt = nltk.WordPunctTokenizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = wpt.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

df['clean'] = df['content'].apply(preprocess)

train_corpus, test_corpus, train_labels, test_labels = train_test_split(
    df['clean'], df['category'], test_size=0.3, random_state=42, stratify=df['category']
)
print(f"Навчальна вибірка: {train_corpus.shape[0]}, Тестова: {test_corpus.shape[0]}\n")

# =====================================================================
# TF-IDF features
# =====================================================================
tv = TfidfVectorizer(min_df=0.0, max_df=1.0, norm='l2', use_idf=True, smooth_idf=True)
tfidf_train = tv.fit_transform(train_corpus)
tfidf_test = tv.transform(test_corpus)

# =====================================================================
# Word2Vec features
# =====================================================================
tokenized_train = [wpt.tokenize(doc) for doc in train_corpus]
tokenized_test = [wpt.tokenize(doc) for doc in test_corpus]

W2V_SIZE = 100
w2v_model = word2vec.Word2Vec(
    tokenized_train, vector_size=W2V_SIZE, window=10,
    min_count=2, sample=1e-3, epochs=50
)

def document_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    def average_word_vectors(words):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        for word in words:
            if word in vocabulary:
                nwords += 1.
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)
        return feature_vector
    return np.array([average_word_vectors(doc) for doc in corpus])

w2v_train = document_vectorizer(tokenized_train, w2v_model, W2V_SIZE)
w2v_test = document_vectorizer(tokenized_test, w2v_model, W2V_SIZE)

results = {}

# =====================================================================
# 1. TF-IDF + Naive Bayes (baseline)
# =====================================================================
print("=" * 60)
print("1) TF-IDF + Naive Bayes")
print("=" * 60)
mnb = MultinomialNB()
mnb.fit(tfidf_train, train_labels)
pred = mnb.predict(tfidf_test)
acc = accuracy_score(test_labels, pred)
results['TF-IDF + NB'] = acc
print(f"Accuracy: {acc:.4f}")
print(classification_report(test_labels, pred))

# =====================================================================
# 2. TF-IDF + SVM (baseline)
# =====================================================================
print("=" * 60)
print("2) TF-IDF + SVM")
print("=" * 60)
svm = LinearSVC(max_iter=5000)
svm.fit(tfidf_train, train_labels)
pred = svm.predict(tfidf_test)
acc = accuracy_score(test_labels, pred)
results['TF-IDF + SVM'] = acc
print(f"Accuracy: {acc:.4f}")
print(classification_report(test_labels, pred))

# =====================================================================
# 3. Word2Vec + Naive Bayes (baseline) - GaussianNB for continuous features
# =====================================================================
from sklearn.naive_bayes import GaussianNB

print("=" * 60)
print("3) Word2Vec + Naive Bayes (GaussianNB)")
print("=" * 60)
gnb = GaussianNB()
gnb.fit(w2v_train, train_labels)
pred = gnb.predict(w2v_test)
acc = accuracy_score(test_labels, pred)
results['W2V + NB'] = acc
print(f"Accuracy: {acc:.4f}")
print(classification_report(test_labels, pred))

# =====================================================================
# 4. Word2Vec + SVM (baseline)
# =====================================================================
print("=" * 60)
print("4) Word2Vec + SVM")
print("=" * 60)
svm2 = LinearSVC(max_iter=5000)
svm2.fit(w2v_train, train_labels)
pred = svm2.predict(w2v_test)
acc = accuracy_score(test_labels, pred)
results['W2V + SVM'] = acc
print(f"Accuracy: {acc:.4f}")
print(classification_report(test_labels, pred))

# =====================================================================
# 5. TF-IDF + NB — GridSearchCV
# =====================================================================
print("=" * 60)
print("5) TF-IDF + Naive Bayes — GridSearchCV")
print("=" * 60)
mnb_pipe = Pipeline([('tfidf', TfidfVectorizer()), ('mnb', MultinomialNB())])
param_grid_mnb = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_df': [0.9, 1.0],
    'mnb__alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0]
}
gs_mnb = GridSearchCV(mnb_pipe, param_grid_mnb, cv=5, scoring='accuracy', n_jobs=-1)
gs_mnb.fit(train_corpus, train_labels)
pred = gs_mnb.predict(test_corpus)
acc = accuracy_score(test_labels, pred)
results['TF-IDF + NB (Grid)'] = acc
print(f"Best params: {gs_mnb.best_params_}")
print(f"Accuracy: {acc:.4f}")
print(classification_report(test_labels, pred))

# =====================================================================
# 6. TF-IDF + SVM — GridSearchCV
# =====================================================================
print("=" * 60)
print("6) TF-IDF + SVM — GridSearchCV")
print("=" * 60)
svm_pipe = Pipeline([('tfidf', TfidfVectorizer()), ('svm', LinearSVC(max_iter=5000))])
param_grid_svm = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_df': [0.9, 1.0],
    'svm__C': [0.1, 1.0, 10.0]
}
gs_svm = GridSearchCV(svm_pipe, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
gs_svm.fit(train_corpus, train_labels)
pred = gs_svm.predict(test_corpus)
acc = accuracy_score(test_labels, pred)
results['TF-IDF + SVM (Grid)'] = acc
print(f"Best params: {gs_svm.best_params_}")
print(f"Accuracy: {acc:.4f}")
print(classification_report(test_labels, pred))

# =====================================================================
# 7. Word2Vec + NB — GridSearchCV
# =====================================================================
print("=" * 60)
print("7) Word2Vec + Naive Bayes — GridSearchCV")
print("=" * 60)
param_grid_gnb = {
    'var_smoothing': np.logspace(-12, -6, 7)
}
gs_gnb = GridSearchCV(GaussianNB(), param_grid_gnb, cv=5, scoring='accuracy', n_jobs=-1)
gs_gnb.fit(w2v_train, train_labels)
pred = gs_gnb.predict(w2v_test)
acc = accuracy_score(test_labels, pred)
results['W2V + NB (Grid)'] = acc
print(f"Best params: {gs_gnb.best_params_}")
print(f"Accuracy: {acc:.4f}")
print(classification_report(test_labels, pred))

# =====================================================================
# 8. Word2Vec + SVM — GridSearchCV
# =====================================================================
print("=" * 60)
print("8) Word2Vec + SVM — GridSearchCV")
print("=" * 60)
param_grid_svm2 = {
    'C': [0.1, 1.0, 10.0, 50.0],
    'max_iter': [5000]
}
gs_svm2 = GridSearchCV(LinearSVC(), param_grid_svm2, cv=5, scoring='accuracy', n_jobs=-1)
gs_svm2.fit(w2v_train, train_labels)
pred = gs_svm2.predict(w2v_test)
acc = accuracy_score(test_labels, pred)
results['W2V + SVM (Grid)'] = acc
print(f"Best params: {gs_svm2.best_params_}")
print(f"Accuracy: {acc:.4f}")
print(classification_report(test_labels, pred))

# =====================================================================
# ПОРІВНЯННЯ
# =====================================================================
print("\n" + "=" * 60)
print("ПОРІВНЯННЯ ТОЧНОСТІ УСІХ МОДЕЛЕЙ")
print("=" * 60)
print(f"\n{'Модель':<30} {'Accuracy':>10}")
print("-" * 42)
for name, acc in results.items():
    marker = " *" if "Grid" in name else ""
    print(f"{name:<30} {acc:>10.4f}{marker}")
print("\n* — після GridSearchCV")
