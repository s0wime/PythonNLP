import warnings
import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

df = pd.read_csv('twitter1.csv', index_col=0)
print(f"Розмір датасету: {df.shape}")
print(f"Розподіл класів:\n{df['category'].value_counts()}\n")

label_map = {-1.0: 'negative', 0.0: 'neutral', 1.0: 'positive'}
df['label'] = df['category'].map(label_map)

# =====================================================================
# 1) Попередня обробка за допомогою spaCy
# =====================================================================
print("=" * 60)
print("1) Попередня обробка за допомогою spaCy")
print("=" * 60)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_spacy(text):
    doc = nlp(str(text))
    tokens = [token.lemma_.lower() for token in doc
              if not token.is_stop and not token.is_punct
              and not token.is_space and len(token.text) > 2]
    return ' '.join(tokens)

print("Лематизація та очищення (це може зайняти хвилину)...")
df['processed'] = df['clean_text'].apply(preprocess_spacy)

print(f"Приклад обробки:")
for i in range(3):
    print(f"  До:    {df['clean_text'].iloc[i][:80]}")
    print(f"  Після: {df['processed'].iloc[i][:80]}\n")

# =====================================================================
# 2) Логістична регресія
# =====================================================================
print("=" * 60)
print("2) Аналіз настроїв — Логістична регресія")
print("=" * 60)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['processed'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)
print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

tv = TfidfVectorizer(min_df=5, max_df=0.95, ngram_range=(1, 2), sublinear_tf=True)
train_features = tv.fit_transform(train_texts)
test_features = tv.transform(test_texts)

lr = LogisticRegression(penalty='l2', max_iter=1000, C=1, random_state=42)
lr.fit(train_features, train_labels)
lr_pred = lr.predict(test_features)

lr_acc = accuracy_score(test_labels, lr_pred)
print(f"\nТочність логістичної регресії: {lr_acc:.4f}")
print(f"\nМатриця невідповідностей:")
cm = confusion_matrix(test_labels, lr_pred, labels=['negative', 'neutral', 'positive'])
print(pd.DataFrame(cm, index=['negative', 'neutral', 'positive'],
                       columns=['pred_neg', 'pred_neu', 'pred_pos']))
print(f"\nClassification report:")
print(classification_report(test_labels, lr_pred))

# =====================================================================
# 3) TextBlob лексикон
# =====================================================================
print("=" * 60)
print("3) Аналіз настроїв — TextBlob (лексикон)")
print("=" * 60)

test_texts_original = df.loc[test_texts.index, 'clean_text']

def textblob_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

tb_pred = test_texts_original.apply(textblob_sentiment)

tb_acc = accuracy_score(test_labels, tb_pred)
print(f"\nТочність TextBlob: {tb_acc:.4f}")
print(f"\nМатриця невідповідностей:")
cm_tb = confusion_matrix(test_labels, tb_pred, labels=['negative', 'neutral', 'positive'])
print(pd.DataFrame(cm_tb, index=['negative', 'neutral', 'positive'],
                          columns=['pred_neg', 'pred_neu', 'pred_pos']))
print(f"\nClassification report:")
print(classification_report(test_labels, tb_pred))

# =====================================================================
# 4) Три випадкові записи — порівняння
# =====================================================================
print("=" * 60)
print("4) Три випадкові записи — порівняння оцінок")
print("=" * 60)

np.random.seed(42)
sample_indices = np.random.choice(test_texts.index, size=3, replace=False)

for idx in sample_indices:
    original = df.loc[idx, 'clean_text']
    actual = df.loc[idx, 'label']
    processed = df.loc[idx, 'processed']

    feature_vec = tv.transform([processed])
    lr_result = lr.predict(feature_vec)[0]

    tb_polarity = TextBlob(str(original)).sentiment.polarity
    tb_result = 'positive' if tb_polarity > 0.1 else ('negative' if tb_polarity < -0.1 else 'neutral')

    print(f"\nТекст: \"{original[:100]}{'...' if len(str(original)) > 100 else ''}\"")
    print(f"  Реальний настрій:      {actual}")
    print(f"  Логістична регресія:   {lr_result}")
    print(f"  TextBlob (polarity={tb_polarity:.3f}): {tb_result}")

# =====================================================================
# Порівняння
# =====================================================================
print("\n" + "=" * 60)
print("ПОРІВНЯННЯ МОДЕЛЕЙ")
print("=" * 60)
print(f"  Логістична регресія: {lr_acc:.4f}")
print(f"  TextBlob (лексикон): {tb_acc:.4f}")
