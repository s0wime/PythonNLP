import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

with open("text1.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

print("=" * 60)
print("ПОЧАТКОВИЙ ТЕКСТ:")
print("=" * 60)
print(text)

print("\n" + "=" * 60)
print("ЗАВДАННЯ 2: Пошук та маскування номерів телефонів")
print("=" * 60)

phone_pattern = re.compile(r'\(?\d[\d\)\-]+\d')

phones = phone_pattern.findall(text)
print(f"\nЗнайдені номери телефонів: {phones}")

def mask_phone(match):
    phone = match.group()
    first_digit_found = False
    result = []
    for ch in phone:
        if ch.isdigit():
            if not first_digit_found:
                result.append(ch)
                first_digit_found = True
            else:
                result.append('*')
        else:
            result.append(ch)
    return ''.join(result)

masked_text = phone_pattern.sub(mask_phone, text)

print(f"\nЗмінений текст (цифри після першої замінені на *):")
print(masked_text)

print("\n" + "=" * 60)
print("ЗАВДАННЯ 3: Обробка тексту за допомогою NLTK")
print("=" * 60)

sentences = sent_tokenize(text)
print(f"\nа) Кількість речень у тексті: {len(sentences)}")
for i, s in enumerate(sentences, 1):
    print(f"   Речення {i}: {s[:80]}{'...' if len(s) > 80 else ''}")

print(f"\nб) Токенізація слів (без номерів телефонів):")

text_no_phones = phone_pattern.sub('', text)
words = word_tokenize(text_no_phones)
words_only = [w for w in words if w.isalpha()]

print(f"   Кількість слів: {len(words_only)}")
print(f"   Перші 20 слів: {words_only[:20]}")

print(f"\nв) 10 слів, які зустрічаються найчастіше:")

word_freq = Counter([w.lower() for w in words_only])
for word, count in word_freq.most_common(10):
    print(f"   '{word}' — {count} раз(ів)")

print(f"\nг) Лематизація передостаннього речення:")

if len(sentences) >= 2:
    penultimate = sentences[-2]
else:
    penultimate = sentences[0]

print(f"   Передостаннє речення: {penultimate}")

pen_tokens = word_tokenize(penultimate)
pen_words = [w for w in pen_tokens if w.isalpha()]
pos_tagged = nltk.pos_tag(pen_words)
print(f"\n   Частини мови (POS-тегування):")
for word, tag in pos_tagged:
    print(f"     {word:20s} -> {tag}")

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

lemmatizer = WordNetLemmatizer()
print(f"\n   Лематизація:")
for word, tag in pos_tagged:
    wn_pos = get_wordnet_pos(tag)
    lemma = lemmatizer.lemmatize(word, pos=wn_pos)
    print(f"     {word:20s} ({tag:5s}) -> {lemma}")
