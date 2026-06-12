import json
import spacy
from spacy import util
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

with open("music.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

user_utterances = []
for d in dialogues:
    for turn in d['turns']:
        if turn['speaker'] == 'USER':
            user_utterances.append(turn['utterance'])

print(f"Витягнуто {len(user_utterances)} висловлювань користувача\n")

# =====================================================================
# 1а) Matcher — виділення назв альбомів за лінгвістичними шаблонами
# =====================================================================
print("=" * 60)
print("1а) Виділення назв альбомів за допомогою Matcher (шаблони)")
print("=" * 60)

matcher_albums = Matcher(nlp.vocab)

_NO_NAME = ["VERB", "PUNCT", "ADP", "CCONJ", "SCONJ", "PART"]

# Шаблон 1: прийменник + опц. "the" + "album" + назва (1–4 слова)
# "from the album Born This Way", "in album camila", "on the album Vessel"
matcher_albums.add("ALBUM_PREP", [[
    {"LOWER": {"IN": ["on", "in", "from", "off", "for"]}},
    {"LOWER": "the", "OP": "?"},
    {"LOWER": "album"},
    {"POS": {"NOT_IN": _NO_NAME}, "IS_ALPHA": True},
    {"POS": {"NOT_IN": _NO_NAME}, "IS_ALPHA": True, "OP": "?"},
    {"POS": {"NOT_IN": _NO_NAME}, "IS_ALPHA": True, "OP": "?"},
    {"POS": {"NOT_IN": _NO_NAME}, "IS_ALPHA": True, "OP": "?"},
]])

# Шаблон 2: артикль/присвійний займенник + "album" + назва (1–4 слова)
# "the album Happiness", "his album Cryptic", "the album My Everything"
matcher_albums.add("ALBUM_DET", [[
    {"LOWER": {"IN": ["the", "a", "an", "his", "her", "my", "our", "their"]}},
    {"LOWER": "album"},
    {"POS": {"NOT_IN": _NO_NAME}, "IS_ALPHA": True},
    {"POS": {"NOT_IN": _NO_NAME}, "IS_ALPHA": True, "OP": "?"},
    {"POS": {"NOT_IN": _NO_NAME}, "IS_ALPHA": True, "OP": "?"},
    {"POS": {"NOT_IN": _NO_NAME}, "IS_ALPHA": True, "OP": "?"},
]])

# Шаблон 3: "the" + власні іменники з великої літери + "album" (назва перед ключовим словом)
# "the Built On Glass album", "the Meliora album", "the Visions album"
matcher_albums.add("ALBUM_NAME_FIRST", [[
    {"LOWER": "the"},
    {"IS_TITLE": True},
    {"IS_TITLE": True, "OP": "?"},
    {"IS_TITLE": True, "OP": "?"},
    {"LOWER": "album"},
]])

# Шаблон 4: "album" без артикля перед власною назвою
# "album To Whom It May Concern", "album Night Visions", "album camila"
matcher_albums.add("ALBUM_DIRECT", [[
    {"LOWER": "album"},
    {"POS": {"IN": ["PROPN", "NOUN", "ADJ"]}, "IS_ALPHA": True},
    {"POS": {"IN": ["PROPN", "NOUN", "ADJ", "DET"]}, "IS_ALPHA": True, "OP": "?"},
    {"POS": {"IN": ["PROPN", "NOUN", "ADJ", "DET"]}, "IS_ALPHA": True, "OP": "?"},
    {"POS": {"IN": ["PROPN", "NOUN", "ADJ", "DET"]}, "IS_ALPHA": True, "OP": "?"},
]])


def extract_album_name(doc, span, match_label):
    for i, token in enumerate(span):
        if token.lower_ == "album":
            if match_label == "ALBUM_NAME_FIRST":
                return doc[span.start + 1 : span.end - 1].text
            else:
                name_start = span.start + i + 1
                if name_start < span.end:
                    return doc[name_start : span.end].text
    return span.text


found_albums = []
for utt in user_utterances:
    doc = nlp(utt)
    matches = matcher_albums(doc)
    if not matches:
        continue
    # filter_spans keeps the longest span when matches overlap
    label_map = {(s, e): nlp.vocab.strings[mid] for mid, s, e in matches}
    spans = util.filter_spans([doc[s:e] for _, s, e in matches])
    for span in spans:
        label = label_map[(span.start, span.end)]
        album_name = extract_album_name(doc, span, label)
        found_albums.append((utt, album_name, label))

print(f"\nЗнайдено {len(found_albums)} згадок альбомів у висловлюваннях:\n")
for utt, album, label in found_albums[:25]:
    print(f"  [{label}] Альбом: \"{album}\"")
    print(f"    -> \"{utt[:90]}{'...' if len(utt) > 90 else ''}\"")
    print()
if len(found_albums) > 25:
    print(f"  ... та ще {len(found_albums) - 25}")

# =====================================================================
# 1б) Matcher — висловлювання-підтвердження
# =====================================================================
print("\n" + "=" * 60)
print("1б) Виділення підтверджень за допомогою Matcher")
print("=" * 60)

matcher = Matcher(nlp.vocab)

# "Yes" на початку (Yes, Yes!, Yes that's...)
pattern_yes = [
    {"LOWER": {"IN": ["yes", "yeah", "yep", "yea"]}},
    {"IS_PUNCT": True, "OP": "?"}
]
matcher.add("CONFIRM_YES", [pattern_yes])

# "Ok" / "Okay" на початку
pattern_ok = [
    {"LOWER": {"IN": ["ok", "okay", "alright"]}},
    {"IS_PUNCT": True, "OP": "?"}
]
matcher.add("CONFIRM_OK", [pattern_ok])

# "Sure" / "Sure thing"
pattern_sure = [
    {"LOWER": "sure"},
    {"LOWER": "thing", "OP": "?"},
    {"IS_PUNCT": True, "OP": "?"}
]
matcher.add("CONFIRM_SURE", [pattern_sure])

# "That's right/it/great/good/wonderful"
pattern_thats = [
    {"LOWER": {"IN": ["that", "that's", "thats"]}},
    {"LOWER": {"IN": ["is", "will", "would"]}, "OP": "?"},
    {"LOWER": {"IN": ["right", "it", "great", "good", "wonderful", "fine", "correct", "perfect"]}}
]
matcher.add("CONFIRM_THAT", [pattern_thats])

# "That sounds good/great"
pattern_sounds = [
    {"LOWER": "that"},
    {"LOWER": {"IN": ["sounds", "looks"]}},
    {"LOWER": {"IN": ["good", "great", "nice", "fine"]}}
]
matcher.add("CONFIRM_SOUNDS", [pattern_sounds])

# "Great" / "Thanks"
pattern_great = [
    {"LOWER": {"IN": ["great", "thanks", "perfect", "wonderful", "excellent"]}},
    {"IS_PUNCT": True, "OP": "?"}
]
matcher.add("CONFIRM_GREAT", [pattern_great])

confirmations = []
for utt in user_utterances:
    doc = nlp(utt)
    matches = matcher(doc)
    if matches:
        labels = set()
        for match_id, start, end in matches:
            label = nlp.vocab.strings[match_id]
            labels.add(label)
        confirmations.append((utt, labels))

print(f"\nЗнайдено {len(confirmations)} висловлювань-підтверджень:\n")
for utt, labels in confirmations[:20]:
    print(f"  [{', '.join(labels)}] \"{utt}\"")
if len(confirmations) > 20:
    print(f"  ... та ще {len(confirmations) - 20}")

# =====================================================================
# 2) Синтаксичні залежності для визначення намірів
# =====================================================================
print("\n" + "=" * 60)
print("2) Синтаксичні залежності — визначення намірів")
print("=" * 60)

def extract_intents_deps(text):
    doc = nlp(text)
    intents = []
    for token in doc:
        if token.dep_ == "dobj" and token.head.pos_ == "VERB":
            verb = token.head.lemma_
            obj = token.text
            conj = [t.text for t in token.conjuncts]
            intents.append({
                "verb": verb,
                "object": obj,
                "conjuncts": conj,
                "intent": verb + obj.capitalize()
            })
        elif token.dep_ == "ROOT" and token.pos_ == "VERB":
            has_dobj = any(c.dep_ == "dobj" for c in token.children)
            if not has_dobj:
                prep_objs = []
                for child in token.children:
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                prep_objs.append(grandchild.text)
                if prep_objs:
                    intents.append({
                        "verb": token.lemma_,
                        "object": prep_objs[0],
                        "conjuncts": [],
                        "intent": token.lemma_ + prep_objs[0].capitalize()
                    })
    return intents

print("\nа) Приклади визначення намірів з діалогів:\n")

sample_utterances = [
    "Find me some rock songs",
    "Play the song on my kitchen speaker",
    "I want to listen to some music",
    "Can you find songs by Adele",
    "Play the track and show the lyrics",
    "Search for pop music from the album Camila",
    "I want to hear jazz and blues songs",
]

for utt in sample_utterances:
    intents = extract_intents_deps(utt)
    print(f"  \"{utt}\"")
    if intents:
        for intent in intents:
            conj_str = f" (+ {intent['conjuncts']})" if intent['conjuncts'] else ""
            print(f"    -> {intent['intent']}{conj_str}")
    else:
        print(f"    -> (намір не визначено)")
    print()

print("б) Визначення намірів у висловлюваннях з music.json:\n")

intent_counter = {}
for utt in user_utterances[:200]:
    intents = extract_intents_deps(utt)
    for intent in intents:
        key = intent['intent']
        intent_counter[key] = intent_counter.get(key, 0) + 1

sorted_intents = sorted(intent_counter.items(), key=lambda x: x[1], reverse=True)
print(f"Топ-15 найчастіших намірів:")
for intent, count in sorted_intents[:15]:
    print(f"  {intent:<30} — {count} разів")

print("\nв) Дерево залежностей — приклад:\n")
example = "Play the song on my kitchen speaker"
doc = nlp(example)
print(f"  \"{example}\"")
print(f"  {'Token':<15} {'POS':<8} {'Dep':<12} {'Head':<15}")
print(f"  {'-'*50}")
for token in doc:
    print(f"  {token.text:<15} {token.pos_:<8} {token.dep_:<12} {token.head.text:<15}")
