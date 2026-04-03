import json
import spacy
from spacy.matcher import Matcher, PhraseMatcher

nlp = spacy.load("en_core_web_sm")

with open("music.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

albums = set()
user_utterances = []
for d in dialogues:
    for turn in d['turns']:
        if turn['speaker'] == 'USER':
            user_utterances.append(turn['utterance'])
        for frame in turn['frames']:
            for act in frame.get('actions', []):
                if act.get('slot') == 'album':
                    for v in act['values']:
                        albums.add(v)

albums = sorted(albums)
print(f"Витягнуто {len(user_utterances)} висловлювань користувача")
print(f"Знайдено {len(albums)} назв альбомів у даних\n")

# =====================================================================
# 1а) Matcher / PhraseMatcher — виділення назв альбомів
# =====================================================================
print("=" * 60)
print("1а) Виділення назв альбомів за допомогою PhraseMatcher")
print("=" * 60)

phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
album_patterns = [nlp.make_doc(album) for album in albums]
phrase_matcher.add("ALBUM", album_patterns)

found_count = 0
for utt in user_utterances:
    doc = nlp(utt)
    matches = phrase_matcher(doc)
    if matches:
        for match_id, start, end in matches:
            span = doc[start:end]
            print(f"  \"{utt[:80]}{'...' if len(utt)>80 else ''}\"")
            print(f"    -> Альбом: \"{span.text}\" (позиція {start}-{end})")
            found_count += 1

print(f"\nЗнайдено {found_count} згадок альбомів у висловлюваннях користувача")

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
