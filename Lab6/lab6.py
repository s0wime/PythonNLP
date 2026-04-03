import json
import random
import warnings
import spacy
from spacy.training import Example

warnings.filterwarnings('ignore')
random.seed(42)

# =====================================================================
# ЗАВДАННЯ 1: Навчання NER — розпізнавання музичних сутностей
# =====================================================================
print("=" * 60)
print("1) Навчання NER — музичні сутності (ARTIST, SONG, GENRE)")
print("=" * 60)

ner_trainset = [
    ("I love listening to Bohemian Rhapsody by Queen", {"entities": [(20, 38, "SONG"), (42, 47, "ARTIST")]}),
    ("Play some jazz music for me", {"entities": [(10, 14, "GENRE")]}),
    ("Can you find songs by The Beatles", {"entities": [(21, 33, "ARTIST")]}),
    ("I want to hear Hotel California by Eagles", {"entities": [(17, 33, "SONG"), (37, 43, "ARTIST")]}),
    ("Put on some rock and roll", {"entities": [(12, 25, "GENRE")]}),
    ("Find me tracks by Eminem", {"entities": [(18, 24, "ARTIST")]}),
    ("Play Stairway to Heaven by Led Zeppelin", {"entities": [(5, 24, "SONG"), (28, 40, "ARTIST")]}),
    ("I enjoy hip hop music", {"entities": [(8, 15, "GENRE")]}),
    ("Show me songs from Adele", {"entities": [(19, 24, "ARTIST")]}),
    ("Play Yesterday by The Beatles please", {"entities": [(5, 14, "SONG"), (18, 30, "ARTIST")]}),
    ("I love classical music", {"entities": [(7, 16, "GENRE")]}),
    ("Can you play Thriller by Michael Jackson", {"entities": [(13, 21, "SONG"), (25, 40, "ARTIST")]}),
    ("Find pop songs for the party", {"entities": [(5, 8, "GENRE")]}),
    ("Play Imagine by John Lennon", {"entities": [(5, 12, "SONG"), (16, 27, "ARTIST")]}),
    ("I want some country music", {"entities": [(12, 19, "GENRE")]}),
    ("Put on Shape of You by Ed Sheeran", {"entities": [(7, 19, "SONG"), (23, 33, "ARTIST")]}),
    ("Play electronic dance music", {"entities": [(5, 27, "GENRE")]}),
    ("I want to listen to Drake", {"entities": [(20, 25, "ARTIST")]}),
    ("Find Blinding Lights by The Weeknd", {"entities": [(5, 20, "SONG"), (24, 34, "ARTIST")]}),
    ("Play some blues tonight", {"entities": [(10, 15, "GENRE")]}),
]

nlp_ner = spacy.load("en_core_web_sm")
ner = nlp_ner.get_pipe("ner")
ner.add_label("ARTIST")
ner.add_label("SONG")
ner.add_label("GENRE")

other_pipes = [pipe for pipe in nlp_ner.pipe_names if pipe != 'ner']
epochs = 30

print(f"Навчання NER ({epochs} епох, {len(ner_trainset)} прикладів)...")
with nlp_ner.disable_pipes(*other_pipes):
    optimizer = nlp_ner.create_optimizer()
    for i in range(epochs):
        random.shuffle(ner_trainset)
        losses = {}
        for text, annotation in ner_trainset:
            doc = nlp_ner.make_doc(text)
            example = Example.from_dict(doc, annotation)
            nlp_ner.update([example], sgd=optimizer, losses=losses)
        if (i + 1) % 10 == 0:
            print(f"  Епоха {i+1}, loss: {losses.get('ner', 0):.4f}")

print("\nТестування NER на нових реченнях:")
test_sentences = [
    "Play Lose Yourself by Eminem on my speaker",
    "I want to hear some jazz from Miles Davis",
    "Can you find rock songs by Metallica",
    "Play Watermelon Sugar by Harry Styles",
    "I love listening to classical and pop music",
]
for sent in test_sentences:
    doc = nlp_ner(sent)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"  \"{sent}\"")
    if entities:
        for text, label in entities:
            print(f"    -> {text} [{label}]")
    else:
        print(f"    -> (сутностей не знайдено)")

# =====================================================================
# ЗАВДАННЯ 2: TextCategorizer — визначення намірів (intents)
# =====================================================================
print("\n" + "=" * 60)
print("2) TextCategorizer — визначення намірів з music.json")
print("=" * 60)

with open("music.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

utterance_intent_pairs = []
for d in dialogues:
    for turn in d['turns']:
        if turn['speaker'] == 'USER':
            for frame in turn['frames']:
                for act in frame.get('actions', []):
                    if act.get('slot') == 'intent':
                        intent = act['values'][0]
                        utterance_intent_pairs.append((turn['utterance'], intent))

all_intents = list(set(i for _, i in utterance_intent_pairs))
all_intents.sort()
print(f"Знайдено {len(utterance_intent_pairs)} висловлювань з намірами")
print(f"Наміри: {all_intents}")

train_data = []
for text, intent in utterance_intent_pairs:
    cats = {i: 1.0 if i == intent else 0.0 for i in all_intents}
    train_data.append((text, {"cats": cats}))

random.shuffle(train_data)
split = int(len(train_data) * 0.8)
train_set = train_data[:split]
test_set = train_data[split:]
print(f"Train: {len(train_set)}, Test: {len(test_set)}")

nlp_cat = spacy.blank("en")
textcat = nlp_cat.add_pipe("textcat")
for intent in all_intents:
    textcat.add_label(intent)

train_examples = [Example.from_dict(nlp_cat.make_doc(text), label)
                  for text, label in train_set]
textcat.initialize(lambda: train_examples, nlp=nlp_cat)

epochs = 30
print(f"\nНавчання TextCategorizer ({epochs} епох)...")
with nlp_cat.select_pipes(enable="textcat"):
    optimizer = nlp_cat.resume_training()
    for i in range(epochs):
        random.shuffle(train_set)
        losses = {}
        for text, label in train_set:
            doc = nlp_cat.make_doc(text)
            example = Example.from_dict(doc, label)
            nlp_cat.update([example], sgd=optimizer, losses=losses)
        if (i + 1) % 10 == 0:
            print(f"  Епоха {i+1}, loss: {losses.get('textcat', 0):.4f}")

print(f"\nОцінка на тестовій вибірці ({len(test_set)} прикладів):")
correct = 0
for text, label in test_set:
    doc = nlp_cat(text)
    predicted = max(doc.cats, key=doc.cats.get)
    actual = max(label['cats'], key=label['cats'].get)
    if predicted == actual:
        correct += 1
accuracy = correct / len(test_set)
print(f"Accuracy: {accuracy:.4f} ({correct}/{len(test_set)})")

print(f"\nПриклади передбачень на тестових даних:")
for text, label in test_set[:8]:
    doc = nlp_cat(text)
    predicted = max(doc.cats, key=doc.cats.get)
    actual = max(label['cats'], key=label['cats'].get)
    scores = {k: f"{v:.2f}" for k, v in doc.cats.items()}
    mark = "+" if predicted == actual else "X"
    print(f"  [{mark}] \"{text[:70]}{'...' if len(text) > 70 else ''}\"")
    print(f"      Реальний: {actual}, Передбачений: {predicted} {scores}")

print(f"\nТестування на нових висловлюваннях:")
new_utterances = [
    "Find me some good songs to listen to",
    "Play that track right now",
    "I want to discover new rock music",
    "Can you start playing the last song",
]
for text in new_utterances:
    doc = nlp_cat(text)
    predicted = max(doc.cats, key=doc.cats.get)
    score = doc.cats[predicted]
    print(f"  \"{text}\" -> {predicted} ({score:.3f})")
