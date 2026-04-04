import json
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

with open("../Lab6/music.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

data = []
for d in dialogues:
    turns = d["turns"]
    for i in range(len(turns) - 1):
        if turns[i]["speaker"] == "USER" and turns[i + 1]["speaker"] == "SYSTEM":
            user_utt = turns[i]["utterance"]
            sys_utt = turns[i + 1]["utterance"]
            data.append((user_utt, sys_utt))

print(f"Витягнуто {len(data)} пар (USER -> SYSTEM) з діалогів")
print(f"\nПриклади пар:")
for inp, tgt in data[:5]:
    print(f"  USER:   {inp}")
    print(f"  SYSTEM: {tgt}")
    print()

model_name = "facebook/bart-base"
print(f"Завантаження моделі {model_name}...")
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

for param in model.get_encoder().parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Параметрів: {total:,} всього, {trainable:,} навчаються (декодер)")

train_data = data[:80]
test_data = data[80:100]

max_len = 64

input_ids_list = []
attention_masks_list = []
target_ids_list = []

for input_text, target_text in train_data:
    encoded_input = tokenizer(
        input_text, max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    encoded_target = tokenizer(
        target_text, max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids_list.append(encoded_input["input_ids"].squeeze())
    attention_masks_list.append(encoded_input["attention_mask"].squeeze())
    target_ids_list.append(encoded_target["input_ids"].squeeze())

input_ids = torch.stack(input_ids_list)
attention_masks = torch.stack(attention_masks_list)
target_ids = torch.stack(target_ids_list)

print(f"\nНавчальні дані: {input_ids.shape[0]} прикладів")
print(f"Тестові дані: {len(test_data)} прикладів")

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5
)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

num_epochs = 10
batch_size = 8
model.train()

print(f"\nНавчання ({num_epochs} епох, batch_size={batch_size}):")
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    for start in range(0, len(train_data), batch_size):
        end = min(start + batch_size, len(train_data))
        batch_input = input_ids[start:end]
        batch_mask = attention_masks[start:end]
        batch_target = target_ids[start:end]

        outputs = model(
            input_ids=batch_input,
            attention_mask=batch_mask,
            labels=batch_target
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    print(f"  Епоха {epoch + 1:2d}/{num_epochs}: loss = {avg_loss:.4f}")

def generate_response(input_text, max_length=60):
    model.eval()
    encoded = tokenizer(input_text, return_tensors="pt", max_length=max_len, truncation=True)
    with torch.no_grad():
        output = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("\n" + "=" * 60)
print("Генерація відповідей на тестових даних:")
print("=" * 60)
for inp, real in test_data[:10]:
    generated = generate_response(inp)
    print(f"\n  USER:        {inp}")
    print(f"  Реальна:     {real}")
    print(f"  Згенерована: {generated}")

print("\n" + "=" * 60)
print("Генерація відповідей на нових запитах:")
print("=" * 60)

new_queries = [
    "I want to listen to some rock music",
    "Can you play something by Adele?",
    "Find me jazz songs please",
    "Play the song on my kitchen speaker",
    "I'd like to hear some pop music from the 90s",
]

for query in new_queries:
    response = generate_response(query)
    print(f"\n  USER:     {query}")
    print(f"  SYSTEM:   {response}")
