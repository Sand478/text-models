import streamlit as st

# !pip install -q transformers

import numpy as np
# import pandas as pd
import re
# import random

import torch
# from tqdm.notebook import tqdm
import transformers
# from torch.optim import AdamW

import textwrap

# Загружаем токенайзер модели
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

# import re
with open('anekdoty.txt', encoding='utf8') as f:
    text = f.read()

text = re.sub('\n{2,}', '\n', text)
print(text[:1000])


# токенизируем текст
tokens = tokenizer.encode(text, add_special_tokens=True)
tokens = np.array(tokens)
print(len(tokens))
tokens[:10]


# разбиваем на train и test

l = len(tokens)//15
train = []
test = []
for i in range(15):
    if i%5 > 0:
        train.extend(tokens[i*l: (i+1)*l])
    else:
        test.extend(tokens[i*l: (i+1)*l])
train = np.array(train)
test = np.array(test)

print(len(tokens), len(train), len(test))



from transformers import GPT2LMHeadModel

# Эту модель просто подгружаем и не будем дообучать 
model_init = GPT2LMHeadModel.from_pretrained(
    'sberbank-ai/rugpt3small_based_on_gpt2',
    output_attentions = False,
    output_hidden_states = False,
)


# Эту модель подгрузим и далее обучим
model = GPT2LMHeadModel.from_pretrained(
    'sberbank-ai/rugpt3small_based_on_gpt2',
    output_attentions = False,
    output_hidden_states = False,
)

model.to(device);
model_init.to(device);


batch_size = 8
max_len = 256
epochs = 5

n_train = len(train)//(batch_size*max_len)
n_test = len(test)//(batch_size*max_len)
print(n_train, n_test)

# устанавливаем оптимизатор
optimizer = AdamW(model.parameters(), lr = 1e-5, eps = 1e-8)

# трансформеры с трудом обучаются, для них нужны разные способы повышения 
# эффективности градиентного спуска
total_steps = n_train * epochs
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)


# зададим точность, хотя ориентироваться будем на качество генерации
def accuracy(y_true, logits):
    return torch.mean((y_true[1:] == torch.argmax(logits, dim=2)[:-1]).float()).detach().cpu().numpy()



# готовим тензоры для обучения размера [batch_size, max_len]

def prep_tensors(x, i, batch_size=batch_size, max_len=max_len):
    batch_ids = x[i*batch_size*max_len: (i+1)*batch_size*max_len]
    batch_ids = batch_ids.reshape(batch_size, max_len)
    batch_ids = torch.tensor(batch_ids).to(device)
    return batch_ids


# обучающий цикл
for epoch in range(1, epochs+1):
    print(f'epoch {epoch}/{epochs} : training')

    train_loss = []
    train_acc = []
    model.train()
    pbar = range(n_train)
    # pbar = tqdm(range(n_train))
    for i in pbar:
        batch_ids = prep_tensors(train, i)

        model.zero_grad()
        loss, logits, _ = model(batch_ids,
                                token_type_ids=None, 
                                labels=batch_ids
                             ).values()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        train_loss.append(loss.item())
        train_acc.append(accuracy(batch_ids, logits))
        print(f'acc {np.mean(train_acc):.4f} loss {np.mean(train_loss):.4f}')
        # pbar.set_description(f'acc {np.mean(train_acc):.4f} loss {np.mean(train_loss):.4f}', refresh=True)

    
    print(f'epoch {epoch}/{epochs} : validation')
    model.eval()
    val_acc = []
    val_loss = []
    pbar = range(n_test)
    # pbar = tqdm(range(n_test))
    for i in pbar:
        batch_ids = prep_tensors(test, i)
        with torch.no_grad():        
            loss, logits, _ = model(batch_ids, 
                                token_type_ids=None,
                                labels=batch_ids
                                 ).values()
        
        val_loss.append(loss.item())
        val_acc.append(accuracy(batch_ids, logits))
        print(f'acc {np.mean(val_acc):.4f} loss {np.mean(val_loss):.4f}')
        # pbar.set_description(f'acc {np.mean(val_acc):.4f} loss {np.mean(val_loss):.4f}', refresh=True)


# Применим модель, которую мы не дообучали: просто для понимания разницы между дообученной на собственных данных моделью и предобученной.
# https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate
# модель без дообучения

# prompt – строка, которую модель примет на вход и продолжит
prompt = 'Мужик спрашивает официанта'

# токенизируем строку
prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)

# out будет содержать результаты генерации в виде списка
out = model_init.generate(
    # входная строка
    input_ids=prompt,
    # максимальная длина генерируемой последовательности
    max_length=250,
    # num_beams
    num_beams=5,
    # применяем сэмплирование
    do_sample=True,
    # применяем температуру
    temperature=55.,
    # топ слов по вероятности
    top_k=50,
    # топ слов по суммарной вероятности
    top_p=0.6,
    # сколько (постараться) не повторять n_gram подряд
    no_repeat_ngram_size=3,
    # сколько вернуть генераций
    num_return_sequences=7,
    ).cpu().numpy()

# out содержит результаты


# декодируем и печатаем
for out_ in out:
    print(tokenizer.decode(out_))


# дообученная модель
with torch.inference_mode():
    prompt = 'Мужик спрашивает официанта'
    prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)
    out = model.generate(
        input_ids=prompt,
        max_length=150,
        num_beams=1,
        do_sample=True,
        temperature=1.,
        top_k=5,
        top_p=0.6,
        no_repeat_ngram_size=2,
        num_return_sequences=7,
        ).cpu().numpy()
    for out_ in out:
        print(textwrap.fill(tokenizer.decode(out_), 100), end='\n------------------\n')



# Сохраняем веса обученной модели
torch.save(model.state_dict(), 'model.pt')

# Задаем класс модели (уже в streamlit/tg_bot)
model_finetuned = GPT2LMHeadModel.from_pretrained(
    'sberbank-ai/rugpt3small_based_on_gpt2',
    output_attentions = False,
    output_hidden_states = False,
)

# Вешаем сохраненные веса на нашу модель
model = model_finetuned.load_state_dict(torch.load('model.pt'))

# -> <All keys matched successfully>