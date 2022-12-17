import numpy
import streamlit as st
import torch

st.title('Генерация текста GPT-моделью')
st.subheader('Это приложение показывает разницу в генерации текста моделью rugpt3small, обученной на документах общей тематики и этой же моделью, дообученной на анекдотах')
# Загружаем токенайзер модели
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

from transformers import GPT2LMHeadModel

# Эту модель просто подгружаем
model_init = GPT2LMHeadModel.from_pretrained(
    'sberbank-ai/rugpt3small_based_on_gpt2',
    output_attentions = False,
    output_hidden_states = False,
)

# Это обученная модель, в нее загружаем веса
model = GPT2LMHeadModel.from_pretrained(
    'sberbank-ai/rugpt3small_based_on_gpt2',
    output_attentions = False,
    output_hidden_states = False,
)

m = torch.load('model.pt')
model.load_state_dict(m)


str = st.text_input('Введите 1-4 слова начала текста, и подождите минутку', 'Мужик спрашивает у официанта')

# модель без дообучения
# prompt – строка, которую примет на вход и продолжит модель

# токенизируем строку
prompt = tokenizer.encode(str, return_tensors='pt')

# out будет содержать результаты генерации в виде списка
out1 = model_init.generate(
    # входная строка
    input_ids=prompt,
    # максимальная длина генерируемой последовательности
    max_length=150,
    # num_beams
    num_beams=5,
    # применяем сэмплирование
    do_sample=True,
    # применяем температуру
    temperature=1.,
    # топ слов по вероятности
    top_k=50,
    # топ слов по суммарной вероятности
    top_p=0.6,
    # сколько (постараться) не повторять n_gram подряд
    no_repeat_ngram_size=3,
    # сколько вернуть генераций
    num_return_sequences=3,
    ).numpy() #).cpu().numpy()

st.write('\n------------------\n')
st.subheader('Тексты на модели, обученной документами всех тематик:')
# out содержит результаты
# декодируем и печатаем
n = 0
for out_ in out1:
    n += 1
    st.write(tokenizer.decode(out_).rpartition('.')[0],'.')
    st.write('\n------------------\n')
    # print(tokenizer.decode(out_))


# дообученная модель
with torch.inference_mode():
    # prompt = 'Мужик спрашивает официанта'
    # prompt = tokenizer.encode(str, return_tensors='pt')
    out2 = model.generate(
        input_ids=prompt,
        max_length=150,
        num_beams=1,
        do_sample=True,
        temperature=1., 
        top_k=5,
        top_p=0.6,
        no_repeat_ngram_size=2,
        num_return_sequences=3,
        ).numpy() #).cpu().numpy()
    
    st.subheader('Тексты на модели, обученной документами всех тематик и дообученной анекдотами:')
    n = 0
    for out_ in out2:
        n += 1
        st.write(tokenizer.decode(out_).rpartition('.')[0],'.')
        # print(textwrap.fill(tokenizer.decode(out_), 100), end='\n------------------\n')
        st.write('\n------------------\n')
