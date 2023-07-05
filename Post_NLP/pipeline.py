import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

from libs.TextPreprocessing import TextPreprocessing

import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras_preprocessing import text, sequence

from navec import Navec


from IPython.display import display
from tqdm import tqdm
tqdm.pandas()

from psycopg import Connection


#                           ИСХОДНЫЕ ДАННЫЕ


#
uri = 'postgresql://twin:58er2504Vb@85.172.79.228:5432/dwh'

with Connection.connect(uri, autocommit=True) as connection:
    with connection.cursor() as cursor:
        # проверяем наличие схемы "public" в бд
        cursor.execute('''
            SELECT
         Entrypoint.client_id
        ,Profile.params
        ,Feature.name

        FROM "Entrypoint" as Entrypoint
        inner join "Profile" as Profile ON
            Profile.entrypoint_id = Entrypoint.entrypoint_id
        inner join "Feature" as Feature ON
            Feature.id = Profile.feature_id

        where
            Entrypoint.datasource_id = 2 AND Feature.name = 'id'
        ''')
        df_profile = pd.DataFrame(cursor.fetchall(), columns=list(map(lambda x: x.name, cursor.description)))
        cursor.execute('''
            SELECT
         Entrypoint.client_id
        ,Post.owner_id
        ,Post.text::json->'text' as text

        FROM "Entrypoint" as Entrypoint
        inner join "Post" as Post ON
            Post.entrypoint_id = Entrypoint.entrypoint_id
        WHERE
            Entrypoint.datasource_id = 2 and Post.type = 'items'
        ''')
        df_post = pd.DataFrame(cursor.fetchall(), columns=list(map(lambda x: x.name, cursor.description)))
display(df_post.head(3))
display(df_profile.head(3))

#                           ПСИХОТИПЫ
psychotypes    = list('ABCDEFG')
# количество возможных классов (психотипов)
NUM_CLASSES    = len(psychotypes)

df_psychotypes = df_profile[['client_id']].drop_duplicates()
df_psychotypes['Subject'] = np.random.choice(psychotypes, df_psychotypes.shape[0])

# оставляем идентфиикатор
df_profile = df_profile[['client_id', 'params']].rename(columns={'params': 'user_id'})

display(df_psychotypes)
display(df_post.shape)
display(df_post.merge(df_profile, on='client_id'))

df_post = df_post.merge(df_profile, on='client_id').query("owner_id == user_id")#.iloc[:50000]
display(df_post.shape)
df_post = df_post.merge(df_psychotypes, on='client_id')#[['text', 'Subject']]
display(df_post.shape)

# #                           ПРЕДОБРАБОТКА


# приведение к одному формату
df_post['new_text']  = df_post['text'].progress_apply(TextPreprocessing.clean_format, duration_log=False)
# удаление тегов html
df_post['new_text']  = df_post['new_text'].progress_apply(TextPreprocessing.clean_html, duration_log=False)
# удаление спец символов html
df_post['new_text']  = df_post['new_text'].progress_apply(TextPreprocessing.clean_html_special_characters_v2, duration_log=False)

# удаление ссылок и адресов электронной почты
df_post['new_text']  = df_post['new_text'].progress_apply(TextPreprocessing.clean_url_and_email, duration_log=False)

# очистка сокращений
df_post['new_text']  = df_post['new_text'].progress_apply(TextPreprocessing.clean_short_constructions, duration_log=False)

# очистка текста от всего, кроме букв, пробелов и тире(если оно стоит между буквами)
df_post['new_text']  = df_post['new_text'].progress_apply(TextPreprocessing.clean_text_total, duration_log=False)

# леммматизация
df_post['new_text']  = df_post['new_text'].progress_apply(TextPreprocessing.lemmatization_v2, duration_log=False)

# стоп слова
df_post['new_text']  = df_post['new_text'].progress_apply(TextPreprocessing.remove_Stopwords, duration_log=False)

# удаление лишних пробелов
df_post['new_text']  = df_post['new_text'].progress_apply(TextPreprocessing.clean_extra_spaces, duration_log=False)

df_post['len'] = df_post['new_text'].str.strip().str.len()
# отбрасываем пустые посты
df_post = df_post[df_post['len'] > 0].sort_values('len', ascending=False)
display(df_post.shape)

df_post.to_csv('df_post_preprocessed.csv', index=False)
# df_post = pd.read_csv('df_post_preprocessed.csv')
# df_post

#                           СБОР ПОСТОВ НА КАЖДОГО ПОЛЬЗОВАТЕЛЯ
# собираем все посты по пользвоател без дубликатов (одинаковый текст постов исклюячаестя)
df_new = df_post.groupby(['client_id', 'new_text'], as_index=False)['Subject'].first()
df_new = df_new.groupby('client_id').agg({'new_text': ' '.join, 'Subject': 'first'})
display(df_new)

df_new = pd.concat([df_new, pd.get_dummies(df_new['Subject'])], axis=1)
df_new = df_new.drop(columns='Subject')
posts  = df_new['new_text']
label  = df_new[psychotypes]
post_train, post_test, label_train, label_test = train_test_split(posts, label, test_size=0.2, random_state=123)


def delete_part_of_speech(word):
    for p in ['_NOUN', '_ADJ', '_VERB', '_PROPN', '_ADV', '_NUM', '_X', '_INTJ', '_SYM']:
        word = word.replace(p, '')
    return word
#                                        ЗАГРУЖАЕМ ВЕКТОРА Glove
# путь к предобученным векторам и их размерность
embeddings_book = {
    'glove_300': {'path': 'experiments/data/glove.6B.300d.txt', 'dim': 300},
    'glove_100_twitter_multy-lang': {'path': 'experiments/data/glove.twitter.27B.100d.txt', 'dim': 100},
    'navec': {'path': 'experiments/data/navec_hudlit_v1_12B_500K_300d_100q.tar', 'dim': 300},
    'russian_news': {'path': 'experiments/data/russian_news.txt', 'dim': 300},
    'fasttext_rus_300': {'path': 'experiments/data/cc.ru.300.vec', 'dim': 300}  # может не хватить ОЗУ
}
selected  = 'fasttext_rus_300'

#              чтение всех векторов сразу (теперь не нужно, так как способ с пачками использует меньше ОЗУ)
# # читаем из файла (структура у них одинакова на каждой строке массив значений, разделенных проблеами,
# # первое значение - слово, остальное - его вектор)
# if selected in ['glove_300', 'glove_100_twitter_multy-lang', 'russian_news', 'fasttext_rus_300']:
#     trained_embeddings = {}
#     with open(embeddings_book[selected]['path'], encoding='utf-8') as file:
#         lines = file.readlines()
#         # для fasttext не нужна первая строка файла, содержит метаданные
#         # проверяем по числу значений в строке (если < размерности вектора => не вектор)
#         if len(lines[0].split()) < embeddings_book[selected]['dim']:
#             lines = lines[1:]
#         # формируем словарь векторов
#         for line in lines:
#             values = line.split()
#             word   = delete_part_of_speech(values[0])  # отбрасываем _NOUN, _PREP, итд
#             trained_embeddings[word] = np.array(values[1:])
#     print(len(trained_embeddings))
# elif selected == 'navec':
#     trained_embeddings = Navec.load(embeddings_book[selected]['path'])
# else:
#     raise Exception("Не определен способ загрузки")


# ТОКЕНЕЗАЦИЯ СЛОВ
max_words = 20000
max_len   = 400
tokenizer = text.Tokenizer(num_words=max_words)
# составляем словарь слов (на обучающей выборке)
tokenizer.fit_on_texts(post_train)
# меняем слова на индекс в словаре (tokenizer)
xtrain_seq = tokenizer.texts_to_sequences(post_train)
xtest_seq  = tokenizer.texts_to_sequences(post_test)

# обрезаем предложения выше max_len слов и дополняем нулевыми векторами слишком короктие предложения
# (нужна равная размерность)
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xtest_pad  = sequence.pad_sequences(xtest_seq, maxlen=max_len)
word_index = tokenizer.word_index


#               читаем из файла предобученные векторы для СЛОВ ИЗ СЛОВАРЯ пачками (для оптимизации памяти)
# читаем из файла (структура у них одинакова на каждой строке массив значений, разделенных проблеами,
# первое значение - слово, остальное - его вектор)
if selected in ['glove_300', 'glove_100_twitter_multy-lang', 'russian_news', 'fasttext_rus_300']:
    trained_embeddings = {}
    with open(embeddings_book[selected]['path'], encoding='utf-8') as file:
        while True:
            lines = file.readlines(2**30)  # 1GB batches
            if not lines:
                break
            print(len(lines))
            # для fasttext не нужна первая строка файла, содержит метаданные
            # проверяем по числу значений в строке (если < размерности вектора => не вектор)
            if len(lines[0].split()) < embeddings_book[selected]['dim']:
                lines = lines[1:]
            # формируем словарь векторов
            for line in tqdm(lines):
                values = line.split()
                word   = delete_part_of_speech(values[0])  # отбрасываем _NOUN, _PREP, итд
                # если слово есть в наших текстах - сохраняем для него вектор
                if word in word_index:
                    trained_embeddings[word] = np.array(values[1:])
    print(len(trained_embeddings))
elif selected == 'navec':
    # можно использовать все, т.к. занимает мало места, читается иначе
    trained_embeddings = Navec.load(embeddings_book[selected]['path'])
else:
    raise Exception("Не определен способ загрузки")



# ЗАГРУЖАЕМ ВЕСА ДЛЯ ВСТРЕЧАЮЩИХСЯ В ТЕКСТЕ СЛОВ

# размерность текущего векторного представления слов
emb_dim   = embeddings_book[selected]['dim']

embedding_matrix = np.zeros((max_words, emb_dim))
# unknown          = trained_embeddings.get('<unk>') # 0, 0, 0...
unknown_words = 0
for word, idx in word_index.items():
    if idx < max_words:
        # получаем вектора на слово из словаря токенезатора
        embedding_vector = trained_embeddings.get(word, None)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
        else:
            unknown_words += 1
        # elif unknown:
        #     embedding_matrix[idx] = unknown
print(unknown_words)
# МОДЕЛЬ
lstm_model = tf.keras.Sequential()
lstm_model.add(tf.keras.layers.Embedding(max_words, emb_dim, trainable=False, weights=[embedding_matrix]))
lstm_model.add(tf.keras.layers.LSTM(128, return_sequences=False))
lstm_model.add(tf.keras.layers.Dropout(0.5))
# слой предсказания результата (по 1 unit па психотип), рассчитывается вероятность каждого
lstm_model.add(tf.keras.layers.Dense(len(psychotypes), activation='sigmoid'))
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(lstm_model.summary())

batch_size = 64
epochs     = 10
history    = lstm_model.fit(xtrain_pad, np.asarray(label_train), validation_data=(xtest_pad, np.asarray(label_test)), batch_size=batch_size, epochs=epochs)

train_lstm_results = lstm_model.evaluate(xtest_pad, np.asarray(label_test), verbose=0, batch_size=256)

K.clear_session()








