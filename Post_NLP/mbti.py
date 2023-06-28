# import sys
# sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from libs.TextPreprocessing import TextPreprocessing
from libs.methods import remove_Stopwords, lemmatize_text, clean_text, stemSentence
from IPython.display import display
from tqdm import tqdm
tqdm.pandas()

# from navec import Navec
from nltk.stem.porter import PorterStemmer

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping


from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import regex as re
import transformers
from keras import backend as K
import plotly.express as px

from transformers import BertTokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split


from sqlalchemy import create_engine



#                           ИСХОДНЫЕ ДАННЫЕ

from psycopg import Connection

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

df_post = df_post.merge(df_profile, on='client_id').query("owner_id == user_id")#.iloc[:100000]
display(df_post.shape)
df_post = df_post.merge(df_psychotypes, on='client_id')#[['text', 'Subject']]
display(df_post.shape)



#                           ПРЕДОБРАБОТКА


# приведение к одному формату
df_post['new_text']  = df_post['text'].progress_apply(TextPreprocessing.clean_format, duration_log=False)
# удаление тегов html
df_post['new_text']  = df_post['new_text'].progress_apply(TextPreprocessing.clean_html, duration_log=False)
# удаление спец символов html
df_post['new_text']  = df_post['new_text'].progress_apply(TextPreprocessing.clean_html_special_characters_v2, duration_log=False)
# стоп слова
df_post['new_text']  = df_post['new_text'].progress_apply(remove_Stopwords)  # работают для русского

# знаки препинания
df_post['new_text']  = df_post['new_text'].progress_apply(TextPreprocessing.clean_text_total, duration_log=False)  # знаки препинания удаляются во всех языках

# леммматизация (ускорить, возможно есть другая модель не для английского вместо stemSentence)
df_post['new_text']  = df_post['new_text'].progress_apply(TextPreprocessing.stemming_and_lemmatization_v2, duration_log=False) # стемминг для русского


df_post['len'] = df_post['new_text'].str.strip().str.len()
# отбрасываем пустые посты
df_post = df_post[df_post['len'] > 0].sort_values('len', ascending=False)
display(df_post.shape)


#                           СБОР ПОСТОВ НА КАЖДОГО ПОЛЬЗОВАТЕЛЯ

df_new = df_post.groupby('client_id').agg({'new_text': ' '.join, 'Subject': 'first'})
display(df_new)


#Check if TPU is available
use_tpu = False
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.MirroredStrategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


#Split dataset


posts  = df_new['new_text'].values
labels = df_new['Subject'].values
train_data, test_data = train_test_split(df_new, random_state=0, test_size=0.2)

train_size = len(train_data)
test_size  = len(test_data)
display(train_size, test_size)




#                           ТОКЕНЕЗАЦИЯ

#Initialize Bert tokenizer and masks

bert_model_name = 'cointegrated/rubert-tiny2' # cointegrated/rubert-tiny2    bert-base-uncased

tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
MAX_LEN = 511

def tokenize_sentences(sentences, tokenizer, max_seq_len = MAX_LEN):
    tokenized_sentences = []

    for sentence in tqdm(sentences):
        tokenized_sentence = tokenizer.encode(
                            sentence,                  # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_seq_len,  # Truncate all sentences.
                    )
        
        tokenized_sentences.append(tokenized_sentence)
        
    return tokenized_sentences

def create_attention_masks(tokenized_and_padded_sentences):
    attention_masks = []

    for sentence in tokenized_and_padded_sentences:
        att_mask = [int(token_id > 0) for token_id in sentence]
        attention_masks.append(att_mask)

    return np.asarray(attention_masks)

train_input_ids = tokenize_sentences(train_data['new_text'], tokenizer, MAX_LEN)
train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
train_attention_masks = create_attention_masks(train_input_ids)

test_input_ids = tokenize_sentences(test_data['new_text'], tokenizer, MAX_LEN)
test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
test_attention_masks = create_attention_masks(test_input_ids)

#Create train and test datasets
BATCH_SIZE=2
NR_EPOCHS=20


#Define f1 functions for evaluation
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


#                           ПОСТРОЕНИЕ МОДЕЛИ

def create_model(): 
    input_word_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32,
                                           name="input_word_ids")
    bert_layer = transformers.TFBertModel.from_pretrained(bert_model_name)
    bert_outputs = bert_layer(input_word_ids)[0]
    pred = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(bert_outputs[:,0,:])
    
    model = tf.keras.models.Model(inputs=input_word_ids, outputs=pred)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.00002), metrics=['accuracy', f1_m, precision_m, recall_m])
    return model


use_tpu = False
from keras import backend as K 
K.clear_session()

if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = create_model()
else:
    model = create_model()
    
model.summary()

types = np.unique(df_new['Subject'].values)

def get_type_index(string):
    return list(types).index(string)

train_data['type_index'] = df_new['Subject'].apply(get_type_index)
display(train_data)
one_hot_labels = tf.keras.utils.to_categorical(train_data.type_index.values, num_classes=NUM_CLASSES)

model.fit(np.array(train_input_ids), one_hot_labels, verbose = 1, epochs = NR_EPOCHS, batch_size = BATCH_SIZE,  callbacks = [tf.keras.callbacks.EarlyStopping(patience = 5)])









#                           ТЕСТИРОВАНИЕ

test_data['type_index'] = df_new['Subject'].apply(get_type_index)
test_data
test_labels = tf.keras.utils.to_categorical(test_data.type_index.values, num_classes=len(psychotypes))
model.evaluate(np.array(test_input_ids), test_labels)


cols = df_post['Subject'].unique()
cols = cols.tolist()

colnames = ['sentence']
colnames = colnames+cols

df_predict = pd.DataFrame(columns = colnames)
sentence = "Time to debate on it. Strike at the weakest point and make others cry with facts"

df_predict.loc[0, 'sentence'] = sentence
sentence_inputs = tokenize_sentences(df_predict['sentence'], tokenizer, MAX_LEN)
sentence_inputs = pad_sequences(sentence_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
prediction = model.predict(np.array(sentence_inputs))
df_predict.loc[0, cols] = prediction

df_predict

