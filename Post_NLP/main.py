import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split

from libs.TextPreprocessing import TextPreprocessing
from libs.methods import remove_Stopwords, lemmatize_text,clean_text,stemSentence
from IPython.display import display
from tqdm import tqdm
tqdm.pandas()

from sqlalchemy import create_engine
url = 'postgresql://twin:58er2504Vb@85.172.79.228:5432/dwh'
alchemyEngine   = create_engine(url)
dbConnection    = alchemyEngine.connect()

df_profile = pd.read_sql('''
SELECT 
         Client.client_id
        ,Profile.params
        ,Feature.name

        FROM "Client" as Client

        inner join "Entrypoint" as Entrypoint on 
        Entrypoint.client_id = Client.client_id

        inner join "Profile" as Profile ON
            Profile.entrypoint_id = Entrypoint.entrypoint_id
        inner join "Feature" as Feature ON
            Feature.id = Profile.feature_id

where
    Entrypoint.datasource_id = 2

   ''', dbConnection)

psychotypes    = list('ABCDEFG')
df_psychotypes = df_profile[['client_id']].drop_duplicates()
df_psychotypes['Subject'] = np.random.choice(psychotypes, df_psychotypes.shape[0])



# оставляем последний id клиента в вк
df_profile = df_profile[df_profile['name'] == 'id'][['client_id', 'params']].drop_duplicates(keep='last')
df_profile = df_profile.rename(columns={'params': 'user_id'})

df_post    = pd.read_sql('''
SELECT 
         Client.client_id
        ,Post.owner_id
        ,Post.text::json->'text' as text

        FROM "Client" as Client
        inner join "Entrypoint" as Entrypoint on 
            Entrypoint.client_id = Client.client_id

        inner join "Post" as Post ON
            Post.entrypoint_id = Entrypoint.entrypoint_id 
WHERE
    Entrypoint.datasource_id = 2 and Post.type = 'items'
   ''', dbConnection)


display(df_psychotypes)
display(df_post.shape)

df_post = df_post.merge(df_profile, on='client_id').query("owner_id == user_id")
display(df_post.shape)
df_post = df_post.merge(df_psychotypes, on='client_id')[['text', 'Subject']]
display(df_post.shape)

from nltk.stem.porter import PorterStemmer

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping


category = pd.get_dummies(df_post['Subject'])
df_new = pd.concat([df_post, category], axis=1)
df_new = df_new.drop(columns='Subject')


# TextPreprocessing.clean_stop_words_from_sentences
# TextPreprocessing.clean_stop_words
# TextPreprocessing.clean_html_special_characters


# приведение к одному формату
df_new['new_text']  = df_new['text'].progress_apply(TextPreprocessing.clean_format, duration_log=False)
# удаление тегов html
df_new['new_text']  = df_new['new_text'].progress_apply(TextPreprocessing.clean_html, duration_log=False)
# удаление спец символов html
df_new['new_text']  = df_new['new_text'].progress_apply(TextPreprocessing.clean_html_special_characters_v2, duration_log=False)
# стоп слова
df_new['new_text']  = df_new['new_text'].progress_apply(remove_Stopwords) # работают для русского

# df_new['new_text']  = df_new['new_text'].progress_apply(lemmatize_text)    # только для английского
# знаки препинания
df_new['new_text']  = df_new['new_text'].progress_apply(TextPreprocessing.clean_text_total, duration_log=False)  # знаки препинания удаляются во всех языках

df_new['new_text']  = df_new['new_text'].progress_apply(clean_text)  # знаки препинания удаляются во всех языках
# леммматизация (ускорить, возможно есть другая модель не для английского вместо stemSentence)
df_new['new_text']  = df_new['new_text'].progress_apply(TextPreprocessing.stemming_and_lemmatization_v2, duration_log=False) # стемминг для русского
# df_new['stem_text'] = df_new['new_text'].progress_apply(stemSentence)  # ? не работает

df_new['len'] = df_new['new_text'].str.strip().str.len()
display(df_new.shape)
df_new = df_new[df_new['len'] > 0].sort_values('len', ascending=False)
display(df_new.shape)
display(df_new)