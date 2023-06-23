import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
from random import choice

psychotypes    = list('ABCDEFG')
df_psychotypes = df_profile[['client_id']].drop_duplicates()
df_psychotypes['Subject'] = np.random.choice(psychotypes, df_psychotypes.shape[0])
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
# print(1)
# display(df_profile.columns)
# df = pd.read_csv('data/subjects-questions/subjects-questions.csv')
# display(df)
# df.Subject.unique()
# df.isna().sum()
# df.head()
# print(1)

category = pd.get_dummies(df_post.Subject)
df_new = pd.concat([df_post, category], axis=1)
df_new = df_new.drop(columns='Subject')
# print(1)
df_new['new_text'] = df_new['text'].progress_apply(remove_Stopwords)
df_new['new_text'] = df_new['text'].progress_apply(lemmatize_text)
df_new['new_text'] = df_new['text'].progress_apply(clean_text)
# print(1)
print(df_new)
#
