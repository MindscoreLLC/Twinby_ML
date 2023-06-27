import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from libs.TextPreprocessing import TextPreprocessing
from libs.methods import remove_Stopwords, lemmatize_text, clean_text, stemSentence
from IPython.display import display
from tqdm import tqdm
tqdm.pandas()

from navec import Navec



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
display(df_profile)

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
display(df_post.merge(df_profile, on='client_id'))

df_post = df_post.merge(df_profile, on='client_id').query("owner_id == user_id")#.iloc[:100000]
display(df_post.shape)
df_post = df_post.merge(df_psychotypes, on='client_id')#[['text', 'Subject']]
display(df_post.shape)

from nltk.stem.porter import PorterStemmer

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping


category = pd.get_dummies(df_post['Subject'])
df_new = pd.concat([df_post, category], axis=1)
df_new = df_new.drop(columns='Subject')
#
# русский внгл
# prediction type personality post facebook

# TextPreprocessing.clean_stop_words_from_sentences
# TextPreprocessing.clean_stop_words
# TextPreprocessing.clean_html_special_characters
display(TextPreprocessing.stemming_and_lemmatization_v2('собаки', duration_log=False))

# приведение к одному формату
df_new['new_text']  = df_new['text'].progress_apply(TextPreprocessing.clean_format, duration_log=False)
# удаление тегов html
df_new['new_text']  = df_new['new_text'].progress_apply(TextPreprocessing.clean_html, duration_log=False)
# удаление спец символов html
df_new['new_text']  = df_new['new_text'].progress_apply(TextPreprocessing.clean_html_special_characters_v2, duration_log=False)
# стоп слова
df_new['new_text']  = df_new['new_text'].progress_apply(remove_Stopwords)  # работают для русского

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
display(df_new['new_text'])

news = df_new['new_text'].values
label = df_new[psychotypes].values

news_train, news_test, label_train, label_test = train_test_split(news, label, test_size = 0.2, random_state = 123)
length = df_new['new_text'].str.len().max()
print(length)
tokenizer = Tokenizer(num_words = length, oov_token = '<OOV>')
tokenizer.fit_on_texts(news_train)
tokenizer.fit_on_texts(news_test)

sequences_train = tokenizer.texts_to_sequences(news_train)
sequences_train
sequences_test = tokenizer.texts_to_sequences(news_test)

padded_train = pad_sequences(sequences_train,
                             maxlen = 5,
                             padding = 'post',
                             truncating = 'post')
padded_test = pad_sequences(sequences_test,
                            maxlen = 5,
                            padding = 'post',
                            truncating = 'post')



model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=15000, output_dim=16),  # input_dim >= max_len(text)
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(254, activation='relu'),
    tf.keras.layers.Dense(128, activation= 'relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])
model.compile(loss ='categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
model.summary()
class accCallback(Callback):
   def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.98 and logs.get('val_accuracy') >= 0.98):
            print("\nAccuracy and Val_Accuracy has reached 90%!", "\nEpoch: ", epoch)
            self.model.stop_training = True

callbacks = accCallback()

auto_reduction_LR = ReduceLROnPlateau(
    monitor = 'val_accuracy',
    patience = 2, #setelah 2 epoch, jika tidak ada kenaikan maka LR berkurang
    verbose = 1,
    factor = 0.2,
    min_lr = 0.000003
)

auto_stop_learn = EarlyStopping(
    monitor = 'val_accuracy',
    min_delta = 0,
    patience = 4,
    verbose = 1,
    mode = 'auto'
)

#latih model
history = model.fit(padded_train, label_train,
                    steps_per_epoch = 30,
                    epochs = 100,
                    validation_data = (padded_test, label_test),
                    verbose = 1,
                    validation_steps = 50,
                    callbacks=[callbacks, auto_reduction_LR, auto_stop_learn],
                    )
pd.DataFrame(history.history).plot(figsize=(7, 4))
plt.grid(True)
plt.gca().set_ylim(0,3) #sumbu y

plt.show()
print(11111111)