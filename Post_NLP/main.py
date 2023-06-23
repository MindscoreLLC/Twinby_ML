import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split

from libs.methods import remove_Stopwords, lemmatize_text,clean_text,stemSentence
import string
string.punctuation
from nltk.stem.porter import PorterStemmer


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping


df = pd.read_csv('data/subjects-questions/subjects-questions.csv')
df.Subject.unique()
df.isna().sum()
df.head()

#plt.figure(figsize = (8, 8))
#sns.countplot(df['Subject'])

category = pd.get_dummies(df.Subject)
df_new = pd.concat([df, category], axis = 1)
df_new = df_new.drop(columns = 'Subject')

df_new['new_eng'] = df_new['eng'].apply(remove_Stopwords)
df_new['new_eng'] = df_new['eng'].apply(lemmatize_text)
df_new['new_eng'] = df_new['eng'].apply(clean_text)
df_new
print()