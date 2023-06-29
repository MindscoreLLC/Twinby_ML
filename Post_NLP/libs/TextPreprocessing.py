import re
import pandas as pd
import nltk
import pymorphy2
# import spacy

from time import time
from nltk.tokenize import sent_tokenize, word_tokenize
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import snowball
import spacy
import ru_core_news_md
nlp = ru_core_news_md.load()

import os
# загрузка стоп-слов
file = open('libs/stopwords_russian.txt', 'r', encoding='utf-8')

russian_stopwords = file.read().split()
file.close()

# загрузка html спец-символов
file = open('libs/html_special_characters.txt', 'r', encoding='utf-8')
html_special_characters = file.read().split()
file.close()




class TextPreprocessing:
    '''
                                        ИНСТРУКЦИЯ К КЛАССУ TextPreprocessing
            
        содержит методы для преобразования и анализа текста:
        clean_html         —— очистка текста от html-тегов
        clean_format       —— приведение текста к единому регистру и формату (нижний регистр, кодировка utf-8)
        lemmatization      —— лемматизация текста
        drop_duplicates    —— метод, проверяющий наличие дубликатов описаний в датафрейме, может вернуть очищенных от них датафрейм
        drop_short_descriptions —— метод, проверяющий наличие коротких (менее 1 предложения) описаний в датафрейме,
                                   может вернуть очищенных от них (коротких описаний) датафрейм
    
        методы данного класса, предназначенные для работы со строками, также могут обрабатывать текстовое поле датафрейма
        (df['desc'])
    '''

    def process_text_in_df(function):
        '''                                 
                                            декоратор
           метод позволяет другим методам работы с текстом преобразовать весь датафрейм

           function: функция, которая должна быть вызвана для текстового объекта или датафрейма
           
           возвращает функцию, которая использует необходимую функцию преобразования текста для нужного типа данных
           (датафрейм или строка), замещает исходный метод класса
           
        '''
        def wrapped(text_obj, *args, **kwargs):
            if isinstance(text_obj, str): # если метод необходимо вызвать для экземпляра класса str
                # вызов необходимой функции для текстового объекта
                return function(text_obj, *args, **kwargs)
            elif isinstance(text_obj, pd.DataFrame): # если метод необходимо вызвать для экземпляра класса pd.DataFrame
                transformed_df         = text_obj.copy() # создается копия исходного датарейма
                # изменение описания товаров в соответствии с вызванной функцией
                transformed_df['desc'] = transformed_df['desc'].apply(lambda text: function(text, *args, **kwargs)) 
                return transformed_df
            else: # если метод необходимо вызвать для экземпляра класса, отличного от str и pd.DataFrame
                raise Exception('Неправильный тип данных параметра text_obj, должен быть строкой или датафреймом!')
        return wrapped
    
    
    
    def time_log(function):
        '''
                                     декоратор
           метод рассчитывает время работы передаваемой функции и возвращает его вместе с результатом выполнения функции,
           если в вызываемой функции будет указан параметр duration_log=True
           
        '''
        def wrapped(*args, duration_log=True, **kwargs):
            time_start = time()
            result     = function(*args, **kwargs)
            time_end   = time()
            if duration_log:
                return result, time_end - time_start
            else:
                return result
            
        return wrapped
    
    
    @staticmethod
    @time_log
    @process_text_in_df
    def clean_html(text_obj):
        '''метод очистки текста от html-тегов
           поведение метода зависит от типа передаваемого текстового объекта [text_obj]
           
           text_obj: может быть - 1) экземпляр класса str, тогда метод возвращает измененную строку
                                  2) экземпляр класса pd.DataFrame, тогда методов возвращает преобразованный датафрейм
           если text_obj является датафреймом, он должен иметь поле ['desc']
           
           используется декоратор CreationBagWords.process_text_in_df, благодаря которому в данный метод
           будет передана строка, даже если при ее вызове был указан датафрейм в качестве параметра text_obj 
        '''
        return BeautifulSoup(text_obj, 'lxml').get_text() # очистка от html тегов

    
    
    @staticmethod
    @time_log
    @process_text_in_df
    def clean_text_total(text_obj):
        '''очистка текста от любых символов, кроме букв русского и английского алфавита, цифр
           поведение метода зависит от типа передаваемого текстового объекта [text_obj]
           
           text_obj: может быть - 1) экземпляр класса str, тогда метод возвращает измененную строку
                                  2) экземпляр класса pd.DataFrame, тогда методов возвращает преобразованный датафрейм
           если text_obj является датафреймом, он должен иметь поле ['desc']
           
           используется декоратор CreationBagWords.process_text_in_df, благодаря которому в данный метод
           будет передана строка, даже если при ее вызове был указан датафрейм в качестве параметра text_obj 
        '''
        return re.sub(r'[^А-Яа-яЁёA-Za-z0-9 ]+', ' ', text_obj)
    
    
    @staticmethod
    @time_log
    @process_text_in_df
    def clean_extra_spaces(text_obj):
        '''очистка текста от лишних пробелов (в начале и конце строки, множественные пробелы в середине строки будут заменены на 1 пробел)
        '''
        return re.sub(' +', ' ', text_obj).strip()
        

    @staticmethod
    @time_log
    @process_text_in_df
    def clean_format(text_obj):
        '''метод преобразования текста в нижний регистр и кодировку utf-8
           поведение метода зависит от типа передаваемого текстового объекта [text_obj]
           
           text_obj: может быть - 1) экземпляр класса str, тогда метод возвращает измененную строку
                                  2) экземпляр класса pd.DataFrame, тогда методов возвращает преобразованный датафрейм
           если text_obj является датафреймом, он должен иметь поле ['desc']
           
           используется декоратор CreationBagWords.process_text_in_df, благодаря которому в данный метод
           будет передана строка, даже если при ее вызове был указан датафрейм в качестве параметра text_obj 
        '''
        return text_obj.encode('utf-8', 'ignore').decode().lower()
        
        
    
    @staticmethod
    @time_log
    @process_text_in_df
    def get_sentences(text_obj, tokenize_words=False):
        '''метод получения списка предложений из текста
           поведение метода зависит от типа передаваемого текстового объекта [text_obj]
           5e
           text_obj: может быть - 1) экземпляр класса str, тогда метод возвращает измененную строку
                                  2) экземпляр класса pd.DataFrame, тогда методов возвращает преобразованный датафрейм
           если text_obj является датафреймом, он должен иметь поле ['desc']
           tokenize_words: bool (True или False), параметр, указывающий, нужна ли токенизация слов в предложениях
                           (необходимо ли разбить текст не только на предложения, но и слова)
                           если указано True, метод возвращает матрицу из список слов, которые участвуют в предложениях
           используется декоратор CreationBagWords.process_text_in_df, благодаря которому в данный метод
           будет передана строка, даже если при ее вызове был указан датафрейм в качестве параметра text_obj 
        '''
        if tokenize_words:
            return [word_tokenize(sentence) for sentence in sent_tokenize(text_obj)]
        return sent_tokenize(text_obj) # получение списка предложений из текста, функция из библиотеки nltk
     

    @staticmethod
    @time_log
    def drop_duplicates(df: pd.DataFrame, delete=True):
        '''
        функция проверки на дубли и их устранение
        df:     экземпляр класса pd.DataFrame с данными описаний товаров, должен содержать поле ['desc']
        delete: True  - возвращает датафрейм без повторяющих описаний товара
                False - возвращает bool-значение, присутствуют ли дубликаты
        '''
        without_duplicates = df[['desc']].drop_duplicates().index
        comparison         = len(without_duplicates) != len(df)
        if delete:
            return df.loc[without_duplicates, :]
        else:
            return comparison
        
        
    @staticmethod
    @time_log
    def drop_short_descriptions(df: pd.DataFrame, delete=True, min_sentences=2, min_symbols=10):
        '''
        функция проверки на наличие коротких описаний и их устранения
        сначала устраняются предложения меньше минимальной длины (в символах, указывается в качестве параметра)
        df:     экземпляр класса pd.DataFrame с данными описаний товаров, должен содержать поле ['desc']
        delete: True  - возвращает датафрейм без описаний товара, состоящих из одного предложения
                False - возвращает bool-значение, присутствуют ли описания, состоящие из одного предложения
        '''
        def correct_desription(description, min_symbols):
            """проверка описания (списка предложений) на наличие необходимой длины"""
            # удаление предложение меньше минимальной длины
            description = [sentence for sentence in description if len(sentence) >= min_symbols]
            return description
            
        df_modified                = df.copy()
        df_modified['desc']        = df_modified['desc'].apply(lambda x: correct_desription(x, min_symbols))
        df_modified                = df_modified[df_modified['desc'].apply(lambda x: len(x) >= min_sentences)]
        
        comparison                 = sum(df_modified['desc'].apply(lambda x: len(x))) != sum(df['desc'].apply(lambda x: len(x)))
        if delete:
            return df_modified
        else:
            return comparison
        

    @staticmethod
    @time_log
    def clean_stop_words_from_sentences(df):
        russian_stop_words = stopwords.words('russian') + russian_stopwords 
        modified_df = df.copy()
        def delete_stop_words_from_sentence(sentence: str) -> str:
            return ' '.join([word for word in sentence.split() if re.sub(r'[^А-Яа-яЁё]+', ' ', word) not in russian_stop_words])
        modified_df['desc'] = modified_df['desc'].apply(lambda one_desc: [delete_stop_words_from_sentence(sentence) for sentence in one_desc])
        return modified_df
    
    
    @staticmethod
    @time_log
    @process_text_in_df
    def clean_stop_words(text_obj):
        russian_stop_words = stopwords.words('russian') + russian_stopwords 
        return ' '.join([word for word in text_obj.split() if re.sub(r'[^А-Яа-яЁё]+', ' ', word) not in russian_stop_words])

    
    @staticmethod
    @time_log
    @process_text_in_df
    def clean_html_special_characters(text_obj):
        return ' '.join([word for word in text_obj.split() if word not in html_special_characters])
    
    
    
    @staticmethod
    @time_log
    @process_text_in_df
    def clean_html_special_characters_v2(text_obj):
        """метод очистки текста от спец символов html
           для каждого символа из файла html_special_characters.txt составляется регулярное выражение,
           устраняющее его из текста
        """
        for special_character in html_special_characters:
            text_obj = re.sub(special_character, '', text_obj)
        return text_obj
    


    @staticmethod
    @time_log
    @process_text_in_df
    def delete_short_words(text_obj, longer_then=5):
        """удаление из текста слов, длина которых меньше указаанной"""
        return ' '.join([word for word in text_obj.split() if len(word) >= longer_then]) # удаление слов длиной короче n символов

    
    
    @staticmethod
    @time_log
    @process_text_in_df
    def stemming_and_lemmatization(text_obj, progress=[], cashe={}):
        '''
        cashe: 
        если score N (по умолчанию 0.6) то к этому слову применяем не лемитизацию а стемминг
        '''  
        # progress.append(1)
        # display(len(progress)) # счетчик расчета значений

        processed_text = []
        stemmer        = snowball.RussianStemmer() # стеммер для русских слов
        for word in text_obj.split(): # применяем лемматезацию или стемминг к словам из текста
            if word in cashe:
                processed_text.append(cashe[word])
            else:
                pymorphy_analysis = pymorphy2.MorphAnalyzer().parse(word)[0]
                processed_text.append(pymorphy_analysis.normal_form)
                cashe[word] = processed_text[-1]
        return ' '.join(processed_text)
                #if pymorphy_analysis.score >= 0.7: # если вероятность интерпретации слова достаточно высокая, то можно применяь лемматезацию 
                    #processed_text.append(pymorphy_analysis.normal_form) # приведение слова к начальной форме
                #elif 0.7 > pymorphy_analysis.score >= 0.4:
                    #processed_text.append(spacy.load("ru_core_news_sm")(word)[0].lemma_)
               # else: # иначе применяется стемминг (удаление окончания слова)
                    #processed_text.append(stemmer.stem(word))

    @staticmethod
    @time_log
    @process_text_in_df
    def stemming_and_lemmatization_v2(text_obj):
        """Лемматизация"""
        return ' '.join([token.lemma_ for token in nlp(text_obj)])
