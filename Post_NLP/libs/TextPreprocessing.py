import re
import contractions

from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

import ru_core_news_md
nlp = ru_core_news_md.load()

# загрузка стоп-слов
with open('libs/stopwords_russian.txt', encoding='utf-8') as file:
    russian_stopwords = set(file.read().split())

# загрузка html спец-символов
with open('libs/html_special_characters.txt', encoding='utf-8') as file:
    html_special_characters = set(file.read().split())

# слова-сокращения в русском
with open('libs/short_constructions.txt', encoding='utf-8') as file:
    short_constructions = set(file.read().split())


class TextPreprocessing:
    """
                                        ИНСТРУКЦИЯ К КЛАССУ TextPreprocessing
    """

    @staticmethod
    def clean_text(text):
        # приведение к одному формату и нижнему регистру
        text = text.encode('utf-8', 'ignore').decode().lower()
        # очистка от html тегов
        text = BeautifulSoup(text, 'lxml').get_text()
        # очистка от спец-символов html (&nbsp, итд)
        for special_character in html_special_characters:
            text = re.sub(special_character, ' ', text)
        # очистка от ссылок
        text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
        # очистка от email
        text = re.sub(r'\S*@\S*\s?', ' ', text)

        words = []
        for word in word_tokenize(text):
            # проверяем наличие слова в списке русских сокращений (если есть - ничего не делаем)
            if word not in short_constructions:
                # исправляем слово с библиотекой для раскрытия английских сокращений (не скоращения - не меняется)
                word = contractions.fix(word)
                # удаляем все, что не является буквой или тире (окруженное буквами)
                word = re.sub(r'[^А-Яа-яЁёA-Za-z\- ]+', ' ', word)
                word = re.sub(r'[^\w]-[^\w]', ' ', word)
                words.append(word)
        return ' '.join(words)

    @staticmethod
    def clean_extra_spaces(text):
        """Очистка текста от лишних пробелов (в начале и конце строки, множественные пробелы в середине строки
        будут заменены на 1 пробел)"""
        return re.sub(' +', ' ', text).strip()

    @staticmethod
    def remove_stopwords(text):
        stop_words = set(stopwords.words('russian')) | set(stopwords.words('english')) | set(russian_stopwords)
        words = word_tokenize(text.lower())
        sentence = [w for w in words if not w in stop_words]
        return " ".join(sentence)

    @staticmethod
    def lemmatization(text):
        """Лемматизация"""
        return ' '.join([token.lemma_ for token in nlp(text)])
