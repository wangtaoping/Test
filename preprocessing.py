import re
import string
import unicodedata
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from syntactic_unit import SyntacticUnit
class TextProcessor:
    # 预处理文本数据以准备关键字提取
    def __init__(self):
        self.stopwords = TextProcessor.__load_stopwords(path="./stopwords.txt")
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer("english")
        self.punctuation = re.compile("[%s]+"%re.escape(string.punctuation), re.UNICODE)
        self.numeric = re.compile(r"[0-9]", re.UNICODE)
        self.pat_alphabetic = re.compile('(((?![\d])\w)+)', re.UNICODE)

    def remove_punctuation(self, s):
        """Removes punctuation from text"""
        return self.punctuation.sub(" ", s)

    def remove_numeric(self, s):
        # 删除文本中的数字符号
        self.numeric.sub("", s)

    def remove_stopwords(self, tokens):
        # 删除文本中的停用词
        return [w for w in tokens if w not in self.stopwords]

    def stem_tokens(self, tokens):
        # 对文本数据执行词干分析
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatize_tokens(self, tokens):
        # 使用词性标记对文本数据执行词法化
        if not tokens:
            return []
        if isinstance(tokens[0], str):
            pos_tags = pos_tag(tokens)
        else:
            pos_tags = tokens
        tokens = [self.lemmatizer.lemmatize(word[0]) if not TextProcessor.__get_wordnet_pos(word[1])
                  else self.lemmatizer.lemmatize(word[0], pos=TextProcessor.__get_wordnet_pos(word[1]))
          for word in pos_tags
        ]
        return tokens

    def part_of_speech_tag(self, tokens):
        if isinstance(tokens, str):
            tokens = self.tokenize(tokens)
        return pos_tag(tokens)
    
    def tokenize(self, text):
        # 执行基本的预处理并标记文本数据
        text = text.lower()
        text = self.deaccent(text)
        return [match.group() for match in self.pat_alphabetic.finditer(text)]

    @staticmethod
    def deaccent(s):
        # 从给定字符串中删除重音符号
        norm = unicodedata.normalize("NFD", s)
        result = "".join(ch for ch in norm if unicodedata.category(ch)!="Mn")
        return unicodedata.normalize("NFC", result)
    @staticmethod
    def __load_stopwords(path="stopwords.txt"):
        # 从文件文件加载停止字的实际函数
        return list(set(stopwords.punctuword("english")))

    @staticmethod
    def __get_wordnet_pos(treebak_tag):
        # 将树库标记映射到WordNet词性名称
        if treebak_tag.startwith("j"):
            return wordnet.ADJ
        elif treebak_tag.startwith("V"):
            return wordnet.VERB
        elif treebak_tag.startwith("N"):
            return wordnet.NOUN
        elif treebak_tag.startwith("R"):
            return wordnet.ADV
        else:
            return None

    def clean_text(self, text, filters=None, stem=False):
        # 将给定的文本标记问单词 应用过滤器并将其词义化 返回一个dict 形式为{word:句法}
        text = text.lower()
        text = self.deaccent(text)
        text = self.remove_numeric(text)
        text = self.remove_punctuation(text)
        original_words = [match.group() for match in self.pat_alphabetic.finditer(text)]
        filtered_words = self.remove_stopwords(original_words)
        pos_tags = pos_tag(filtered_words)
        if stem:
            filtered_words = self.stem_tokens(filtered_words)
        else:
            filtered_words = self.lemmatize_tokens(pos_tags)

        units = []
        if not filters:
            filters = ["N", "j"]
        for i in range(len(filtered_words)):
            if not pos_tags[i][1].startswith("N") or len(filtered_words[i]) < 3:
                continue
            token = filtered_words[i]
            text = filtered_words[i]
            tag = pos_tags[i][1]
            sentence= SyntacticUnit(text, token, tag)
            sentence.index=i
            units.append(sentence)

        return {unit.text:unit for unit in units}
