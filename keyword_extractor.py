from itertools import combinations
from queue import Queue
from graph import Graph
from preprocessing import TextProcessor
from gensim.models import KeyedVectors

class KeywordExtractor:
    # 利用TextRank算法从文本中提取关键词
    def __init__(self, word2vec=None):
        self.preprocess = TextProcessor()
        self.graph = Graph()
        if word2vec:
            print("Loading word2vec embedding...")
            self.word2vec = KeyedVectors.load_word2vec_format(word2vec, binary=True)
            print("Succesfully loaded word2vec embeddings!")
        else:
            self.word2vec = None

    def init_graph(self):
        self.preprocess = TextProcessor()
        self.graph = Graph()

    def extract(self, text, ratio=0.4, split=False, scores=False):
        """
        :param text: 要从中提取关键词的文本数据
        :param ratio:
        :param split:
        :param scores:
        :return: 从文本中提取的关键字列表
        """
        self.init_graph()
        words = self.preprocess.tokenize(text)
        tokens = self.preprocess.clean_text(text)
        for word, item in tokens.items():
            if not self.graph.has_node(item.token):
                self.graph.add_node(item.token)
        self._set_graph_edges(self.graph, tokens, words)
        del words
        KeywordExtractor.__remove_unreachable_nodes(self.graph)
        if len(self.graph.nodes()) == 0:
            return [] if split else ""
        pagerank_scores = self.__textrank()
        extracted_lemmas = KeywordExtractor.__extract_tokens(self.graph.nodes(), pagerank_scores, ratio)
        lemmas_to_word = KeywordExtractor.__lemmas_to_words(tokens)
        keywords = KeywordExtractor.__get_keywords_with_score(extracted_lemmas, lemmas_to_word)
        combined_keywords = self.__get_combined_keywords(keywords, text.split())
        return KeywordExtractor.__format_results(keywords, combined_keywords, split, scores)

    @staticmethod
    def __remove_unreachable_nodes(graph):
        for node in graph.nodes():
            if sum(graph.edge_weight((node, other))  for other in graph.neighbors(node)) == 0:
                graph.del_node(node)

    @staticmethod
    def __lemmas_to_words(tokens):
        # 返回给定网络的对应词
        lemma_to_word = {}
        for word, unit in tokens.items():
            lemma = unit.token
            if lemma in lemma_to_word:
                lemma_to_word[lemma].append(word)
            else:
                lemma_to_word[lemma] = [word]
        return lemma_to_word

    @staticmethod
    def __get_keywords_with_score(extracted_lemas, lemma_to_word):
        """
        :param extracted_lemas: tuple的列表
        :param lemma_to_word: dict of {lemma:list of words}
        :return: dict of {keyword: score}
        """
        keywords = {}
        for score, lemma in extracted_lemas:
            keyword_list = lemma_to_word[lemma]
            for keyword in keyword_list:
                keywords[keyword] = score
        return keywords

    def __textrank(self, initial_value=None, damping=0.85, covergence_threshold=0.0001):
        # 无向图的textrank的实现
        if not initial_value:
            initial_value = 1.0 / len(self.graph.nodes())
        scores = dict.fromkeys(self.graph.nodes(), initial_value)
        iteration_quantity = 0
        for iteration_number in range(100):
            iteration_quantity += 1
            convergence_achieved = 0
            for i in self.graph.nodes():
                rank = 1 - damping
                for j in self.graph.neighbors(i):
                    neighbors_sum = sum(self.graph.edge_weight((j, k)) for k in self.graph.neighbors(j))
                    rank += damping * scores[j] * self.graph.edge_weight((j, i)) / neighbors_sum
                if abs(scores[i]-rank) <= covergence_threshold:
                    convergence_achieved+=1
                scores[i] = rank
            if convergence_achieved == len(self.graph.nodes()):
                break
        return scores

    @staticmethod
    def __extract_tokens(lemmas, scores, ratio):
        lemmas.sort(key=lambda s: scores[s], reverse=True)
        length = len(lemmas) * ratio
        return [(scores[lemmas[i]], lemmas[i]) for i in range(int(length))]

    def __get_combined_keywords(self, _keywords, split_text):
        """
        :param _keywords:  dict of keywords:scores
        :param split_text:  list of strings
        :return: combined_keywords:list
        """
        result = []
        _keywords = _keywords.copy()
        len_text = len(split_text)
        for i in range(len_text):
            word = self._strip_word(split_text[i])
            if word in _keywords:
                combined_word = [word]
                if i+1 == len_text:
                    result.append(word)
                for j in range(i+1, len_text):
                    other_word = self._strip_word(split_text[j])
                    if other_word in _keywords and other_word == split_text[j] and other_word not in combined_word:
                        combined_word.append(other_word)
                    else:
                        for keyword in combined_word:
                            _keywords.pop(keyword)
                        result.append(" ".join(combined_word))
                        break
        return result

    @staticmethod
    def __format_results(_keywords, combined_keywords, split, scores):
        combined_keywords.sort(key=lambda w:KeywordExtractor.__get_average_score(w, _keywords), reverse=True)
        if scores:
            return [(word, KeywordExtractor.__get_average_score(word, _keywords)) for word in combined_keywords]
        if split:
            return combined_keywords
        return "\n".join(combined_keywords)
    @staticmethod
    def __get_average_score(concept, _keywords):
        # 计算平均分
        word_list = concept.split()
        word_counter = 0
        total = 0
        for word in word_list:
            total += _keywords[word]
            word_counter += 1
        return total/word_counter
