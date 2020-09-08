# -*- coding: utf-8 -*-
import jieba
from keras.preprocessing.text import Tokenizer
from gensim.models import word2vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def extract_data(file):
    startword_list= ["本院认为"]
    endword_list = ["据此", "鉴此"]
    extract_sents = []
    source_sentence = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            start_index = 0
            end_index = 0
            sent_list = jieba.lcut(line.strip())
            for i in range(len(startword_list)):
                if startword_list[i] in sent_list:
                    start_index = sent_list.index(startword_list[i])

            for j in range(len(endword_list)):
                if endword_list[j] in sent_list:
                    end_index = sent_list.index(endword_list[j])
                else:
                    end_index =sent_list.index(sent_list[-1])
            print(sent_list[start_index: end_index])
            extract_sents.append(sent_list[start_index: end_index])
            source_sentence.append("".join(sent_list[start_index:end_index]))
    return extract_sents,source_sentence
def process_data(file):
    extract_sents,source_sentence = extract_data(file)
    # 求句子的最大长度
    max_len = 0
    for sent in extract_sents:
        length = 0
        for word in sent:
            length += 1
        if (length>max_len): max_len=length
    # fit_on_texts: 函数可以将输入文本中的每个词编号 编号是根据词频的 词频越大编号越小
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(extract_sents)
    vocab = tokenizer.word_index  # 得到每个词的编号
    print(vocab)

    # 设置词语向量纬度
    num_features = 100
    # 保证被忽略词语的最低频度, 对于小预料, 设置为1才可能输出所有的词, 因为有的词可能在每个句子中只出现一次
    min_word_count = 1
    # 设置并行化使用cpu计算核心数量
    num_workers = 4
    # 设置词语上下窗口大小
    context = 4
    model = word2embedding(extract_sents, num_workers, num_features, min_word_count, context)

    # 利用训练后的word2vec自定义Embedding
    word_embedding = {}
    count = 0
    for word, i in vocab.items():
        try:
            # model.wv[word]存的就是这个word的词向量
            word_embedding[word] = model.wv[word]
        except KeyError:
            continue
    sentence_vectors = sentece_2_vec(extract_sents, word_embedding)
    # 计算句子之间的余弦相似度
    sim_mat = np.zeros([len(source_sentence), len(source_sentence)])

def word2embedding(extract_sents, num_workers, num_features, min_word_count, context):
    # 开始训练
    model = word2vec.Word2Vec(extract_sents, workers=num_workers, size=num_features, min_count=min_word_count, window=context)
    model.init_sims()
    # 如果有需要的话, 可以输入一个路径, 保存训练好的模型
    # model.save("w2vModel1")
    # 加载模型
    # model = word2vec.Word2Vec.load("w2vModel1")
    return model

def sentece_2_vec(sentence_word_list, word_embedding):
    sentence_vectors = []
    for line in sentence_word_list:
        if len(line) != 0:
            # 如果句子中的词语不在字典中,那么把embedding设为100纬元素为0的向量
            #得到句子中全部词的词向量后, 求平均值,得到句子的向量表示
            v = np.round(sum(word_embedding.get(word, np.zeros((100, ))) for word in line)/(len(line)))
        else:
            # 如果句子是[], 那么就向量表示为100纬元素为0个向量
            v = np.zeros((100, ))
        sentence_vectors.append(v)
    return sentence_vectors
if __name__ == '__main__':
    # process_data("./text.txt")
    from gensim.models import KeyedVectors
    pass