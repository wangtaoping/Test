class SyntacticUnit(object):
    """
    单词已处理标记 相应的词性标记和分数的标记类
    """
    def __init__(self, text, token=None, tag=None):
        self.text = text
        self.token = token
        self.tag = tag[:2]  if tag else None  # just first two letters of tag
        self.index = -1
        self.score = -1

    # 返回一个对象的描述信息
    def __str__(self):
        return self.text + "\t" + self.token + "\n"

    def __repr__(self):
        return str(self)
