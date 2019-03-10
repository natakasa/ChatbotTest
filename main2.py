import MeCab
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 文章から単語を抽出
def words(text):
    words = []
    tagger = MeCab.Tagger('-Ochasen')
    node = tagger.parseToNode(text)
    while node:
        word_type = node.feature.split(",")[0]
        if word_type in ["名詞"]:
            words.append(node.surface)
            print(node.surface)
        node = node.next
    return words

# 各文章における重み付け
def vecs_array(documents):
    print('documents:' + documents)
    docs = np.array(documents)
    vectorizer = TfidfVectorizer(
        analyzer=words,
#        stop_words='|',
#        min_df=1,
        token_pattern='(?u)\\b\\w+\\b' #文字列長が1の単語を処理対象に含める
    )
    vecs = vectorizer.fit_transform(docs)
    return vecs.toarray()

# input question
q = input("question:")

# read faq file
faq = pd.read_csv("faq.csv").values.tolist()
f = []
for i, r in enumerate(faq):
    print('r:' + r[0])
    f.append(r[0])
    cs_array = cosine_similarity(vecs_array(q), vecs_array(r[0]))
    print('cs_array:' + cs_array)
