import MeCab
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer

# 各文章における重み付け
def get_cosine_similarity(sample, test):
    # print('get_cosine_similarity sample:' + sample)
    # print('get_cosine_similarity test:' + test)
    tagger = MeCab.Tagger('-Owakati')
    transformer = TfidfTransformer()
    vectorizer = TfidfVectorizer(
        token_pattern=u'(?u)\\b\\w+\\b' #文字列長が1の単語を処理対象に含める
    )
    tfidf_sample = vectorizer.fit_transform([tagger.parse(sample).strip()])
    tfidf_test = vectorizer.transform([tagger.parse(test).strip()])
    cs_array = cosine_similarity(tfidf_sample, tfidf_test)
    return cs_array

# input question
q = input("question:")

# read faq file
faq = pd.read_csv("faq.csv", header=None).values.tolist()
faq_read = []
max_idx = -1
max_cs_sim = 0
cs_sim = 0
for i, r in enumerate(faq):
    faq_read.append(r)
    cs_sim = get_cosine_similarity(q, r[0])
    # print(cs_sim[0][0])
    if cs_sim > 0.5:
        if cs_sim > max_cs_sim:
            max_cs_sim = cs_sim
            max_idx = i

if max_idx < 0:
    print("answer:" + "該当する回答がありません。")
else:
    print("answer:" + faq_read[max_idx][1])
