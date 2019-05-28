#!/usr/bin/env python
# coding: utf-8

# In[2]:


import jieba.analyse
import nltk
with open("Desktop/novel.txt") as word:
    for line in word:
        jay_lyrics = nltk.text.Text(jieba.cut(line, cut_all=True))
        jay_lyrics.concordance("小女")


# In[4]:


def lexical_diversity(text):
    return len(set(text)) / len(text)
#詞的多樣性
raw = open("Desktop/novel.txt").read()
single = nltk.text.Text(jieba.cut(raw))
lexical_diversity(single)


# In[11]:


import jieba
seg_list = jieba.cut("我來到明志科技大學上課", HMM=False)
print("Full Mode: " + "/".join(seg_list))


# In[19]:


import jieba

stopWords=[]
segments=[]
remainderWords=[]

with open('Desktop/stopwords.txt', 'r') as file:
    for data in file.readlines():
        data = data.strip()
        stopWords.append(data)


# In[49]:


segments = jieba.cut('這一些鳳梨酥好吃嗎', cut_all=False)
x = list(segments)
type(x)


# In[50]:


remainderWords = list(filter(lambda a: a not in stopWords and a != '\n', x))
remainderWords


# In[ ]:





# In[34]:


import nltk


# In[35]:


nltk.download()


# In[36]:


from nltk.book import *


# In[51]:


#text4: Inaugural Address Corpus
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])


# In[52]:


len(set(text4))


# In[40]:


text4.index('awaken')


# In[41]:


text4[1:10]


# In[42]:


fdist1 = FreqDist(text4)


# In[43]:


fdist1


# In[53]:


#高頻率詞
fdist1.plot(50, cumulative=False) #True為累計頻率, False為總數


# In[54]:


#就職演說進階版
#詞偏移
import nltk
from nltk.corpus import inaugural
inaugural.fileids()
[fileid[:4] for fileid in inaugural.fileids()]


# In[46]:


cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target)) 
cfd.plot()


# In[47]:


from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    '你 住 的 巷子 裡',
    '我 租了 一間 公寓',
    '爲了 想 與 你 不期而遇',
    '高中 三年',
]
#文本轉換詞頻矩陣
vectorizer = CountVectorizer()
#計算詞語出現次數
X = vectorizer.fit_transform(corpus)
#文本關鍵字獲取
word = vectorizer.get_feature_names()
print(word)
#檢查結果
print(X.toarray())
 
from sklearn.feature_extraction.text import TfidfTransformer
 
#功能調用
transformer = TfidfTransformer()
print(transformer)
#將矩陣X轉成TF-IDF值
tfidf = transformer.fit_transform(X)


print (tfidf.toarray())


# In[48]:


__author__ = "liuxuejiang"
import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
 
if __name__ == "__main__":
    corpus=["你 住 的 巷子 裡",
		"我 租了 一間 公寓",
		"爲了 想 與 你 不期而遇",
		"高中 三年"]
    vectorizer=CountVectorizer()#
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一個fit_transform計算tf-idf，第二個fit_transform是文本轉為詞頻矩陣
    word=vectorizer.get_feature_names()#獲取詞袋模型中所有詞語
    weight=tfidf.toarray()#抽取TF-IDF的矩陣，元素a[i][j]表示j詞在i類文本中的tf-idf權重
    for i in range(len(weight)):
        print ("*****輸出第",i,"個內容權重*****")
        for j in range(len(word)):
            print (word[j],weight[i][j])


# In[ ]:




