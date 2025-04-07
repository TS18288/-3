import re
import os
import numpy as np
import jieba
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE  # 过采样

def get_text(filename):
    """读取文本并进行清理"""
    with open(filename, 'r', encoding='utf-8') as fr:
        text = fr.read()
    text = re.sub(r'[.【】0-9、——。，！~\*]', '', text)  # 过滤特殊字符
    words = list(jieba.cut(text))  # 结巴分词
    words = [word for word in words if len(word) > 1]  # 过滤单字符词
    return ' '.join(words)  # 返回字符串（空格分割），用于 TF-IDF

# 读取所有邮件数据
file_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
documents = [get_text(file) for file in file_list]

# **使用 TfidfVectorizer 计算 TF-IDF**
vectorizer = TfidfVectorizer(max_features=100)  # 取前 100 重要特征
tfidf_matrix = vectorizer.fit_transform(documents)

# 标签设置：前 127 封邮件为垃圾邮件（1），后 24 封为普通邮件（0）
labels = np.array([1] * 127 + [0] * 24)

# **应用 SMOTE 进行过采样**
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(tfidf_matrix, labels)

# **训练 Naive Bayes 朴素贝叶斯分类器**
model = MultinomialNB()
model.fit(X_resampled, y_resampled)

def predict(filename):
    """对未知邮件进行分类"""
    text = get_text(filename)
    text_tfidf = vectorizer.transform([text])  # 计算 TF-IDF 向量
    result = model.predict(text_tfidf)
    return '垃圾邮件' if result[0] == 1 else '普通邮件'

# **测试新的邮件**
for i in range(151, 156):
    print(f'{i}.txt 分类情况: {predict(f"邮件_files/{i}.txt")}')