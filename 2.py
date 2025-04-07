import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


class EmailClassifier:
    def __init__(self, feature_type='frequency', top_num=100):
        """
        初始化分类器
        :param feature_type: 特征类型，'frequency'(高频词)或'tfidf'(TF-IDF)
        :param top_num: 特征数量
        """
        self.feature_type = feature_type
        self.top_num = top_num
        self.model = MultinomialNB()
        self.top_words = None
        self.vectorizer = None

    def clean_text(self, text):
        """清洗文本"""
        text = re.sub(r'[.【】0-9、——。，！~\*]', '', text)
        words = cut(text)
        return [word for word in words if len(word) > 1]

    def get_words(self, filename):
        """读取文件并返回清洗后的词语列表"""
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.clean_text(text)

    def get_texts(self, filenames):
        """获取所有文件的文本内容"""
        texts = []
        for filename in filenames:
            words = self.get_words(filename)
            texts.append(' '.join(words))  # 用空格连接词语
        return texts

    def get_top_words(self, filenames):
        """获取高频词特征"""
        all_words = []
        for filename in filenames:
            all_words.append(self.get_words(filename))

        # 统计词频
        freq = Counter(chain(*all_words))
        return [i[0] for i in freq.most_common(self.top_num)]

    def extract_features(self, filenames, fit=False):
        """提取特征"""
        texts = self.get_texts(filenames)

        if self.feature_type == 'frequency':
            # 高频词特征
            if fit or self.top_words is None:
                self.top_words = self.get_top_words(filenames)

            features = []
            for text in texts:
                words = text.split()
                word_map = list(map(lambda word: words.count(word), self.top_words))
                features.append(word_map)
            return np.array(features)

        elif self.feature_type == 'TF-IDF':
            # TF-IDF特征
            if fit:
                self.vectorizer = TfidfVectorizer(
                    tokenizer=self.clean_text,
                    max_features=self.top_num,
                    token_pattern=None
                )
                features = self.vectorizer.fit_transform(texts)
            else:
                features = self.vectorizer.transform(texts)
            return features.toarray()

        else:
            raise ValueError("feature_type必须是'frequency'或'TF-IDF'")

    def train(self, train_files):
        """训练模型"""
        # 提取特征
        train_features = self.extract_features(train_files, fit=True)

        # 0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0
        train_labels = np.array([1] * 127 + [0] * 24)

        # 训练模型
        self.model.fit(train_features, train_labels)

    def predict(self, filename):
        """预测单个文件"""
        # 提取特征
        features = self.extract_features([filename])

        # 预测结果
        result = self.model.predict(features)
        return '垃圾邮件' if result == 1 else '普通邮件'

    def evaluate(self, test_files):
        """评估多个测试文件"""
        results = {}
        for test_file in test_files:
            results[test_file] = self.predict(test_file)
        return results


def main(feature_type='frequency'):
    # 训练文件列表
    train_files = [f'邮件_files/{i}.txt' for i in range(151)]
    # 测试文件列表
    test_files = [f'邮件_files/{i}.txt' for i in range(151, 156)]

    # 创建并训练分类器
    classifier = EmailClassifier(feature_type=feature_type, top_num=100)
    classifier.train(train_files)

    # 评估测试文件
    results = classifier.evaluate(test_files)

    # 打印结果
    print(f"\n使用{feature_type}特征的结果:")
    for filename, prediction in results.items():
        print(f"{filename}分类情况: {prediction}")


if __name__ == '__main__':
    # 使用高频词特征
    main(feature_type='frequency')

    # 使用TF-IDF特征
    main(feature_type='TF-IDF')