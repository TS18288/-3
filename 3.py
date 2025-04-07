def forward_max_matching(vocab, sentence):
    """
    基于正向最大匹配算法 (FMM) 的中文分词函数

    参数：
    vocab (list): 词典，包含了所有可能的词汇
    sentence (str): 需要分词的句子

    返回：
    list: 分词结果的列表
    """
    fmmresult = []  # 存储分词结果的列表
    max_len = max([len(item) for item in vocab])  # 获取词典中最长词的长度
    start = 0  # 分词的起始位置

    # 开始遍历句子，直到处理完整个句子
    while start != len(sentence):
        index = start + max_len  # 尝试匹配最大长度的词
        if index > len(sentence):  # 如果索引超出句子长度，修正为句子末尾
            index = len(sentence)

        # 从当前起始位置尝试从最大长度开始逐步缩小词的长度进行匹配
        while index > start:
            current_substr = sentence[start:index]  # 截取当前子串
            # 如果子串在词典中，或者子串长度为1，则认为是一个有效词
            if current_substr in vocab or len(current_substr) == 1:
                fmmresult.append(current_substr)  # 将有效词加入结果列表
                start = index  # 更新起始位置，跳过已处理的部分
                break  # 找到一个有效词后跳出内层循环继续处理下一个子串
            index -= 1  # 如果没有匹配到有效词，缩短子串长度再试

    return fmmresult  # 返回最终的分词结果
def reverse_directional_max_matching(vocab, sentence):
    """
    Reverse Maximum Matching (RMM) 分词算法。
    从句子的末尾开始，尝试从词典中匹配最长的词直到句子被分割。

    Args:
    vocab (list): 词典，包含了所有可能的词汇
    sentence (str): 需要分词的句子

    返回：
    list: 分词后的结果，按顺序返回分词列表
    """
    rmmresult = []  # 存储分词结果
    max_len = max([len(item) for item in vocab])  # 获取词典中最大词的长度
    start = len(sentence)  # 从句子的末尾开始

    while start != 0:  # 直到处理完整个句子
        index = start - max_len  # 尝试从当前位置往前推最大长度的子串
        if index < 0:
            index = 0  # 防止下标越界，调整为从0开始

        while index < start:  # 向前查找直到找到匹配的词
            current_substr = sentence[index:start]  # 截取当前子串

            # 如果当前子串在词典中，或当前子串长度为1（即单个字符）
            if current_substr in vocab or len(current_substr) == 1:
                rmmresult.insert(0, current_substr)  # 匹配成功，插入到结果列表的开头
                start = index  # 更新起始位置，继续向前匹配
                break  # 找到一个词后跳出内层循环
            index += 1  # 如果当前子串没有匹配，向前移动一个字符继续尝试

    return rmmresult  # 返回最终的分词结果
def bi_directional_matching(vocab, sentence):
    # 获取正向和反向最大匹配的分词结果
    res1 = forward_max_matching(vocab, sentence)
    res2 = reverse_directional_max_matching(vocab, sentence)

    len_res1, len_res2 = len(res1), len(res2)  # 保存长度

    # 如果两个结果的长度相同
    if len_res1 == len_res2:
        # 如果两个结果相同，直接返回
        if res1 == res2:
            return res1
        else:
            # 统计每个结果中长度为1的词的数量
            res1_sn = sum(1 for i in res1 if len(i) == 1)
            res2_sn = sum(1 for i in res2 if len(i) == 1)
            # 返回包含较少单字符词的结果
            return res1 if res1_sn < res2_sn else res2
    else:
        # 返回词数较少的结果
        return res1 if len_res1 < len_res2 else res2
vocab = ['我们', '今天', '在', '野生动物园', '玩']
sentence = '我们是今天在野生动物园玩了'
fmm_result = forward_max_matching(vocab, sentence)
rmm_result = reverse_directional_max_matching(vocab, sentence)
print(fmm_result)
print(rmm_result)
bm_result = bi_directional_matching(vocab, sentence)
print(bm_result)