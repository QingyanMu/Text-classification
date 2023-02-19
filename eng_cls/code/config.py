# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 14:15
# @Author  : chile
# @Email   : realchilewang@foxmail.com
# @File    : config.py
# @Software: PyCharm


class Config(object):
    def __init__(self):
        self.save_model = ''  # 模型路径
        self.result_file = 'result/'
        self.label_name = ['negative', 'neutral', 'positive']
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.pretrainning_model = 'bert'
        self.embed_dense = 512

        self.decay_rate = 0.5  # 学习率衰减参数

        self.train_epoch = 20  # 训练迭代次数

        self.learning_rate = 1e-4  # 下接结构学习率
        self.embed_learning_rate = 5e-5  # 预训练模型学习率

        if self.pretrainning_model == 'bert':
            model = 'D:\\预训练模型\\en\\'  # 英文bert-base
        elif self.pretrainning_model=='nezha':
            model = '/home/wangzhili/pretrained_model/Torch_model/pre_model_nezha_base/'  # 中文nezha-base
        else:
            raise KeyError('albert nezha roberta bert bert_wwm is need')
        self.cls_num = 3
        self.sequence_length = 64
        self.batch_size = 128

        self.model_path = model

        self.bert_file = model + 'pytorch_model.bin'
        self.bert_config_file = model + 'config.json'
        self.vocab_file = model + 'vocab.txt'

        self.use_origin_bert = 'ori'  # 'ori':使用原生bert, 'dym':使用动态融合bert,'weight':初始化12*1向量
        self.is_avg_pool = 'ori'  #  dym, max, mean, weight
        self.model_type = 'bilstm'  # bilstm; bigru,gat

        self.rnn_num = 2
        self.flooding = 0
        self.embed_name = 'bert.embeddings.word_embeddings.weight'  # 词
        self.restore_file = None
        self.gradient_accumulation_steps = 1
        # 模型预测路径
        self.checkpoint_path = "/home/wangzhili/lei/de_noising4torch/processed_data/Savemodel/runs_2/1611568898/model_0.9720_0.9720_0.9720_3500.bin"

        """
        实验记录
        """
