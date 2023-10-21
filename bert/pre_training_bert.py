# coding=utf-8

"""
Pre-training BERT模型程序
"""

import torch
from torch import nn
from tqdm import tqdm
import os
from bert.bert import Bert
from config import Config

config = Config()


class PreTrainingBert(nn.Module):
    """
    Pre-training BERT，包括Masked LM-遮蔽语言模型+NSP-下一句预测两个部分
    """

    def __init__(self, bert, vocab_size, hidden_size):
        """
        预训练bert模型
        :param bert_encoder: bert编码器，即bert模型的主体计算部分
        :param vocab_size: 词典词数
        :param hidden_size: 隐藏层大小
        """
        super(PreTrainingBert, self).__init__()
        self.bert = bert.to(config.device)

        self.mlm_linear = nn.Linear(hidden_size, vocab_size).to(config.device)  # 遮蔽语言模型的线性输出层

        self.nsp_linear = nn.Linear(hidden_size, 2).to(config.device)  # 下一句预测的线性输出层，2表示是IsNext或者NotNext

    def _masked_lm(self, seq):
        """

        :param seq:
        :return:
        """
        return self.mlm_linear(seq)

    def _next_sentence_prediction(self, seq):
        """

        :param seq:
        :return:
        """
        return self.nsp_linear(seq[:, 0])

    def pre_training(self, optimizer, criterion, data_loader):
        """
        预训练bert模型，包括mlm训练+nsp训练
        :param optimizer:
        :param criterion:
        :param data_loader:
        :return:
        """
        for epoch in range(config.epochs):
            for batch_data in data_loader:  # 迭代批次数据，训练模型
                optimizer.zero_grad()  # 优化器梯度清零

                data = {key: value.to(config.device) for key, value in batch_data.items()}  # 选择训练设备

                bert_out = self.bert(data['token'], data['segment'])  # bert-transformer encoder模块计算

                nsp_out = self.nsp_linear(bert_out[:, 0])  # nsp下一句任务

                mlm_out = self.mlm_linear(bert_out)  # msl遮蔽语言模型任务

                nsp_loss = criterion(nsp_out, data['nsp'])  # 计算nsp任务的损失

                mlm_loss = criterion(mlm_out.transpose(1, 2), data['mask'])  # 计算mlm任务的损失

                loss = nsp_loss + mlm_loss  # 预训练bert损失是两个任务损失的和

                loss.backward()  # 损失反向传播
                lr = optimizer.step_and_update_learning_rate()  # 优化器更新学习率

                # torch.save(self.bert.state_dict(),'model_state{}'.format(epoch))
                print('epoch:', epoch, '损失率:', loss.detach(), '学习率：', lr)
            #模型保存，只保存最后一次的训练结果
            '''保存整个模型'''
            if os.path.exists('model/bert_model.pth'):
                os.remove('model/bert_model.pth')
            torch.save(self.bert, 'model/bert_model.pth')
            '''保存模型参数'''
            if os.path.exists('model/bert_model_state.pkl'):
                os.remove('model/bert_model_state.pkl')
            torch.save(self.bert.state_dict(),'model/bert_model_state.pkl')
