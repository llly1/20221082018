#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import argparse
import os
from difflib import SequenceMatcher
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import jieba
from collections import Counter
import pickle

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Vocabulary:
    def __init__(self):
        self.word2idx = {'PAD': 0, 'UNK': 1, 'BOS': 2, 'EOS': 3}
        self.idx2word = {0: 'PAD', 1: 'UNK', 2: 'BOS', 3: 'EOS'}
        self.vocab_size = 4
    
    def build_vocab(self, texts, min_freq=2):
        word_counts = Counter()
        for text in texts:
            words = list(jieba.cut(text))
            word_counts.update(words)
        
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text, max_len=128):
        words = list(jieba.cut(text))
        encoded = [self.word2idx.get(word, 1) for word in words]
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        else:
            encoded = encoded + [0] * (max_len - len(encoded))
        return encoded
    
    def decode(self, indices):
        words = [self.idx2word.get(idx, 'UNK') for idx in indices if idx != 0]
        return ''.join(words)

class CorrectedHateModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_groups=10):
        super(CorrectedHateModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        
        # 分类器
        self.group_classifier = nn.Linear(hidden_dim * 2, num_groups)
        self.hate_classifier = nn.Linear(hidden_dim * 2, 2)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x, mask=None):
        embedded = self.embedding(x)
        lstm_out, (h, c) = self.lstm(embedded)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(lstm_out)
            masked_out = lstm_out * mask_expanded
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            sent_repr = masked_out.sum(dim=1) / lengths
        else:
            sent_repr = lstm_out.mean(dim=1)
        
        sent_repr = self.dropout(sent_repr)
        
        group_logits = self.group_classifier(sent_repr)
        hate_logits = self.hate_classifier(sent_repr)
        
        return {
            'group_logits': group_logits,
            'hate_logits': hate_logits,
            'sentence_repr': sent_repr
        }

class CorrectedHateDataset(Dataset):
    def __init__(self, data, vocab, max_len=128, mode='train'):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        self.mode = mode
        
        # 构建标签映射 - 修正为单一类别
        self.group_labels = {}
        self.hate_labels = {'non-hate': 0, 'hate': 1}
        
        if mode == 'train':
            self._build_label_maps()
    
    def _normalize_group(self, group):
        """将群体名称标准化为单一主类别"""
        group = group.strip()
        
        # 移除", others"后缀
        if ', others' in group:
            group = group.replace(', others', '')
        
        # 处理组合类别，取第一个主要类别
        if ',' in group:
            group = group.split(',')[0].strip()
        
        # 标准化类别名称
        group_mapping = {
            'Racism': 'Racism',
            'Sexism': 'Sexism', 
            'Region': 'Region',
            'LGBTQ': 'LGBTQ',
            'Disability': 'Disability',
            'Religion': 'Religion',
            'Age': 'Age',
            'Politics': 'Politics',
            'Nationality': 'Nationality',
            'Appearance': 'Appearance',
            'Sexis': 'Sexism',  # 修正拼写错误
            'others': 'others'
        }
        
        return group_mapping.get(group, 'others')
    
    def _build_label_maps(self):
        groups = set()
        for item in self.data:
            if 'output' in item and item['output']:
                tuples = self._parse_output(item['output'])
                for t in tuples:
                    if len(t) >= 3:
                        normalized_group = self._normalize_group(t[2])
                        groups.add(normalized_group)
        
        for i, group in enumerate(sorted(groups)):
            self.group_labels[group] = i
    
    def _parse_output(self, output):
        tuples = []
        output = output.replace('[END]', '').strip()
        
        if '[SEP]' in output:
            parts = output.split('[SEP]')
        else:
            parts = [output]
        
        for part in parts:
            part = part.strip()
            if ' | ' in part:
                elements = [elem.strip() for elem in part.split(' | ')]
                if len(elements) == 4:
                    tuples.append(tuple(elements))
            elif '|' in part:
                elements = [elem.strip() for elem in part.split('|')]
                if len(elements) == 4:
                    tuples.append(tuple(elements))
        
        return tuples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_ids = self.vocab.encode(item['content'], self.max_len)
        mask = [1 if id != 0 else 0 for id in input_ids]
        
        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float),
            'text': item['content'],
            'id': item['id']
        }
        
        if self.mode == 'train' and 'output' in item:
            tuples = self._parse_output(item['output'])
            
            if tuples:
                first_tuple = tuples[0]
                normalized_group = self._normalize_group(first_tuple[2])
                
                group_label = self.group_labels.get(normalized_group, 0)
                hate_label = self.hate_labels.get(first_tuple[3], 0)
            else:
                group_label = 0
                hate_label = 0
            
            result['group_label'] = torch.tensor(group_label, dtype=torch.long)
            result['hate_label'] = torch.tensor(hate_label, dtype=torch.long)
            result['target_text'] = item['output']
        
        return result

class CorrectedTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.vocab = Vocabulary()
        self.model = None
        self.num_groups = 0
        self.group_id2name = {}
        self.dataset = None
    
    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def prepare_data(self, train_file, val_split=0.1):
        print("Loading and preparing data...")
        all_data = self.load_data(train_file)
        
        texts = [item['content'] for item in all_data]
        self.vocab.build_vocab(texts)
        print(f"Vocabulary size: {self.vocab.vocab_size}")
        
        # 创建临时dataset来获取标准化的群体类别
        temp_dataset = CorrectedHateDataset(all_data, self.vocab, mode='train')
        
        # 使用dataset中标准化后的群体标签
        self.group_id2name = {v: k for k, v in temp_dataset.group_labels.items()}
        self.num_groups = len(self.group_id2name)
        
        print(f"Number of target groups: {self.num_groups}")
        print(f"Target groups: {list(self.group_id2name.values())}")
        
        train_data, val_data = train_test_split(all_data, test_size=val_split, random_state=42)
        return train_data, val_data
    
    def create_model(self):
        self.model = CorrectedHateModel(
            vocab_size=self.vocab.vocab_size,
            embed_dim=128,
            hidden_dim=64,
            num_groups=self.num_groups
        )
        self.model.to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def extract_target_argument(self, text):
        """改进的Target和Argument提取"""
        words = list(jieba.cut(text))
        
        # 策略1: 如果文本较短，直接分割
        if len(words) <= 2:
            if len(words) == 1:
                return words[0], words[0]
            else:
                return words[0], words[1]
        
        # 策略2: 寻找关键词来分割
        split_words = ['是', '很', '太', '都', '就', '在', '的', '了', '要', '会', '有', '说', '做', '去', '来', '到', '把', '被', '让', '使']
        
        for i, word in enumerate(words):
            if word in split_words and i > 0 and i < len(words) - 1:
                target = ''.join(words[:i])
                argument = ''.join(words[i:])
                return target, argument
        
        # 策略3: 默认分割 - 前半部分作为target，后半部分作为argument
        mid = len(words) // 2
        if mid == 0:
            mid = 1
        
        target = ''.join(words[:mid])
        argument = ''.join(words[mid:])
        
        return target, argument
    
    def predict_quadruples(self, text):
        """预测四元组 - 修正格式"""
        self.model.eval()
        
        input_ids = torch.tensor([self.vocab.encode(text)], dtype=torch.long).to(self.device)
        mask = torch.tensor([[1 if id != 0 else 0 for id in input_ids[0]]], dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, mask)
            
            hate_pred = torch.argmax(outputs['hate_logits'], dim=1).item()
            group_pred = torch.argmax(outputs['group_logits'], dim=1).item()
            
            # 使用标准格式 - 单一类别
            hate_label = 'hate' if hate_pred == 1 else 'non-hate'
            group_name = self.group_id2name.get(group_pred, 'others')
            
            # 改进的Target和Argument提取
            target, argument = self.extract_target_argument(text)
            
            # 使用标准格式：空格分隔的 | 和单一群体类别
            quadruple = f"{target} | {argument} | {group_name} | {hate_label}"
            return [quadruple]
    
    def train(self, train_file='train.json', epochs=8, batch_size=64, lr=0.0005):
        print("=" * 60)
        print("修正版仇恨言论检测模型训练")
        print("=" * 60)
        
        train_data, val_data = self.prepare_data(train_file)
        
        train_dataset = CorrectedHateDataset(train_data, self.vocab, mode='train')
        val_dataset = CorrectedHateDataset(val_data, self.vocab, mode='train')
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")
        
        self.create_model()
        
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['mask'].to(self.device)
                group_labels = batch['group_label'].to(self.device)
                hate_labels = batch['hate_label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, mask)
                
                group_loss = F.cross_entropy(outputs['group_logits'], group_labels, reduction='mean')
                hate_loss = F.cross_entropy(outputs['hate_logits'], hate_labels, reduction='mean')
                
                # 检查NaN
                if torch.isnan(group_loss) or torch.isinf(group_loss) or torch.isnan(hate_loss) or torch.isinf(hate_loss):
                    continue
                
                loss = hate_loss + 0.5 * group_loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_train_loss = total_loss / max(batch_count, 1)
            train_losses.append(avg_train_loss)
            
            val_acc = self.evaluate(val_loader)
            val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if epoch == 0 or val_acc == max(val_accuracies):
                self.save_model('best_corrected_model')
        
        self.plot_curves(train_losses, val_accuracies)
        print("训练完成！")
        
        self.predict_test_files()
    
    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['mask'].to(self.device)
                hate_labels = batch['hate_label'].to(self.device)
                
                outputs = self.model(input_ids, mask)
                
                _, predicted = torch.max(outputs['hate_logits'], 1)
                total += hate_labels.size(0)
                correct += (predicted == hate_labels).sum().item()
        
        return correct / total
    
    def predict_test_files(self):
        print("\n开始预测测试文件...")
        
        for test_file in ['test1.json', 'test2.json']:
            if os.path.exists(test_file):
                output_file = f"corrected_submit_{test_file.replace('.json', '.txt')}"
                self.predict_file(test_file, output_file)
    
    def predict_file(self, input_file, output_file):
        print(f"预测 {input_file} -> {output_file}")
        
        test_data = self.load_data(input_file)
        predictions = []
        
        for item in tqdm(test_data, desc="Predicting"):
            quadruples = self.predict_quadruples(item['content'])
            pred = quadruples[0] + ' [END]'
            predictions.append(pred)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(pred + '\n')
        
        print(f"预测完成！保存到 {output_file}")
        
        print("\n前5个预测结果:")
        for i, pred in enumerate(predictions[:5], 1):
            print(f"{i}: {pred}")
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pth'))
        with open(os.path.join(path, 'vocab.pkl'), 'wb') as f:
            pickle.dump(self.vocab, f)
        with open(os.path.join(path, 'group_mapping.pkl'), 'wb') as f:
            pickle.dump(self.group_id2name, f)
    
    def plot_curves(self, train_losses, val_accuracies):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('corrected_training.png', dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='修正版仇恨言论检测')
    parser.add_argument('--epochs', type=int, default=8, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    
    args = parser.parse_args()
    
    set_seed(42)
    
    trainer = CorrectedTrainer()
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

if __name__ == "__main__":
    main() 