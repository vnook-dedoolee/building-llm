# Авторские права (c) Себастьян Рашка, под лицензией Apache 2.0 (см. LICENSE.txt)
# Исходный код для книги "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Репозиторий с кодом: https://github.com/rasbt/LLMs-from-scratch

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Токенизируем весь текст
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Разбиваем книгу на перекрывающиеся последовательности длиной max_length,
        # используя скользящее окно
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Инициализируем токенизатор
    tokenizer = tiktoken.get_encoding("gpt2")

    # Создаем набор данных (датасет)
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Создаем загрузчик данных
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out должен делиться на num_heads без остатка"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Уменьшаем размерность проекции до целевой выходной размерности

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Линейный слой для объединения выходов всех голов
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Тензор формы: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Неявно разделяем матрицу, добавляя размерность `num_heads`
        # Разворачиваем последнее измерение: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Транспонируем: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Вычисляем scaled dot-product attention (то есть self-attention) с каузальной маской
        attn_scores = queries @ keys.transpose(2, 3)  # Скалярное произведение для каждой головы

        # Исходную маску обрезаем до количества токенов и преобразуем в булевый тип
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Заполняем запрещённые позиции в матрице внимания бесконечно малыми значениями
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Тензор формы: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Объединяем головы, при этом self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # опциональная проекция

        return context_vec