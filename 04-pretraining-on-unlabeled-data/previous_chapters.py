# Авторские права (c) Sebastian Raschka под лицензией Apache License 2.0 (см. LICENSE.txt).
# Источник для книги "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Код: https://github.com/rasbt/LLMs-from-scratch
#
# Этот файл собирает весь соответствующий код, который мы рассмотрели
# в главах 2-4.
# Этот файл можно запускать как самостоятельный скрипт.

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Глава 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Токенизируем весь текст
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Используем скользящее окно для разбиения книги на перекрывающиеся последовательности длиной max_length
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

    # Создаем набор данных
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Создаем загрузчик данных
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Глава 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out должен делиться на n_heads без остатка"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Уменьшаем размерность проекции, чтобы соответствовать желаемой выходной размерности

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Линейный слой для объединения выходов голов
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Форма: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Неявно разделяем матрицу, добавляя измерение `num_heads`
        # Разворачиваем последнее измерение: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Транспонируем: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Вычисляем масштабированное скалярное произведение внимания (self-attention) с каузальной маской
        attn_scores = queries @ keys.transpose(2, 3)  # Скалярное произведение для каждой головы

        # Исходная маска, обрезанная до количества токенов и преобразованная в булеву
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Используем маску для заполнения оценок внимания
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Форма: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Объединяем головы, где self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # опциональная проекция

        return context_vec


#####################################
# Глава 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Сокращенное соединение (shortcut) для блока внимания
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Форма [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Добавляем исходный вход обратно

        # Сокращенное соединение (shortcut) для блока прямого распространения
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Добавляем исходный вход обратно

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Форма [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx — это массив (B, T) индексов в текущем контексте
    for _ in range(max_new_tokens):

        # Обрезаем текущий контекст, если он превышает поддерживаемый размер контекста
        # Например, если LLM поддерживает только 5 токенов, а размер контекста равен 10,
        # то только последние 5 токенов используются в качестве контекста
        idx_cond = idx[:, -context_size:]

        # Получаем предсказания
        with torch.no_grad():
            logits = model(idx_cond)

        # Сосредотачиваемся только на последнем временном шаге
        # (batch, n_token, vocab_size) становится (batch, vocab_size)
        logits = logits[:, -1, :]

        # Получаем индекс записи в словаре с наибольшим значением logits
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Добавляем выбранный индекс к текущей последовательности
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Размер словаря
        "context_length": 1024,  # Длина контекста
        "emb_dim": 768,          # Размерность эмбеддинга
        "n_heads": 12,           # Количество голов внимания
        "n_layers": 12,          # Количество слоев
        "drop_rate": 0.1,        # Коэффициент дропаута
        "qkv_bias": False        # Смещение Query-Key-Value
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # отключаем дропаут

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}ВХОД\n{50*'='}")
    print("\nВходной текст:", start_context)
    print("Закодированный входной текст:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}ВЫХОД\n{50*'='}")
    print("\nВыход:", out)
    print("Длина выхода:", len(out[0]))
    print("Выходной текст:", decoded_text)