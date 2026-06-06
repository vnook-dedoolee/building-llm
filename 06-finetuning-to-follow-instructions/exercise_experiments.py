# Copyright (c) Sebastian Raschka под лицензией Apache License 2.0 (см. LICENSE.txt).
# Источник: "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Код: https://github.com/rasbt/LLMs-from-scratch
#
# Код для запуска упражнений; подробнее см. exercise-solutions.ipynb

from functools import partial
from importlib.metadata import version
import json
import math
import os
import re
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import requests
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Импорт из локальных файлов в этой папке
from gpt_download import download_and_load_gpt2
from previous_chapters import (
    calc_loss_loader,
    generate,
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    train_model_simple,
    token_ids_to_text
)


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Предварительно токенизируем тексты
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class InstructionDatasetWithMasking(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Новое: отдельный список для длин инструкций
        self.instruction_lengths = []
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

            # Новое: собираем длины инструкций
            instruction_length = len(tokenizer.encode(instruction_plus_input))
            self.instruction_lengths.append(instruction_length)

    def __getitem__(self, index):
        # Новое: возвращаем длины инструкций и тексты по отдельности
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class InstructionDatasetPhi(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Предварительно токенизируем тексты
        self.encoded_texts = []
        for entry in data:

            ###################################################################
            # НОВОЕ: Используем `format_input_phi` и изменяем шаблон текста ответа
            instruction_plus_input = format_input_phi(entry)
            response_text = f"\n<|assistant|>:\n{entry['output']}"
            ###################################################################
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # аналогично стандартной инициализации весов
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Заменяем линейный слой на LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Рекурсивно применяем ту же функцию к дочерним модулям
            replace_linear_with_lora(module, rank, alpha)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Находим самую длинную последовательность в пакете
    batch_max_length = max(len(item)+1 for item in batch)

    # Дополняем и подготавливаем входы и цели
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Добавляем токен <|endoftext|>
        new_item += [pad_token_id]
        # Дополняем последовательности до max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Обрезаем последний токен для входов
        targets = torch.tensor(padded[1:])  # Сдвигаем на +1 вправо для целей

        # Новое: заменяем все, кроме первого, токены заполнения в целях на ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Новое: опционально обрезаем до максимальной длины последовательности
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Преобразуем списки входов и целей в тензоры и переносим на целевое устройство
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def custom_collate_with_masking_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Находим самую длинную последовательность в пакете
    batch_max_length = max(len(item)+1 for instruction_length, item in batch)   # Новое: пакет теперь является кортежем

    # Дополняем и подготавливаем входы и цели
    inputs_lst, targets_lst = [], []

    for instruction_length, item in batch:  # Новое: пакет теперь является кортежем
        new_item = item.copy()
        # Добавляем токен <|endoftext|>
        new_item += [pad_token_id]
        # Дополняем последовательности до max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Обрезаем последний токен для входов
        targets = torch.tensor(padded[1:])  # Сдвигаем на +1 вправо для целей

        # Заменяем все, кроме первого, токены заполнения в целях на ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Новое: маскируем все токены входа и инструкции в целях
        targets[:instruction_length-1] = -100

        # Опционально обрезаем до максимальной длины последовательности
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Преобразуем списки входов и целей в тензоры и переносим на целевое устройство
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def format_input_phi(entry):
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )

    input_text = f"\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, plot_name):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Строим график потерь на обучении и валидации по эпохам
    ax1.plot(epochs_seen, train_losses, label="Потери на обучении")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Потери на валидации")
    ax1.set_xlabel("Эпохи")
    ax1.set_ylabel("Потери")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # показываем только целые метки на оси x

    # Создаём вторую ось x для просмотренных токенов
    ax2 = ax1.twiny()  # Создаём вторую ось x, которая делит ту же ось y
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Невидимый график для выравнивания меток
    ax2.set_xlabel("Просмотрено токенов")

    fig.tight_layout()  # Настраиваем расположение, чтобы освободить место
    print(f"График сохранён как {plot_name}")
    plt.savefig(plot_name)
    # plt.show()


def main(mask_instructions=False, alpaca52k=False, phi3_prompt=False, lora=False):
    #######################################
    # Вывод версий пакетов
    #######################################
    print()
    pkgs = [
        "matplotlib",  # Библиотека для построения графиков
        "tiktoken",    # Токенизатор
        "torch",       # Библиотека для глубокого обучения
        "tqdm",        # Прогресс-бар
        "tensorflow",  # Для предобученных весов OpenAI
    ]
    for p in pkgs:
        print(f"{p} версия: {version(p)}")
    print(50*"-")

    #######################################
    # Скачивание и подготовка набора данных
    #######################################
    file_path = "instruction-data.json"

    if alpaca52k:
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    else:
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    train_portion = int(len(data) * 0.85)  # 85% на обучение
    test_portion = int(len(data) * 0.1)    # 10% на тестирование

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Длина обучающего набора:", len(train_data))
    print("Длина валидационного набора:", len(val_data))
    print("Длина тестового набора:", len(test_data))
    print(50*"-")

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    print(50*"-")

    if alpaca52k:
        allowed_max_length = 512
    else:
        allowed_max_length = 1024

    if mask_instructions and phi3_prompt:
        raise ValueError("Одновременная поддержка маскирования инструкций и шаблона промптов Phi-3 ещё не реализована.")

    if mask_instructions:
        customized_collate_fn = partial(custom_collate_with_masking_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetWithMasking
    elif phi3_prompt:
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetPhi
    else:
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDataset

    num_workers = 0

    if alpaca52k:
        batch_size = 4
    else:
        batch_size = 8

    torch.manual_seed(123)

    train_dataset = CustomDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = CustomDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    #######################################
    # Загрузка предобученной модели
    #######################################
    BASE_CONFIG = {
        "vocab_size": 50257,     # Размер словаря
        "context_length": 1024,  # Длина контекста
        "drop_rate": 0.0,        # Коэффициент дропаута
        "qkv_bias": True         # Смещение для query-key-value
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)

    print("Загружена модель:", CHOOSE_MODEL)
    print(50*"-")

    if lora:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Всего обучаемых параметров до: {total_params:,}")

        for param in model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Всего обучаемых параметров после: {total_params:,}")
        replace_linear_with_lora(model, rank=16, alpha=16)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Всего обучаемых параметров LoRA: {total_params:,}")
        model.to(device)

    #######################################
    # Тонкая настройка модели
    #######################################
    print("Начальные потери")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   Потери на обучении:", train_loss)
    print("   Потери на валидации:", val_loss)

    start_time = time.time()

    num_epochs = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    torch.manual_seed(123)

    start_context = format_input_phi(val_data[0]) if phi3_prompt else format_input(val_data[0])

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=start_context, tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Обучение завершено за {execution_time_minutes:.2f} минут.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

    plot_name = "loss-plot.pdf"
    if mask_instructions:
        plot_name = plot_name.replace(".pdf", "-mask-instructions.pdf")
    if alpaca52k:
        plot_name = plot_name.replace(".pdf", "-alpaca52k.pdf")
    if phi3_prompt:
        plot_name = plot_name.replace(".pdf", "-phi3-prompt.pdf")
    if lora:
        plot_name = plot_name.replace(".pdf", "-lora.pdf")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        plot_name = plot_name.replace(".pdf", "-baseline.pdf")

    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, plot_name)
    print(50*"-")

    #######################################
    # Сохранение результатов
    #######################################
    print("Генерация ответов")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input_phi(entry) if phi3_prompt else format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        if phi3_prompt:
            response_text = generated_text[len(input_text):].replace("<|assistant|>:", "").strip()
        else:
            response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    test_data_path = "instruction-data-with-response.json"
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"

    if mask_instructions:
        test_data_path = test_data_path.replace(".json", "-mask-instructions.json")
        file_name = file_name.replace(".pth", "-mask-instructions.pth")
    if alpaca52k:
        test_data_path = test_data_path.replace(".json", "-alpaca52k.json")
        file_name = file_name.replace(".pth", "-alpaca52k.pth")
    if phi3_prompt:
        test_data_path = test_data_path.replace(".json", "-phi3-prompt.json")
        file_name = file_name.replace(".pth", "-phi3-prompt.pth")
    if lora:
        test_data_path = test_data_path.replace(".json", "-lora.json")
        file_name = file_name.replace(".pth", "-lora.pth")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        test_data_path = test_data_path.replace(".json", "-baseline.json")
        file_name = file_name.replace(".pth", "-baseline.pth")

    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" для красивого форматирования
    print(f"Ответы сохранены как {test_data_path}")

    torch.save(model.state_dict(), file_name)
    print(f"Модель сохранена как {file_name}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Инструктивная тонкая настройка GPT-модели"
    )
    options = {"baseline", "mask_instructions", "alpaca_52k", "phi3_prompt", "lora"}
    parser.add_argument(
        "--exercise_solution",
        type=str,
        default="baseline",
        help=(
            f"Какой эксперимент запустить. Варианты: {options}."
        )
    )
    args = parser.parse_args()

    if args.exercise_solution == "baseline":
        main()
    elif args.exercise_solution == "mask_instructions":
        main(mask_instructions=True)
    elif args.exercise_solution == "alpaca_52k":
        main(alpaca52k=True)
    elif args.exercise_solution == "phi3_prompt":
        main(phi3_prompt=True)
    elif args.exercise_solution == "lora":
        main(lora=True)
    else:
        raise ValueError(f"{args.exercise_solution} не является допустимым значением --args.exercise_solution. Варианты: {options}")