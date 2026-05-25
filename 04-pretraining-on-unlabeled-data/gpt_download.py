# Copyright (c) Sebastian Raschka под лицензией Apache License 2.0 (см. LICENSE.txt).
# Источник: "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Код: https://github.com/rasbt/LLMs-from-scratch


import os

import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # Проверка размера модели
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Размер модели должен быть одним из: {allowed_sizes}")

    # Определение путей
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Загрузка файлов
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Загрузка настроек и параметров
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))

        # Проверка, существует ли файл и совпадает ли его размер
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"Файл уже существует и актуален: {destination}")
                return True

        block_size = 1024  # 1 КБ
        desc = os.path.basename(download_url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        if backup_url is not None:
            print(f"Основной URL ({url}) недоступен. Попытка загрузки с резервного URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                pass

        error_message = (
            f"Не удалось загрузить ни с основного URL ({url}),"
            f"{' ни с резервного URL (' + backup_url + ')' if backup_url else ''}."
            "\nПроверьте подключение к интернету или доступность файла.\n"
            "За помощью обращайтесь: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")


# Альтернативный способ с использованием `requests`
"""
def download_file(url, destination):
    # Отправляем GET-запрос для скачивания файла в потоковом режиме
    response = requests.get(url, stream=True)

    # Получаем общий размер файла из заголовков, по умолчанию 0, если не указан
    file_size = int(response.headers.get("content-length", 0))

    # Проверяем, существует ли файл и совпадает ли его размер
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"Файл уже существует и актуален: {destination}")
            return

    # Определяем размер блока для чтения файла
    block_size = 1024  # 1 Килобайт

    # Инициализируем прогресс-бар с общим размером файла
    progress_bar_description = url.split("/")[-1]  # Извлекаем имя файла из URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Открываем целевой файл в режиме бинарной записи
        with open(destination, "wb") as file:
            # Итерируемся по данным файла чанками
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Обновляем прогресс-бар
                file.write(chunk)  # Записываем чанк в файл
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Инициализация словаря параметров с пустыми блоками для каждого слоя
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Итерация по каждой переменной в контрольной точке
    for name, _ in tf.train.list_variables(ckpt_path):
        # Загружаем переменную и убираем сингулярные измерения
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Обрабатываем имя переменной для извлечения нужных частей
        variable_name_parts = name.split("/")[1:]  # Пропускаем префикс 'model/'

        # Определяем целевой словарь для переменной
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Рекурсивно получаем или создаём вложенные словари
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Присваиваем массив переменной последнему ключу
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params