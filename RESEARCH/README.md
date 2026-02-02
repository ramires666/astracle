# RESEARCH — Astro Trading Research Pipeline

Модульный пайплайн для исследования корреляций между астрологическими данными и движениями рынка криптовалют.

## Быстрый старт

### 1. Создание/активация окружения Conda

```bash
# Если окружение btc уже есть:
conda activate btc

# Или создайте новое:
conda create -n btc python=3.12 -y
conda activate btc
```

### 2. Установка зависимостей

**Вариант A — Одной командой (рекомендуется):**

```bash
# Основные DS/ML пакеты + psycopg2 через conda
conda install -c conda-forge xgboost scikit-learn matplotlib seaborn tqdm pyarrow psycopg2 ipykernel joblib pandas numpy scipy -y

# Астро-движок (нет в conda, ставим через pip)
pip install pyswisseph
```

**Вариант B — Через pip (если conda не используется):**

```bash
pip install -r RESEARCH/requirements.txt
```

### 3. Запуск в VS Code (интерактивный режим)

1. Откройте `RESEARCH/main_pipeline.py`
2. Убедитесь, что выбран интерпретатор `btc` (`Ctrl+Shift+P` → `Python: Select Interpreter`)
3. Нажмите `Shift+Enter` на любой ячейке (маркер `# %%`) или кликните **Run Cell**

## Структура модулей

| Модуль | Описание |
|--------|----------|
| `config.py` | Конфигурация проекта (пути, настройки БД, субъекты) |
| `data_loader.py` | Загрузка рыночных данных из PostgreSQL |
| `labeling.py` | Создание сбалансированных меток UP/DOWN |
| `astro_engine.py` | Расчёт планетарных позиций и аспектов (Swiss Ephemeris) |
| `features.py` | Построение матрицы признаков |
| `model_training.py` | Обучение XGBoost, подбор порога |
| `visualization.py` | Графики: цена, распределения, confusion matrix |
| `grid_search.py` | Поиск по сетке параметров |
| `main_pipeline.py` | **Главный файл** — оркестрирует весь пайплайн |

## Проверка зависимостей

Запустите первую ячейку `main_pipeline.py` — она покажет недостающие пакеты:

```python
# %%
import importlib.util as iu
required = ["xgboost", "sklearn", "matplotlib", "seaborn", "tqdm", "pyarrow", "psycopg2", "swisseph"]
missing = [pkg for pkg in required if iu.find_spec(pkg) is None]
if missing:
    print("Missing:", ", ".join(missing))
else:
    print("✓ All dependencies found")
```
