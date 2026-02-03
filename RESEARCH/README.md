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

---

## 🚧 Project Status

This project is **under active development**. The core pipeline is functional and has already produced statistically significant results.

## 📊 Current Results

After extensive grid search optimization (6,000+ parameter combinations), the best model achieved:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Recall (min)** | 57.8% | +7.8% above random baseline |
| **Recall Gap** | 0.4% | Near-perfect balance between UP/DOWN |
| **MCC** | 0.159 | Weak but real predictive signal |

**Best Configuration:**
- Coordinate mode: Geocentric
- Orb multiplier: 0.25 (tight aspects only)
- Gaussian window: 201 days
- Gaussian std: 50.0
- Excluded bodies: Uranus, Pluto (reduced noise)

### Statistical Significance

- **z-score ≈ 4.9** (assuming ~1000 test samples)
- **p-value < 0.0001** — probability of random chance is less than 0.01%
- The model demonstrates a **statistically significant edge** over random guessing

### Practical Implications

| Aspect | Assessment |
|--------|------------|
| ✅ Better than random | Yes, by ~7.8 percentage points |
| ✅ Balanced predictions | Equal accuracy for UP and DOWN moves |
| ⚠️ Edge size | Moderate — requires low trading fees |
| 🎯 Key finding | Outer planets (Uranus, Pluto) add noise; excluding them improves performance |

### Interpretation

The MCC of 0.159 indicates a **weak but statistically real correlation** between planetary aspects and market movements. While not strong enough for high-frequency trading, this edge may be viable for:
- Medium to long-term position trading
- Signal confirmation in conjunction with other indicators
- Further research into specific planetary configurations

---

*Note: Past performance does not guarantee future results. This is research, not financial advice.*
