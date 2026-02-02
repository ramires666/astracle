"""
Numba-оптимизированные утилиты для астрологических расчётов.

Этот модуль содержит JIT-компилированные функции для ускорения
вычислительно-ёмких операций:
- Нормализация углов
- Расчёт разниц долготы
- Пакетная обработка аспектов
"""

import numpy as np

# Попытка импортировать Numba, fallback на обычные функции если не установлен
try:
    from numba import jit, prange, float64, int64, boolean
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Создаём заглушку для декоратора @jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ═══════════════════════════════════════════════════════════════════════════════
# БАЗОВЫЕ ФУНКЦИИ РАБОТЫ С УГЛАМИ (JIT-компилированные)
# ═══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True)
def diff_0_180(lon1: float, lon2: float) -> float:
    """
    Разница долгот, нормализованная к [0..180].
    
    Это расстояние между двумя точками на круге (360°),
    минимальное из двух возможных направлений.
    
    Примеры:
        diff_0_180(10, 20) = 10
        diff_0_180(350, 10) = 20  (через 0°)
        diff_0_180(0, 180) = 180
    """
    diff = abs(lon1 - lon2) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


@jit(nopython=True, cache=True)
def normalize_angle(angle: float) -> float:
    """
    Нормализация угла к диапазону [0..360).
    """
    return angle % 360.0


# ═══════════════════════════════════════════════════════════════════════════════
# ПАКЕТНЫЕ ОПЕРАЦИИ (для работы с массивами)
# ═══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, parallel=True, cache=True)
def compute_pairwise_angles(longitudes: np.ndarray) -> np.ndarray:
    """
    Вычисление всех попарных углов между телами.
    
    Args:
        longitudes: 1D массив долгот тел [N]
        
    Returns:
        2D массив углов [N, N] (верхний треугольник)
        angles[i, j] = угол между телом i и телом j (i < j)
    """
    n = len(longitudes)
    angles = np.zeros((n, n), dtype=np.float64)
    
    for i in prange(n):
        for j in range(i + 1, n):
            angles[i, j] = diff_0_180(longitudes[i], longitudes[j])
    
    return angles


@jit(nopython=True, cache=True)
def filter_aspects_by_orb(
    angles: np.ndarray,
    aspect_degrees: np.ndarray,
    orb_limits: np.ndarray,
) -> tuple:
    """
    Фильтрация углов по орбисам для каждого типа аспекта.
    
    Args:
        angles: 2D массив попарных углов [N, N]
        aspect_degrees: 1D массив градусов аспектов (0, 60, 90, 120, 180)
        orb_limits: 1D массив орбисов для каждого аспекта
        
    Returns:
        Tuple of (i_indices, j_indices, aspect_indices, orb_values)
        Каждый — 1D массив с индексами найденных аспектов
    """
    n = angles.shape[0]
    n_aspects = len(aspect_degrees)
    
    # Предварительный подсчёт для аллокации
    max_hits = n * n * n_aspects
    
    i_idx = np.zeros(max_hits, dtype=np.int64)
    j_idx = np.zeros(max_hits, dtype=np.int64)
    asp_idx = np.zeros(max_hits, dtype=np.int64)
    orb_vals = np.zeros(max_hits, dtype=np.float64)
    
    count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            angle = angles[i, j]
            
            for a in range(n_aspects):
                orb = abs(angle - aspect_degrees[a])
                if orb <= orb_limits[a]:
                    i_idx[count] = i
                    j_idx[count] = j
                    asp_idx[count] = a
                    orb_vals[count] = orb
                    count += 1
                    break  # Один аспект на пару
    
    return i_idx[:count], j_idx[:count], asp_idx[:count], orb_vals[:count]


@jit(nopython=True, cache=True)
def is_aspect_applying(
    lon1: float, lon2: float,
    speed1: float, speed2: float,
    aspect_degree: float,
    dt_days: float = 1.0
) -> bool:
    """
    Определяет, сближается ли аспект (applying) или расходится (separating).
    
    Сравниваем текущий орбис с орбисом через dt_days дней.
    Если орбис уменьшится — аспект сближается.
    """
    diff_now = diff_0_180(lon1, lon2)
    orb_now = abs(diff_now - aspect_degree)
    
    lon1_next = (lon1 + speed1 * dt_days) % 360.0
    lon2_next = (lon2 + speed2 * dt_days) % 360.0
    diff_next = diff_0_180(lon1_next, lon2_next)
    orb_next = abs(diff_next - aspect_degree)
    
    return orb_next < orb_now


# ═══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════════════════════

def check_numba_available() -> bool:
    """Проверка доступности Numba."""
    return NUMBA_AVAILABLE


def warmup_jit():
    """
    Прогрев JIT-компилятора.
    
    Первый вызов JIT-функции компилирует её, что занимает время.
    Вызовите эту функцию при инициализации, чтобы компиляция
    произошла заранее, а не во время критичных расчётов.
    """
    if not NUMBA_AVAILABLE:
        return
        
    # Прогрев базовых функций
    _ = diff_0_180(10.0, 20.0)
    _ = normalize_angle(370.0)
    
    # Прогрев пакетных функций
    test_lons = np.array([0.0, 90.0, 180.0], dtype=np.float64)
    _ = compute_pairwise_angles(test_lons)
    
    test_angles = np.array([[0, 90, 180], [0, 0, 90], [0, 0, 0]], dtype=np.float64)
    test_aspects = np.array([0.0, 60.0, 90.0, 120.0, 180.0], dtype=np.float64)
    test_orbs = np.array([10.0, 6.0, 8.0, 8.0, 10.0], dtype=np.float64)
    _ = filter_aspects_by_orb(test_angles, test_aspects, test_orbs)
    
    _ = is_aspect_applying(10.0, 20.0, 1.0, 0.5, 0.0)
