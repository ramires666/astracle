"""
Astro engine module for RESEARCH pipeline.
Calculates planetary positions, aspects, and transits on-the-fly.
No caching to DB - calculated fast each time.

TODO: Save best grid search results (orb, birth dates, etc.)
TODO: Add moon phases and other planet phases
TODO: Add houses when doing birth date grid search
"""
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timezone
from typing import List, Optional
from tqdm import tqdm

# Import from existing src modules
from src.astro.engine.settings import AstroSettings
from src.astro.engine.calculator import set_ephe_path, calculate_daily_bodies, calculate_bodies
from src.astro.engine.aspects import calculate_aspects, calculate_transit_aspects
from src.astro.engine.models import AspectConfig, BodyPosition

from .config import cfg, resolve_path


def init_ephemeris() -> AstroSettings:
    """
    Initialize Swiss Ephemeris and return AstroSettings.
    
    Returns:
        AstroSettings with bodies and aspects configurations
    """
    astro_cfg = cfg.get_astro_config()
    
    # Set ephemeris path
    set_ephe_path(str(astro_cfg["ephe_path"]))
    
    # Create settings
    settings = AstroSettings(
        bodies_path=astro_cfg["bodies_path"],
        aspects_path=astro_cfg["aspects_path"],
    )
    
    return settings


def parse_birth_dt_utc(value: str) -> datetime:
    """Parse birth datetime string to UTC datetime."""
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_time_utc(value: str) -> time:
    """Parse time string (HH:MM:SS) to time object."""
    return datetime.strptime(value, "%H:%M:%S").time()


def scale_aspects(aspects: List[AspectConfig], orb_mult: float) -> List[AspectConfig]:
    """Scale aspect orbs by multiplier."""
    return [
        AspectConfig(name=a.name, degree=a.degree, orb=float(a.orb) * orb_mult)
        for a in aspects
    ]


def calculate_bodies_for_dates(
    dates: pd.Series,
    settings: AstroSettings,
    time_utc: Optional[time] = None,
    center: str = "geo",
    progress: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Рассчитать позиции планет для диапазона дат.
    
    ═══════════════════════════════════════════════════════════════════════════════
    ЧТО ЭТА ФУНКЦИЯ ДЕЛАЕТ:
    ═══════════════════════════════════════════════════════════════════════════════
    
    Для каждой даты в списке рассчитывает положение всех планет (Солнце, Луна,
    Меркурий, Венера, Марс, Юпитер, Сатурн, Уран, Нептун, Плутон и т.д.)
    
    КООРДИНАТНЫЕ СИСТЕМЫ (параметр center):
    ─────────────────────────────────────────────────────────────────────────────
    • "geo" (геоцентрическая) - Земля в центре
      Это классическая астрология. Мы смотрим на небо с Земли.
      Солнце "ходит" по знакам зодиака.
      
    • "helio" (гелиоцентрическая) - Солнце в центре
      Это "научная" точка зрения. Солнце неподвижно, планеты вращаются.
      В этой системе Земли НЕТ как объекта (мы сами на ней).
      Вместо этого есть барицентр Земля-Луна (общий центр масс).
    
    ВОЗВРАЩАЕТ:
    ─────────────────────────────────────────────────────────────────────────────
    1. df_bodies - DataFrame со всеми позициями планет:
       • date - дата
       • body - название планеты (Sun, Moon, Mars и т.д.)
       • lon - долгота (0-360°, положение в зодиаке)
       • lat - широта (обычно близко к 0°, эклиптика)
       • speed - скорость движения (°/день)
       • is_retro - ретроградность (True = планета "идёт назад")
       • sign - знак зодиака (Aries, Taurus, Gemini...)
       • declination - склонение (широта относительно экватора)
       
    2. bodies_by_date - словарь {дата: [список BodyPosition]}
       Используется для расчёта аспектов.
    ═══════════════════════════════════════════════════════════════════════════════
    
    Args:
        dates: Series of dates to calculate for
        settings: AstroSettings with body configurations
        time_utc: Time of day for calculations (default from config)
        center: Coordinate center ('geo' or 'helio')
        progress: Show progress bar
    
    Returns:
        Tuple of (df_bodies DataFrame, bodies_by_date dict for aspect calculations)
    """
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 1: Получаем время расчёта из конфига (по умолчанию 00:00:00 UTC)
    # ─────────────────────────────────────────────────────────────────────────────
    astro_cfg = cfg.get_astro_config()
    time_utc = time_utc or parse_time_utc(astro_cfg["daily_time_utc"])
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 2: Подготавливаем контейнеры для результатов
    # ─────────────────────────────────────────────────────────────────────────────
    bodies_rows = []        # Список строк для DataFrame
    bodies_by_date = {}     # Словарь для аспектов {дата -> [BodyPosition, ...]}
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 3: Преобразуем даты в формат date (без времени)
    # ─────────────────────────────────────────────────────────────────────────────
    date_list = pd.to_datetime(dates).dt.date
    iterator = tqdm(date_list, desc="Calculating bodies") if progress else date_list
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 4: Для каждой даты вызываем Swiss Ephemeris и получаем позиции
    # ─────────────────────────────────────────────────────────────────────────────
    for d in iterator:
        # calculate_daily_bodies - вызывает швейцарские эфемериды
        # и возвращает список BodyPosition для каждой планеты
        bodies = calculate_daily_bodies(d, time_utc, settings.bodies, center=center)
        bodies_by_date[d] = bodies  # Сохраняем для аспектов
        
        # ─────────────────────────────────────────────────────────────────────────
        # ШАГ 5: Преобразуем каждую позицию планеты в строку словаря
        # ─────────────────────────────────────────────────────────────────────────
        for b in bodies:
            bodies_rows.append({
                "date": b.date,           # Дата расчёта
                "body": b.body,           # Имя планеты (Sun, Moon, etc.)
                "lon": b.lon,             # Долгота 0-360° (положение в зодиаке)
                "lat": b.lat,             # Широта (обычно около 0°)
                "speed": b.speed,         # Скорость °/день (отрицательная = ретро)
                "is_retro": b.is_retro,   # True если ретроградная
                "sign": b.sign,           # Знак зодиака
                "declination": b.declination,  # Склонение от экватора
            })
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 6: Собираем всё в DataFrame
    # ─────────────────────────────────────────────────────────────────────────────
    df_bodies = pd.DataFrame(bodies_rows)
    return df_bodies, bodies_by_date


def calculate_bodies_for_dates_multi(
    dates: pd.Series,
    settings: AstroSettings,
    coord_mode: str = "geo",
    time_utc: Optional[time] = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    РАСЧЁТ ПЛАНЕТ В РАЗНЫХ КООРДИНАТНЫХ СИСТЕМАХ
    ═══════════════════════════════════════════════════════════════════════════════
    
    Эта функция - "обёртка" над calculate_bodies_for_dates, которая позволяет
    выбрать режим расчёта:
    
    РЕЖИМЫ (coord_mode):
    ─────────────────────────────────────────────────────────────────────────────
    
    • "geo" - ТОЛЬКО геоцентрические координаты (Земля в центре)
      Это классическая астрология. 99% астрологов используют это.
      Солнце "ходит" по зодиаку, Земля неподвижна.
      
    • "helio" - ТОЛЬКО гелиоцентрические координаты (Солнце в центре)
      Это научная/астрономическая точка зрения.
      Солнце неподвижно, все планеты вращаются вокруг него.
      ВАЖНО: В этом режиме НЕТ Солнца и Луны как объектов!
      
    • "both" - ОБЕ системы одновременно
      Считаем и geo и helio, объединяем в один большой набор.
      Geo-фичи остаются как есть: Sun_lon, Moon_lon, etc.
      Helio-фичи получают префикс: helio_Earth_lon, helio_Mars_lon, etc.
      Это УДВАИВАЕТ количество признаков, но даёт модели максимум информации!
    
    ПОЧЕМУ ЭТО ВАЖНО:
    ─────────────────────────────────────────────────────────────────────────────
    Разные трейдеры используют разные системы. Некоторые верят что гелио
    лучше показывает "космическую геометрию", другие предпочитают классику.
    Режим "both" позволяет модели самой решить, какие признаки важнее!
    
    ВОЗВРАЩАЕТ:
    ─────────────────────────────────────────────────────────────────────────────
    1. df_bodies - объединённый DataFrame с позициями (с префиксами для helio)
    2. geo_bodies_by_date - словарь geo позиций для аспектов
    3. helio_bodies_by_date - словарь helio позиций для аспектов (или пустой)
    ═══════════════════════════════════════════════════════════════════════════════
    
    Args:
        dates: Series of dates
        settings: AstroSettings
        coord_mode: 'geo', 'helio', or 'both'
        time_utc: Time of day (optional)
        progress: Show progress bar
        
    Returns:
        Tuple of (df_bodies, geo_bodies_by_date, helio_bodies_by_date)
    """
    # ─────────────────────────────────────────────────────────────────────────────
    # Проверяем корректность режима
    # ─────────────────────────────────────────────────────────────────────────────
    valid_modes = ["geo", "helio", "both"]
    if coord_mode not in valid_modes:
        raise ValueError(f"coord_mode must be one of {valid_modes}, got '{coord_mode}'")
    
    geo_bodies_by_date = {}
    helio_bodies_by_date = {}
    all_dfs = []
    
    # ─────────────────────────────────────────────────────────────────────────────
    # РЕЖИМ "geo" или "both": считаем геоцентрические координаты
    # ─────────────────────────────────────────────────────────────────────────────
    if coord_mode in ["geo", "both"]:
        if progress:
            print("📍 Расчёт ГЕОЦЕНТРИЧЕСКИХ координат (Земля в центре)...")
        df_geo, geo_bodies_by_date = calculate_bodies_for_dates(
            dates, settings, time_utc=time_utc, center="geo", progress=progress
        )
        # Для режима "both" добавляем префикс "geo_" к именам тел
        if coord_mode == "both":
            df_geo["body"] = "geo_" + df_geo["body"].astype(str)
        all_dfs.append(df_geo)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # РЕЖИМ "helio" или "both": считаем гелиоцентрические координаты
    # ─────────────────────────────────────────────────────────────────────────────
    if coord_mode in ["helio", "both"]:
        if progress:
            print("☀️ Расчёт ГЕЛИОЦЕНТРИЧЕСКИХ координат (Солнце в центре)...")
        df_helio, helio_bodies_by_date = calculate_bodies_for_dates(
            dates, settings, time_utc=time_utc, center="helio", progress=progress
        )
        # Добавляем префикс "helio_" к именам тел
        df_helio["body"] = "helio_" + df_helio["body"].astype(str)
        all_dfs.append(df_helio)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Объединяем все DataFrame в один
    # ─────────────────────────────────────────────────────────────────────────────
    if len(all_dfs) == 1:
        df_bodies = all_dfs[0]
    else:
        df_bodies = pd.concat(all_dfs, ignore_index=True)
        if progress:
            print(f"✅ Объединено: {len(df_bodies)} записей из {len(all_dfs)} систем координат")
    
    return df_bodies, geo_bodies_by_date, helio_bodies_by_date


def calculate_aspects_for_dates(
    bodies_by_date: dict,
    settings: AstroSettings,
    orb_mult: float = 1.0,
    progress: bool = True,
    prefix: str = "",
) -> pd.DataFrame:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    РАСЧЁТ АСПЕКТОВ МЕЖДУ ПЛАНЕТАМИ
    ═══════════════════════════════════════════════════════════════════════════════
    
    ЧТО ТАКОЕ АСПЕКТ:
    ─────────────────────────────────────────────────────────────────────────────
    Аспект — это угловое расстояние между двумя планетами на небе.
    Некоторые углы считаются "особыми" и несут определённое значение:
    
    • Conjunction (0°)   — соединение, планеты рядом
    • Sextile (60°)      — гармоничный
    • Square (90°)       — напряжённый
    • Trine (120°)       — гармоничный
    • Opposition (180°)  — напряжённый
    
    ORB (ОРБИС):
    ─────────────────────────────────────────────────────────────────────────────
    Допустимое отклонение от точного угла. Например, если соединение имеет
    орбис 8°, то планеты считаются в соединении при угле 0°±8° = от 352° до 8°.
    
    orb_mult - множитель орбиса:
    • 0.5 — очень узкие орбисы (только точные аспекты)
    • 1.0 — стандартные орбисы
    • 1.5 — широкие орбисы (больше аспектов найдётся)
    
    ВОЗВРАЩАЕТ:
    ─────────────────────────────────────────────────────────────────────────────
    DataFrame с колонками:
    • date - дата
    • p1, p2 - названия планет в аспекте
    • aspect - тип аспекта (Conjunction, Trine, etc.)
    • orb - точный орбис (насколько отклонение от идеального угла)
    • is_exact - True если аспект очень точный
    • is_applying - True если аспект формируется (планеты сближаются)
    ═══════════════════════════════════════════════════════════════════════════════
    
    Args:
        bodies_by_date: Dict mapping date -> list of BodyPosition
        settings: AstroSettings with aspect configurations
        orb_mult: Orb multiplier (1.0 = default orbs)
        progress: Show progress bar
        prefix: Prefix to add to planet names (e.g., "helio_")
    
    Returns:
        DataFrame with aspect data
    """
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 1: Масштабируем орбисы согласно множителю
    # ─────────────────────────────────────────────────────────────────────────────
    aspects_cfg = scale_aspects(settings.aspects, orb_mult)
    aspects_rows = []
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 2: Проходим по каждой дате и вычисляем все аспекты между планетами
    # ─────────────────────────────────────────────────────────────────────────────
    iterator = tqdm(bodies_by_date.items(), desc=f"Calculating aspects (orb={orb_mult})") if progress else bodies_by_date.items()
    
    for d, bodies in iterator:
        # calculate_aspects возвращает список всех аспектов между парами планет
        aspects = calculate_aspects(bodies, aspects_cfg)
        for a in aspects:
            aspects_rows.append({
                "date": a.date,
                "p1": prefix + a.p1,        # Добавляем префикс если есть
                "p2": prefix + a.p2,        # (для различения geo/helio)
                "aspect": a.aspect,
                "orb": a.orb,
                "is_exact": a.is_exact,
                "is_applying": a.is_applying,
            })
    
    return pd.DataFrame(aspects_rows)


def calculate_aspects_for_dates_multi(
    geo_bodies_by_date: dict,
    helio_bodies_by_date: dict,
    settings: AstroSettings,
    coord_mode: str = "geo",
    orb_mult: float = 1.0,
    progress: bool = True,
) -> pd.DataFrame:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    РАСЧЁТ АСПЕКТОВ ДЛЯ РАЗНЫХ КООРДИНАТНЫХ СИСТЕМ
    ═══════════════════════════════════════════════════════════════════════════════
    
    Эта функция вычисляет аспекты для выбранного режима координат:
    
    • "geo"  — аспекты только в геоцентрической системе
    • "helio"— аспекты только в гелиоцентрической системе  
    • "both" — аспекты в ОБЕИХ системах, объединённые в один DataFrame
              Geo-аспекты: p1="Sun", p2="Moon"
              Helio-аспекты: p1="helio_Earth", p2="helio_Mars"
    
    ЗАЧЕМ ЭТО НУЖНО:
    ─────────────────────────────────────────────────────────────────────────────
    Гелио-аспекты показывают "чистую" геометрию Солнечной системы.
    Geo-аспекты показывают, как это выглядит с Земли (что мы реально видим).
    Режим "both" даёт модели оба вида информации — пусть сама решит!
    ═══════════════════════════════════════════════════════════════════════════════
    """
    all_dfs = []
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Геоцентрические аспекты (если нужны)
    # ─────────────────────────────────────────────────────────────────────────────
    if coord_mode in ["geo", "both"] and geo_bodies_by_date:
        prefix = "geo_" if coord_mode == "both" else ""
        if progress:
            print(f"📐 Расчёт GEO-аспектов (orb×{orb_mult})...")
        df_geo_aspects = calculate_aspects_for_dates(
            geo_bodies_by_date, settings, orb_mult, progress, prefix=prefix
        )
        all_dfs.append(df_geo_aspects)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Гелиоцентрические аспекты (если нужны)
    # ─────────────────────────────────────────────────────────────────────────────
    if coord_mode in ["helio", "both"] and helio_bodies_by_date:
        prefix = "helio_"  # Всегда добавляем префикс для helio
        if progress:
            print(f"☀️ Расчёт HELIO-аспектов (orb×{orb_mult})...")
        df_helio_aspects = calculate_aspects_for_dates(
            helio_bodies_by_date, settings, orb_mult, progress, prefix=prefix
        )
        all_dfs.append(df_helio_aspects)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Объединяем результаты
    # ─────────────────────────────────────────────────────────────────────────────
    if not all_dfs:
        return pd.DataFrame()
    
    if len(all_dfs) == 1:
        return all_dfs[0]
    
    result = pd.concat(all_dfs, ignore_index=True)
    if progress:
        print(f"✅ Объединено {len(result)} аспектов из {len(all_dfs)} систем")
    
    return result


def calculate_transits_for_dates(
    bodies_by_date: dict,
    natal_bodies: List[BodyPosition],
    settings: AstroSettings,
    orb_mult: float = 1.0,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Calculate transit-to-natal aspects for all dates.
    
    Args:
        bodies_by_date: Dict mapping date -> list of BodyPosition
        natal_bodies: Natal chart body positions
        settings: AstroSettings with aspect configurations
        orb_mult: Orb multiplier
        progress: Show progress bar
    
    Returns:
        DataFrame with transit aspect data
    """
    aspects_cfg = scale_aspects(settings.aspects, orb_mult)
    transit_rows = []
    
    iterator = tqdm(bodies_by_date.items(), desc=f"Calculating transits (orb={orb_mult})") if progress else bodies_by_date.items()
    
    for d, bodies in iterator:
        hits = calculate_transit_aspects(bodies, natal_bodies, aspects_cfg)
        for h in hits:
            transit_rows.append({
                "date": h.date,
                "transit_body": h.transit_body,
                "natal_body": h.natal_body,
                "aspect": h.aspect,
                "orb": h.orb,
                "is_exact": h.is_exact,
                "is_applying": h.is_applying,
            })
    
    return pd.DataFrame(transit_rows)


def get_natal_bodies(
    birth_dt_str: str,
    settings: AstroSettings,
    center: str = "geo",
) -> List[BodyPosition]:
    """
    Calculate natal chart body positions.
    
    Args:
        birth_dt_str: Birth datetime string (ISO format)
        settings: AstroSettings with body configurations
        center: Coordinate center
    
    Returns:
        List of natal body positions
    """
    birth_dt = parse_birth_dt_utc(birth_dt_str)
    return calculate_bodies(birth_dt, settings.bodies, center=center)


# ═══════════════════════════════════════════════════════════════════════════════
# ФАЗЫ ЛУНЫ И ЭЛОНГАЦИИ ПЛАНЕТ
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_moon_phase(sun_lon: float, moon_lon: float) -> dict:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    РАСЧЁТ ФАЗЫ ЛУНЫ
    ═══════════════════════════════════════════════════════════════════════════════
    
    ЧТО ТАКОЕ ФАЗА ЛУНЫ:
    ─────────────────────────────────────────────────────────────────────────────
    Фаза Луны — это угол между Солнцем и Луной, если смотреть с Земли.
    Этот угол определяет, какая часть Луны освещена и видна нам.
    
    УГЛЫ И ФАЗЫ:
    ─────────────────────────────────────────────────────────────────────────────
    • 0°   = Новолуние (Луна не видна, рядом с Солнцем)
    • 90°  = Первая четверть (половина диска освещена)
    • 180° = Полнолуние (вся Луна освещена)
    • 270° = Последняя четверть (другая половина освещена)
    
    ВОЗВРАЩАЕТ словарь:
    ─────────────────────────────────────────────────────────────────────────────
    • phase_angle — угол 0-360° (угловое расстояние Луна-Солнце)
    • phase_ratio — 0.0-1.0 (0=новолуние, 0.5=полнолуние, 1=снова новолуние)
    • illumination — 0.0-1.0 (доля освещённого диска)
    • lunar_day — 1-29.5 (лунный день, используется в астрологии)
    • phase_name — название фазы на русском
    ═══════════════════════════════════════════════════════════════════════════════
    
    Args:
        sun_lon: Долгота Солнца (0-360°)
        moon_lon: Долгота Луны (0-360°)
    
    Returns:
        Dict with moon phase information
    """
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 1: Вычисляем угол между Луной и Солнцем
    # ─────────────────────────────────────────────────────────────────────────────
    # Луна движется БЫСТРЕЕ Солнца (примерно 13°/день против 1°/день)
    # Угол фазы = позиция Луны минус позиция Солнца
    phase_angle = (moon_lon - sun_lon) % 360.0
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 2: Преобразуем угол в ratio (0-1)
    # ─────────────────────────────────────────────────────────────────────────────
    # 0 = новолуние, 0.5 = полнолуние, 1 = снова новолуние
    phase_ratio = phase_angle / 360.0
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 3: Вычисляем освещённость диска (0-1)
    # ─────────────────────────────────────────────────────────────────────────────
    # Формула: (1 - cos(угол)) / 2
    # При 0° → cos=1 → освещённость=0 (новолуние)
    # При 180° → cos=-1 → освещённость=1 (полнолуние)
    import math
    illumination = (1 - math.cos(math.radians(phase_angle))) / 2.0
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 4: Вычисляем лунный день (1-29.5)
    # ─────────────────────────────────────────────────────────────────────────────
    # Лунный месяц = 29.53 дней (синодический период)
    # Лунный день = (угол / 360) * 29.53 + 1
    SYNODIC_MONTH = 29.53059
    lunar_day = (phase_angle / 360.0) * SYNODIC_MONTH + 1.0
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 5: Определяем название фазы
    # ─────────────────────────────────────────────────────────────────────────────
    if phase_angle < 22.5 or phase_angle >= 337.5:
        phase_name = "Новолуние"
    elif phase_angle < 67.5:
        phase_name = "Молодая Луна"
    elif phase_angle < 112.5:
        phase_name = "Первая четверть"
    elif phase_angle < 157.5:
        phase_name = "Прибывающая Луна"
    elif phase_angle < 202.5:
        phase_name = "Полнолуние"
    elif phase_angle < 247.5:
        phase_name = "Убывающая Луна"
    elif phase_angle < 292.5:
        phase_name = "Последняя четверть"
    else:
        phase_name = "Старая Луна"
    
    return {
        "phase_angle": phase_angle,      # Угол 0-360°
        "phase_ratio": phase_ratio,      # 0-1 (нормализованный)
        "illumination": illumination,    # Освещённость 0-1
        "lunar_day": lunar_day,          # Лунный день 1-29.5
        "phase_name": phase_name,        # Название фазы
    }


def calculate_planet_elongation(sun_lon: float, planet_lon: float) -> dict:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    РАСЧЁТ ЭЛОНГАЦИИ ПЛАНЕТЫ
    ═══════════════════════════════════════════════════════════════════════════════
    
    ЧТО ТАКОЕ ЭЛОНГАЦИЯ:
    ─────────────────────────────────────────────────────────────────────────────
    Элонгация — это угловое расстояние планеты от Солнца (если смотреть с Земли).
    
    ПОЧЕМУ ЭТО ВАЖНО:
    ─────────────────────────────────────────────────────────────────────────────
    • Меркурий и Венера (внутренние планеты) никогда не уходят далеко от Солнца
      - Меркурий: максимум ~28°
      - Венера: максимум ~47°
    • Когда элонгация близка к 0° — планета "за Солнцем" или "перед Солнцем"
    • Максимальная элонгация — лучшее время для наблюдения
    
    ДЛЯ ФИНАНСОВОЙ АСТРОЛОГИИ:
    ─────────────────────────────────────────────────────────────────────────────
    • Нижнее соединение (Меркурий/Венера между нами и Солнцем) — начало нового цикла
    • Верхнее соединение (планета "за" Солнцем) — середина цикла
    • Максимальная элонгация — пик видимости и "силы" планеты
    ═══════════════════════════════════════════════════════════════════════════════
    
    Args:
        sun_lon: Долгота Солнца (0-360°)
        planet_lon: Долгота планеты (0-360°)
    
    Returns:
        Dict with elongation info
    """
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 1: Вычисляем разницу долгот
    # ─────────────────────────────────────────────────────────────────────────────
    diff = (planet_lon - sun_lon) % 360.0
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 2: Преобразуем в диапазон -180 до +180
    # ─────────────────────────────────────────────────────────────────────────────
    # Положительная элонгация = планета ВОСТОЧНЕЕ Солнца (вечерняя звезда)
    # Отрицательная элонгация = планета ЗАПАДНЕЕ Солнца (утренняя звезда)
    if diff > 180:
        diff = diff - 360
    
    elongation = diff  # -180 до +180
    elongation_abs = abs(diff)  # Абсолютное значение 0-180
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 3: Определяем позицию (утренняя/вечерняя)
    # ─────────────────────────────────────────────────────────────────────────────
    if elongation > 0:
        position = "evening"  # Вечерняя (после захода Солнца)
    elif elongation < 0:
        position = "morning"  # Утренняя (до восхода Солнца)
    else:
        position = "conjunction"  # Соединение с Солнцем
    
    return {
        "elongation": elongation,        # -180 до +180°
        "elongation_abs": elongation_abs,  # 0-180° (без знака)
        "position": position,            # morning/evening/conjunction
    }


def calculate_phases_for_dates(
    bodies_by_date: dict,
    progress: bool = True,
) -> pd.DataFrame:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    РАСЧЁТ ФАЗ ЛУНЫ И ЭЛОНГАЦИЙ ДЛЯ ВСЕХ ДАТ
    ═══════════════════════════════════════════════════════════════════════════════
    
    Эта функция вычисляет:
    1. Фазу Луны (угол, освещённость, лунный день)
    2. Элонгации ВСЕХ планет от Солнца
    
    ДЛЯ КАЖДОЙ ДАТЫ возвращает:
    ─────────────────────────────────────────────────────────────────────────────
    • moon_phase_angle    — угол фазы Луны (0-360°)
    • moon_phase_ratio    — нормализованная фаза (0-1)
    • moon_illumination   — освещённость диска (0-1)
    • lunar_day           — лунный день (1-29.5)
    • Mercury_elongation  — элонгация Меркурия (-180 до +180°)
    • Venus_elongation    — элонгация Венеры
    • Mars_elongation     — элонгация Марса
    • ... и т.д. для всех планет
    ═══════════════════════════════════════════════════════════════════════════════
    
    Args:
        bodies_by_date: Dict {date: [BodyPosition, ...]} from calculate_bodies_for_dates
        progress: Show progress bar
    
    Returns:
        DataFrame with phase/elongation data per date
    """
    from tqdm import tqdm
    
    rows = []
    iterator = tqdm(bodies_by_date.items(), desc="Calculating phases & elongations") if progress else bodies_by_date.items()
    
    for d, bodies in iterator:
        row = {"date": d}
        
        # ─────────────────────────────────────────────────────────────────────────
        # Находим позиции Солнца и Луны
        # ─────────────────────────────────────────────────────────────────────────
        sun_lon = None
        moon_lon = None
        planet_lons = {}
        
        for b in bodies:
            if b.body == "Sun":
                sun_lon = b.lon
            elif b.body == "Moon":
                moon_lon = b.lon
            else:
                planet_lons[b.body] = b.lon
        
        # ─────────────────────────────────────────────────────────────────────────
        # Фаза Луны (если есть и Солнце и Луна)
        # ─────────────────────────────────────────────────────────────────────────
        if sun_lon is not None and moon_lon is not None:
            moon_phase = calculate_moon_phase(sun_lon, moon_lon)
            row["moon_phase_angle"] = moon_phase["phase_angle"]
            row["moon_phase_ratio"] = moon_phase["phase_ratio"]
            row["moon_illumination"] = moon_phase["illumination"]
            row["lunar_day"] = moon_phase["lunar_day"]
        
        # ─────────────────────────────────────────────────────────────────────────
        # Элонгации всех планет от Солнца
        # ─────────────────────────────────────────────────────────────────────────
        if sun_lon is not None:
            for planet, lon in planet_lons.items():
                elong = calculate_planet_elongation(sun_lon, lon)
                row[f"{planet}_elongation"] = elong["elongation"]
                row[f"{planet}_elongation_abs"] = elong["elongation_abs"]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if progress:
        print(f"✅ Рассчитано {len(df)} дней: фаза Луны + элонгации планет")
    
    return df


# Сохраняем старую функцию для совместимости
def calculate_moon_phases(dates: pd.Series) -> pd.Series:
    """
    DEPRECATED: Use calculate_phases_for_dates instead.
    This is kept for backwards compatibility.
    """
    # Просто возвращаем NaN - используйте calculate_phases_for_dates
    return pd.Series(np.nan, index=dates.index)
