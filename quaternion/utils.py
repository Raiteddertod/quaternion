# quaternion/utils.py
"""
Вспомогательные функции для работы с векторами и численной устойчивости.

Используются в контроллере ориентации для безопасных операций
с векторами и обработки особых случаев (деление на ноль, параллельные векторы и т.д.).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ----------------------------------------------------------------------
# Константы для численной устойчивости
# ----------------------------------------------------------------------
EPSILON = 1e-12
"""Малое число для проверки на ноль и защиты от деления на ноль."""


# ----------------------------------------------------------------------
# Нормализация векторов
# ----------------------------------------------------------------------
def normalize_vector(
    v: np.ndarray,
    eps: float = EPSILON,
    fallback: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Нормализует вектор к единичной длине.

    Если норма вектора меньше eps, возвращает fallback (по умолчанию [0, 0, 1]).

    Используется в уравнениях 10, 12, 18, 19.

    Параметры:
        v        : массив [3] — вектор для нормализации
        eps      : порог, ниже которого вектор считается нулевым
        fallback : вектор, который возвращается при нулевой норме

    Возвращает:
        np.ndarray [3] — единичный вектор
    """
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError("Вектор должен иметь размерность 3")

    norm = np.linalg.norm(v)

    if norm < eps:
        if fallback is None:
            return np.array([0.0, 0.0, 1.0])
        return np.asarray(fallback, dtype=float)

    return v / norm


def is_zero_vector(v: np.ndarray, eps: float = EPSILON) -> bool:
    """
    Проверяет, является ли вектор практически нулевым.

    Параметры:
        v   : массив [3]
        eps : порог

    Возвращает:
        True, если норма вектора меньше eps
    """
    return np.linalg.norm(v) < eps


# ----------------------------------------------------------------------
# Безопасные векторные операции
# ----------------------------------------------------------------------
def safe_cross(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = EPSILON
) -> Tuple[np.ndarray, float]:
    """
    Вычисляет векторное произведение a × b с защитой от нулевого результата.

    Возвращает также норму полученного вектора.

    Используется в уравнениях 12, 18, 19.

    Параметры:
        a, b : массивы [3]
        eps  : порог для определения нулевого результата

    Возвращает:
        (cross_product, norm) — векторное произведение и его норма
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    cross = np.cross(a, b)
    norm = np.linalg.norm(cross)

    if norm < eps:
        return np.zeros(3), 0.0

    return cross, norm


def safe_cross_normalized(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = EPSILON,
    fallback: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Вычисляет единичный вектор в направлении a × b.

    Если векторное произведение близко к нулю, возвращает fallback.

    Параметры:
        a, b     : массивы [3]
        eps      : порог
        fallback : вектор, возвращаемый при нулевом произведении

    Возвращает:
        np.ndarray [3] — единичный вектор или fallback
    """
    cross, norm = safe_cross(a, b, eps)

    if norm < eps:
        if fallback is None:
            return np.array([0.0, 0.0, 1.0])
        return np.asarray(fallback, dtype=float)

    return cross / norm


# ----------------------------------------------------------------------
# Скалярное произведение и углы
# ----------------------------------------------------------------------
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Скалярное произведение двух векторов."""
    return float(np.dot(a, b))


def angle_between_vectors(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = EPSILON
) -> float:
    """
    Вычисляет угол (в радианах) между двумя векторами.

    Используется в уравнении 11: α = arccos(e_z_B · e_z,des_B)

    Результат всегда находится в диапазоне [0, π].
    """
    a = normalize_vector(a, eps)
    b = normalize_vector(b, eps)

    cos_angle = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.arccos(cos_angle))


# ----------------------------------------------------------------------
# Работа с кватернионами ошибок (для уравнений 15 и 20)
# ----------------------------------------------------------------------
def get_signed_rate_from_error(
    q_error_w: float,
    q_error_component: float,
    gain: float
) -> float:
    """
    Вычисляет желаемую угловую скорость по компоненте ошибки кватерниона.

    Реализует логику из уравнений 15 и 20:
        rate =  2 * gain * component    если w >= 0
        rate = -2 * gain * component    если w <  0

    Это обеспечивает правильное направление вращения.

    Параметры:
        q_error_w         : скалярная часть кватерниона ошибки
        q_error_component : соответствующая компонента (x, y или z)
        gain              : коэффициент усиления (p_rp или p_yaw)

    Возвращает:
        желаемая угловая скорость (p_des, q_des или r_des)
    """
    if q_error_w >= 0:
        return 2.0 * gain * q_error_component
    else:
        return -2.0 * gain * q_error_component


# ----------------------------------------------------------------------
# Проверка особых случаев
# ----------------------------------------------------------------------
def handle_zero_acceleration(
    a_des: np.ndarray,
    eps: float = EPSILON
) -> Tuple[np.ndarray, bool]:
    """
    Обрабатывает случай нулевого желаемого ускорения (уравнение 10).

    Если ||a_des|| ≈ 0, то желаемая ось Z корпуса = [0, 0, 1].

    Параметры:
        a_des : вектор желаемого ускорения [3]
        eps   : порог

    Возвращает:
        (e_z_des_B, is_zero) — желаемая ось Z и флаг, что ускорение было нулевым
    """
    a_des = np.asarray(a_des, dtype=float)
    norm = np.linalg.norm(a_des)

    if norm < eps:
        return np.array([0.0, 0.0, 1.0]), True

    return a_des / norm, False


def is_singularity_case(
    cross_norm: float,
    eps: float = EPSILON
) -> bool:
    """
    Проверяет, находится ли ситуация в особом случае (параллельные векторы).

    Используется для определения, когда α = 0 или α = π
    (уравнения 12, 18).
    """
    return cross_norm < eps


# ----------------------------------------------------------------------
# Утилиты для отладки и валидации
# ----------------------------------------------------------------------
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Ограничивает значение диапазоном [min_val, max_val]."""
    return max(min_val, min(value, max_val))


def almost_equal(a: float, b: float, eps: float = 1e-9) -> bool:
    """Проверяет приближённое равенство двух чисел."""
    return abs(a - b) < eps


def validate_3d_vector(v: np.ndarray, name: str = "vector") -> None:
    """
    Проверяет, что вход является 3-мерным вектором.
    Выбрасывает ValueError при несоответствии.
    """
    v = np.asarray(v)
    if v.shape != (3,):
        raise ValueError(f"{name} должен быть массивом из 3 элементов, получен shape={v.shape}")