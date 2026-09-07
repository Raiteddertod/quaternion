# quaternion/conversions.py
"""
Преобразования между кватернионами, матрицами поворота и углами Эйлера.

Содержит как обёртки над методами класса Quaternion, так и дополнительные
преобразования, удобные для контроллера ориентации и отладки.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from .quaternion import Quaternion


# ----------------------------------------------------------------------
# Кватернион ↔ Матрица поворота (DCM)
# ----------------------------------------------------------------------

def quaternion_to_rotation_matrix(q: Quaternion) -> np.ndarray:
    """
    Преобразует кватернион в матрицу поворота (Direction Cosine Matrix).

    Соответствует методу to_rotation_matrix() класса Quaternion.

    Параметры:
        q : Quaternion

    Возвращает:
        np.ndarray (3, 3) — матрица поворота
    """
    if not isinstance(q, Quaternion):
        raise TypeError("Ожидается объект Quaternion")
    return q.to_rotation_matrix()


def rotation_matrix_to_quaternion(R: np.ndarray) -> Quaternion:
    """
    Преобразует матрицу поворота в кватернион.

    Соответствует методу from_rotation_matrix() класса Quaternion.
    Использует устойчивый алгоритм Shepperd'а.

    Параметры:
        R : np.ndarray (3, 3)

    Возвращает:
        Quaternion
    """
    return Quaternion.from_rotation_matrix(R)


# ----------------------------------------------------------------------
# Кватернион ↔ Углы Эйлера (ZYX convention)
# ----------------------------------------------------------------------

def quaternion_to_euler(
    q: Quaternion,
    degrees: bool = False
) -> Tuple[float, float, float]:
    """
    Преобразует кватернион в углы Эйлера (Roll, Pitch, Yaw).

    Используется последовательность вращений ZYX (yaw → pitch → roll).
    Это наиболее распространённая конвенция в авиации и робототехнике.

    Параметры:
        q       : Quaternion
        degrees : если True — возвращает углы в градусах, иначе в радианах

    Возвращает:
        (roll, pitch, yaw) — углы в радианах или градусах
    """
    if not isinstance(q, Quaternion):
        raise TypeError("Ожидается объект Quaternion")

    R = q.to_rotation_matrix()

    # Извлечение углов из матрицы поворота (ZYX)
    # Предотвращаем численные ошибки за пределами [-1, 1]
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))

    # Особые случаи (gimbal lock)
    if abs(R[2, 0]) > 0.9999999:
        # pitch ≈ ±90°
        yaw = 0.0
        roll = np.arctan2(R[0, 1], R[0, 2])
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])

    if degrees:
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    return float(roll), float(pitch), float(yaw)


def euler_to_quaternion(
    roll: float,
    pitch: float,
    yaw: float,
    degrees: bool = False
) -> Quaternion:
    """
    Создаёт кватернион из углов Эйлера (Roll, Pitch, Yaw).

    Последовательность вращений: Z(yaw) → Y(pitch) → X(roll).

    Параметры:
        roll, pitch, yaw : углы в радианах (или градусах, если degrees=True)
        degrees          : интерпретировать входные углы как градусы

    Возвращает:
        Quaternion
    """
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return Quaternion(w, x, y, z)


# ----------------------------------------------------------------------
# Извлечение и создание yaw (рысканья)
# ----------------------------------------------------------------------

def extract_yaw(q: Quaternion) -> float:
    """
    Извлекает угол рысканья (heading, yaw) из кватерниона.

    Удобно для получения текущего ψ из q_hat.

    Возвращает:
        yaw в радианах, в диапазоне [-π, π]
    """
    _, _, yaw = quaternion_to_euler(q)
    return yaw


def quaternion_from_yaw(yaw: float, degrees: bool = False) -> Quaternion:
    """
    Создаёт кватернион, описывающий только поворот вокруг оси Z (yaw).

    Полезно для построения референсной ориентации по ψ_ref.

    Параметры:
        yaw     : угол рысканья в радианах (или градусах)
        degrees : если True — yaw интерпретируется как градусы

    Возвращает:
        Quaternion, соответствующий повороту только по yaw
    """
    if degrees:
        yaw = np.radians(yaw)

    return Quaternion(
        w=np.cos(yaw * 0.5),
        x=0.0,
        y=0.0,
        z=np.sin(yaw * 0.5)
    )


# ----------------------------------------------------------------------
# Построение ориентации из осей (используется в контроллере)
# ----------------------------------------------------------------------

def rotation_matrix_from_axes(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Строит матрицу поворота из трёх осей тела.

    Оси должны быть заданы в связанной системе координат и образовывать
    правую тройку.

    Используется при построении желаемой ориентации q_des (уравнения 18–19).

    Параметры:
        x, y, z   : оси (векторы длины 3)
        normalize : нормализовать ли оси перед построением матрицы

    Возвращает:
        np.ndarray (3, 3) — матрица поворота, где столбцы = [x, y, z]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if normalize:
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-12 else x
        y = y / np.linalg.norm(y) if np.linalg.norm(y) > 1e-12 else y
        z = z / np.linalg.norm(z) if np.linalg.norm(z) > 1e-12 else z

    R = np.column_stack((x, y, z))
    return R


def quaternion_from_axes(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    normalize: bool = True
) -> Quaternion:
    """
    Создаёт кватернион непосредственно из трёх осей тела.

    Это удобная обёртка над rotation_matrix_from_axes + преобразованием.

    Параметры:
        x, y, z   : оси тела
        normalize : нормализовать ли оси

    Возвращает:
        Quaternion, соответствующий данной ориентации
    """
    R = rotation_matrix_from_axes(x, y, z, normalize=normalize)
    return rotation_matrix_to_quaternion(R)


# ----------------------------------------------------------------------
# Дополнительные полезные преобразования
# ----------------------------------------------------------------------

def quaternion_to_axis_angle(q: Quaternion) -> Tuple[np.ndarray, float]:
    """
    Преобразует кватернион в ось и угол вращения.

    Возвращает:
        (axis, angle) — единичный вектор оси и угол в радианах
    """
    q = q.normalize()
    angle = 2.0 * np.arccos(np.clip(q.w, -1.0, 1.0))

    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0

    s = np.sin(angle / 2.0)
    axis = np.array([q.x / s, q.y / s, q.z / s])
    return axis, angle


def axis_angle_to_quaternion(axis: np.ndarray, angle: float) -> Quaternion:
    """
    Создаёт кватернион из оси и угла.

    Обёртка над Quaternion.from_axis_angle.
    """
    return Quaternion.from_axis_angle(axis, angle)