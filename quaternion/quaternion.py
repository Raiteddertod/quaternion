# quaternion/quaternion.py
"""
Класс Quaternion.

Реализует кватернионы в конвенции [w, x, y, z].

"""

from __future__ import annotations

import numpy as np
from typing import Union, Optional


class Quaternion:
    """
    Кватернион в представлении: q = [w, x, y, z].

    Поддерживает:
    - Кватернионное умножение (Гамильтона)
    - Сопряжение, нормализацию, инверсию
    - Поворот векторов (q ⊙ v)
    - Создание из оси и угла
    - Преобразование в/из матрицы поворота (DCM)
    """

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        Создаёт кватернион.

        Параметры:
            w, x, y, z : float
                Компоненты кватерниона (scalar-first).
        """
        self._w = float(w)
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------
    @property
    def w(self) -> float:
        """Скалярная часть кватерниона."""
        return self._w

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def z(self) -> float:
        return self._z

    @property
    def scalar(self) -> float:
        """Скалярная часть (w)."""
        return self._w

    @property
    def vec(self) -> np.ndarray:
        """Векторная часть [x, y, z]."""
        return np.array([self._x, self._y, self._z])

    # ------------------------------------------------------------------
    # Основные операции
    # ------------------------------------------------------------------
    
    def __mul__(self, other):
        """
        Поддерживает два вида умножения:
        - Кватернион × Кватернион (Гамильтона)
        - Кватернион × скаляр
        """
        if isinstance(other, Quaternion):
            # Кватернионное умножение
            w1, x1, y1, z1 = self._w, self._x, self._y, self._z
            w2, x2, y2, z2 = other._w, other._x, other._y, other._z

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return Quaternion(w, x, y, z)

        elif isinstance(other, (int, float)):
            # Умножение кватерниона на число
            return Quaternion(
                self._w * other,
                self._x * other,
                self._y * other,
                self._z * other
            )

        return NotImplemented


    def __rmul__(self, other):
        """
        Поддержка умножения слева: 0.5 * q
        """
        if isinstance(other, (int, float)):
            return self * other   # используем уже исправленный __mul__
        return NotImplemented


    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        """Сложение двух кватернионов (нужно для интегрирования)."""
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(
            self._w + other._w,
            self._x + other._x,
            self._y + other._y,
            self._z + other._z
        )

    def conjugate(self) -> Quaternion:
        """Возвращает сопряжённый кватернион: [w, -x, -y, -z]."""
        return Quaternion(self._w, -self._x, -self._y, -self._z)

    def inverse(self) -> Quaternion:
        """
        Возвращает обратный кватернион q^{-1} = q* / ||q||^2.
        Для единичных кватернионов совпадает с сопряжённым.
        """
        norm_sq = self.norm_squared()
        if norm_sq < 1e-12:
            raise ValueError("Невозможно инвертировать нулевой кватернион")
        return self.conjugate() / norm_sq

    def normalize(self) -> Quaternion:
        """Возвращает нормализованную (единичную) версию кватерниона."""
        n = self.norm()
        if n < 1e-12:
            return Quaternion.identity()
        return Quaternion(self._w / n, self._x / n, self._y / n, self._z / n)

    def norm(self) -> float:
        """Евклидова норма кватерниона."""
        return np.sqrt(self.norm_squared())

    def norm_squared(self) -> float:
        """Квадрат нормы."""
        return self._w**2 + self._x**2 + self._y**2 + self._z**2

    # ------------------------------------------------------------------
    # Поворот вектора
    # ------------------------------------------------------------------
    def rotate(self, v: np.ndarray) -> np.ndarray:
        """
        Поворачивает вектор v этим кватернионом: v' = q ⊙ v.

        Эквивалентно: q * [0, v] * q^{-1}
        """
        v = np.asarray(v, dtype=float)
        if v.shape != (3,):
            raise ValueError("Вектор должен иметь размерность 3")

        # q * [0, vx, vy, vz]
        qv = Quaternion(0.0, v[0], v[1], v[2])
        result = self * qv * self.conjugate()
        return result.vec

    # ------------------------------------------------------------------
    # Фабричные методы
    # ------------------------------------------------------------------
    @classmethod
    def identity(cls) -> Quaternion:
        """Возвращает единичный кватернион [1, 0, 0, 0]."""
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> Quaternion:
        """
        Создаёт кватернион из оси вращения и угла.

        Параметры:
            axis  : массив [3] — ось вращения (не обязательно единичная)
            angle : угол в радианах
        """
        axis = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return cls.identity()

        axis = axis / n
        half_angle = angle / 2.0
        s = np.sin(half_angle)
        return cls(
            w=np.cos(half_angle),
            x=axis[0] * s,
            y=axis[1] * s,
            z=axis[2] * s
        )

    @classmethod
    def from_rotation_matrix(cls, R: np.ndarray) -> Quaternion:
        """
        Создаёт кватернион из матрицы поворота (DCM) 3x3.

        Использует устойчивый алгоритм Shepperd'а.
        """
        R = np.asarray(R, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("Матрица поворота должна быть 3x3")

        # Shepperd's method
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        q = cls(w, x, y, z)
        return q.normalize()

    # ------------------------------------------------------------------
    # Преобразования
    # ------------------------------------------------------------------
    def to_rotation_matrix(self) -> np.ndarray:
        """
        Преобразует кватернион в матрицу поворота (DCM) 3x3.

        R = I + 2 * [ -||v||^2    -v_z    v_y  ]
                    [  v_z     -||v||^2  -v_x  ]
                    [ -v_y      v_x    -||v||^2 ]
        """
        w, x, y, z = self._w, self._x, self._y, self._z

        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        R = np.array([
            [ww + xx - yy - zz, 2*(xy - wz),     2*(xz + wy)],
            [2*(xy + wz),     ww - xx + yy - zz, 2*(yz - wx)],
            [2*(xz - wy),     2*(yz + wx),     ww - xx - yy + zz]
        ])
        return R

    def to_array(self) -> np.ndarray:
        """Возвращает кватернион как numpy-массив [w, x, y, z]."""
        return np.array([self._w, self._x, self._y, self._z])

    # ------------------------------------------------------------------
    # Специальные методы
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Quaternion(w={self._w:.6f}, x={self._x:.6f}, y={self._y:.6f}, z={self._z:.6f})"

    def __str__(self) -> str:
        return f"[{self._w:.4f}, {self._x:.4f}, {self._y:.4f}, {self._z:.4f}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return NotImplemented
        return np.allclose(self.to_array(), other.to_array(), atol=1e-10)

    def __truediv__(self, scalar: float) -> Quaternion:
        return Quaternion(self._w / scalar, self._x / scalar, self._y / scalar, self._z / scalar)

    def __neg__(self) -> Quaternion:
        return Quaternion(-self._w, -self._x, -self._y, -self._z)


# ----------------------------------------------------------------------
# Вспомогательная функция
# ----------------------------------------------------------------------
def quaternion_from_array(arr: Union[list, np.ndarray]) -> Quaternion:
    """Создаёт Quaternion из массива или списка из 4 элементов."""
    arr = np.asarray(arr, dtype=float)
    if arr.shape != (4,):
        raise ValueError("Ожидается массив из 4 элементов [w, x, y, z]")
    return Quaternion(arr[0], arr[1], arr[2], arr[3])