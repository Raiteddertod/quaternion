# tests/test_quaternions.py
"""
Тесты для класса Quaternion.

Проверяются:
- Базовые операции (умножение, сопряжение, нормализация, инверсия)
- Поворот векторов
- Создание из оси/угла и матрицы поворота
- Взаимные преобразования (q <-> R)
- Особые случаи и численная устойчивость
"""

import numpy as np
import pytest

from quaternion.quaternion import Quaternion, quaternion_from_array


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def assert_quaternion_equal(q1: Quaternion, q2: Quaternion, atol: float = 1e-10):
    """Сравнивает два кватерниона с допуском."""
    np.testing.assert_allclose(q1.to_array(), q2.to_array(), atol=atol)


def assert_vector_equal(v1: np.ndarray, v2: np.ndarray, atol: float = 1e-10):
    """Сравнивает два вектора с допуском."""
    np.testing.assert_allclose(v1, v2, atol=atol)


# ----------------------------------------------------------------------
# Тесты конструктора и свойств
# ----------------------------------------------------------------------
def test_constructor_default():
    """Единичный кватернион по умолчанию."""
    q = Quaternion()
    assert q.w == 1.0
    assert q.x == 0.0
    assert q.y == 0.0
    assert q.z == 0.0
    print(q)


def test_constructor_from_values():
    q = Quaternion(0.5, 0.1, 0.2, 0.3)
    assert q.w == 0.5
    assert q.x == 0.1
    assert q.y == 0.2
    assert q.z == 0.3
    print(q)


def test_to_array():
    q = Quaternion(0.7071, 0.0, 0.0, 0.7071)
    arr = q.to_array()
    np.testing.assert_allclose(arr, [0.7071, 0.0, 0.0, 0.7071])


# ----------------------------------------------------------------------
# Тесты умножения
# ----------------------------------------------------------------------
def test_multiplication_identity():
    """q * 1 = q"""
    q = Quaternion(0.8, 0.1, 0.2, 0.3).normalize()
    assert_quaternion_equal(q * Quaternion.identity(), q)


def test_multiplication_inverse():
    """q * q^{-1} ≈ 1"""
    q = Quaternion(0.6, 0.1, 0.2, 0.3).normalize()
    q_inv = q.inverse()
    result = q * q_inv
    assert_quaternion_equal(result, Quaternion.identity(), atol=1e-12)


def test_multiplication_90deg_z():
    """
    Поворот на 90° вокруг Z:
    [0,1,0] должен перейти в [-1,0,0]
    """
    q = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
    v = np.array([0.0, 1.0, 0.0])
    v_rot = q.rotate(v)
    assert_vector_equal(v_rot, [-1.0, 0.0, 0.0])


def test_multiplication_associativity():
    """(q1 * q2) * q3 ≈ q1 * (q2 * q3)"""
    q1 = Quaternion(0.8, 0.1, 0.2, 0.3).normalize()
    q2 = Quaternion(0.7, 0.2, 0.1, 0.4).normalize()
    q3 = Quaternion(0.9, 0.05, 0.15, 0.25).normalize()

    left = (q1 * q2) * q3
    right = q1 * (q2 * q3)
    assert_quaternion_equal(left, right, atol=1e-10)


# ----------------------------------------------------------------------
# Тесты сопряжения, нормализации, инверсии
# ----------------------------------------------------------------------
def test_conjugate():
    q = Quaternion(0.5, 0.1, 0.2, 0.3)
    qc = q.conjugate()
    assert qc.w == 0.5
    assert qc.x == -0.1
    assert qc.y == -0.2
    assert qc.z == -0.3


def test_normalize():
    q = Quaternion(3.0, 4.0, 0.0, 0.0)  # норма = 5
    qn = q.normalize()
    assert abs(qn.norm() - 1.0) < 1e-12
    assert abs(qn.w - 0.6) < 1e-12
    assert abs(qn.x - 0.8) < 1e-12


def test_normalize_zero_quaternion():
    """Нормализация нулевого кватерниона должна возвращать identity."""
    q = Quaternion(0.0, 0.0, 0.0, 0.0)
    qn = q.normalize()
    assert_quaternion_equal(qn, Quaternion.identity())


def test_inverse_unit_quaternion():
    """Для единичного кватерниона q^{-1} = q*"""
    q = Quaternion(0.6, 0.1, 0.2, 0.3).normalize()
    assert_quaternion_equal(q.inverse(), q.conjugate())


def test_inverse_non_unit():
    q = Quaternion(2.0, 0.0, 0.0, 0.0)
    q_inv = q.inverse()
    assert abs(q_inv.w - 0.5) < 1e-12


# ----------------------------------------------------------------------
# Тесты поворота векторов
# ----------------------------------------------------------------------
def test_rotate_preserves_length():
    """Поворот не должен менять длину вектора."""
    q = Quaternion.from_axis_angle([1, 1, 0], 1.234)
    v = np.array([3.0, 4.0, 5.0])
    v_rot = q.rotate(v)
    assert abs(np.linalg.norm(v) - np.linalg.norm(v_rot)) < 1e-10


def test_rotate_x_axis_180deg():
    """Поворот на 180° вокруг X: Y -> -Y, Z -> -Z"""
    q = Quaternion.from_axis_angle([1, 0, 0], np.pi)
    v = np.array([0.0, 1.0, 1.0])
    v_rot = q.rotate(v)
    assert_vector_equal(v_rot, [0.0, -1.0, -1.0])


def test_rotate_composition():
    """
    Два последовательных поворота:
    q_total = q2 * q1
    """
    q1 = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)  # 90° вокруг Z
    q2 = Quaternion.from_axis_angle([1, 0, 0], np.pi / 2)  # 90° вокруг X

    v = np.array([0.0, 1.0, 0.0])

    # Сначала q1, потом q2
    v1 = q1.rotate(v)
    v2 = q2.rotate(v1)
    print(v1)
    print(v2)
    # То же самое через композицию
    q_total = q2 * q1
    v_total = q_total.rotate(v)
    print(v_total)

    assert_vector_equal(v2, v_total)


def test_rotate_identity():
    """identity.rotate(v) == v"""
    q = Quaternion.identity()
    v = np.array([1.0, 2.0, 3.0])
    assert_vector_equal(q.rotate(v), v)


# ----------------------------------------------------------------------
# Тесты from_axis_angle
# ----------------------------------------------------------------------
def test_from_axis_angle_90deg():
    q = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
    # Ожидаем [cos(45°), 0, 0, sin(45°)]
    expected = Quaternion(np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)
    assert_quaternion_equal(q, expected, atol=1e-10)


def test_from_axis_angle_zero_angle():
    q = Quaternion.from_axis_angle([1, 0, 0], 0.0)
    assert_quaternion_equal(q, Quaternion.identity())


def test_from_axis_angle_zero_axis():
    """Нулевая ось должна возвращать identity."""
    q = Quaternion.from_axis_angle([0, 0, 0], 1.0)
    assert_quaternion_equal(q, Quaternion.identity())


# ----------------------------------------------------------------------
# Тесты преобразования в/из матрицы поворота
# ----------------------------------------------------------------------
def test_to_rotation_matrix_identity():
    q = Quaternion.identity()
    R = q.to_rotation_matrix()
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_roundtrip_quaternion_rotation_matrix():
    """q -> R -> q должно давать тот же кватернион (с точностью до знака)."""
    original = Quaternion(0.8, 0.1, 0.2, 0.3).normalize()
    R = original.to_rotation_matrix()
    recovered = Quaternion.from_rotation_matrix(R)

    # Кватернион определён с точностью до знака
    assert (
        np.allclose(recovered.to_array(), original.to_array(), atol=1e-10) or
        np.allclose(recovered.to_array(), -original.to_array(), atol=1e-10)
    )


def test_rotation_matrix_90deg_z():
    """Матрица поворота на 90° вокруг Z."""
    q = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
    R = q.to_rotation_matrix()

    expected = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])
    np.testing.assert_allclose(R, expected, atol=1e-10)


def test_from_rotation_matrix_singularities():
    """Проверка устойчивости near gimbal lock."""
    # Поворот на 90° вокруг Y
    q = Quaternion.from_axis_angle([0, 1, 0], np.pi / 2)
    R = q.to_rotation_matrix()
    recovered = Quaternion.from_rotation_matrix(R)

    assert (
        np.allclose(recovered.to_array(), q.to_array(), atol=1e-9) or
        np.allclose(recovered.to_array(), -q.to_array(), atol=1e-9)
    )


# ----------------------------------------------------------------------
# Тесты нормы
# ----------------------------------------------------------------------
def test_norm():
    q = Quaternion(1, 2, 3, 4)
    expected_norm = np.sqrt(1 + 4 + 9 + 16)
    assert abs(q.norm() - expected_norm) < 1e-12


def test_norm_squared():
    q = Quaternion(1, 2, 3, 4)
    assert abs(q.norm_squared() - 30.0) < 1e-12


# ----------------------------------------------------------------------
# Тесты специальных случаев
# ----------------------------------------------------------------------
def test_quaternion_from_array():
    q = quaternion_from_array([0.5, 0.1, 0.2, 0.3])
    assert q.w == 0.5
    assert q.x == 0.1


def test_quaternion_from_array_wrong_size():
    with pytest.raises(ValueError):
        quaternion_from_array([1, 2, 3])


def test_equality():
    q1 = Quaternion(0.5, 0.1, 0.2, 0.3).normalize()
    q2 = Quaternion(0.5, 0.1, 0.2, 0.3).normalize()
    assert q1 == q2


def test_negative():
    q = Quaternion(0.5, 0.1, 0.2, 0.3)
    neg = -q
    assert neg.w == -0.5
    assert neg.x == -0.1