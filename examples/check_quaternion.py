# examples/manual_check_quaternion.py
"""
Интерактивный скрипт для ручной проверки операций кватернионной алгебры.

Запуск:
    python examples/manual_check_quaternion.py

Позволяет пошагово вводить данные и смотреть результаты.
"""

import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from quaternion.quaternion import Quaternion
from quaternion.conversions import (
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    quaternion_from_yaw
)


def print_separator(title: str = ""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def input_quaternion(prompt: str = "Введите кватернион") -> Quaternion:
    """
    Запрашивает у пользователя кватернион в формате: w x y z
    Пример ввода: 0.7071 0 0 0.7071
    """
    while True:
        try:
            raw = input(f"{prompt} [w x y z]: ").strip()
            parts = [float(x) for x in raw.split()]
            if len(parts) != 4:
                print("Ошибка: нужно ввести ровно 4 числа (w x y z)")
                continue
            q = Quaternion(*parts).normalize()  # сразу нормализуем для удобства
            print(f"  → Получен кватернион: {q}")
            return q
        except ValueError:
            print("Ошибка: введите 4 числа через пробел, например: 1 0 0 0")


def input_vector(prompt: str = "Введите вектор") -> np.ndarray:
    """Запрашивает вектор из 3 чисел."""
    while True:
        try:
            raw = input(f"{prompt} [x y z]: ").strip()
            parts = [float(x) for x in raw.split()]
            if len(parts) != 3:
                print("Ошибка: нужно ввести ровно 3 числа")
                continue
            v = np.array(parts)
            print(f"  → Получен вектор: {v}")
            return v
        except ValueError:
            print("Ошибка: введите 3 числа через пробел, например: 1 0 0")


def input_angle(prompt: str = "Введите угол") -> float:
    """Запрашивает угол в градусах."""
    while True:
        try:
            deg = float(input(f"{prompt} (в градусах): "))
            rad = np.radians(deg)
            print(f"  → {deg}° = {rad:.6f} рад")
            return rad
        except ValueError:
            print("Ошибка: введите число")


# ----------------------------------------------------------------------
# Операции
# ----------------------------------------------------------------------

def op_create():
    print_separator("Создание кватерниона")
    q = input_quaternion("Введите кватернион")
    print(f"Норма: {q.norm():.10f}")
    print(f"Векторная часть: {q.vec}")


def op_multiply():
    print_separator("Умножение кватернионов (q1 * q2)")
    print("Сначала введите первый кватернион:")
    q1 = input_quaternion("q1")
    print("\nТеперь введите второй кватернион:")
    q2 = input_quaternion("q2")

    result = q1 * q2
    print(f"\nРезультат q1 * q2:")
    print(f"  {result}")
    print(f"  Норма результата: {result.norm():.10f}")


def op_conjugate_inverse_normalize():
    print_separator("Операции над кватернионом")
    q = input_quaternion("Введите кватернион")

    print(f"\nИсходный:           {q}")
    print(f"Сопряжённый:        {q.conjugate()}")
    print(f"Нормализованный:    {q.normalize()}")
    try:
        print(f"Обратный:           {q.inverse()}")
    except ValueError as e:
        print(f"Обратный:           Ошибка — {e}")


def op_rotate_vector():
    print_separator("Поворот вектора кватернионом (q ⊙ v)")
    q = input_quaternion("Кватернион поворота")
    v = input_vector("Вектор для поворота")

    v_rot = q.rotate(v)
    print(f"\nРезультат поворота:")
    print(f"  Исходный вектор:  {v}")
    print(f"  После поворота:   {v_rot}")
    print(f"  Длина до:         {np.linalg.norm(v):.6f}")
    print(f"  Длина после:      {np.linalg.norm(v_rot):.6f}")


def op_axis_angle():
    print_separator("Создание кватерниона из оси и угла")
    axis = input_vector("Ось вращения")
    angle_rad = input_angle("Угол поворота")

    q = Quaternion.from_axis_angle(axis, angle_rad)
    print(f"\nПолученный кватернион:")
    print(f"  {q}")
    print(f"  Норма: {q.norm():.10f}")

    # Покажем, что он действительно поворачивает
    v = input_vector("\n(Опционально) Введите вектор для проверки поворота")
    if np.linalg.norm(v) > 1e-9:
        v_rot = q.rotate(v)
        print(f"  Вектор после поворота: {v_rot}")


def op_to_matrix():
    print_separator("Кватернион → Матрица поворота")
    q = input_quaternion("Кватернион")
    R = q.to_rotation_matrix()

    print("\nМатрица поворота (3x3):")
    np.set_printoptions(precision=6, suppress=True)
    print(R)

    # Проверим, что это действительно матрица поворота
    print(f"\nПроверка: R @ R.T ≈ I ?")
    print(np.allclose(R @ R.T, np.eye(3)))


def op_from_matrix():
    print_separator("Матрица поворота → Кватернион")
    print("Введите матрицу 3x3 построчно (9 чисел через пробел):")
    try:
        raw = [float(x) for x in input().strip().split()]
        if len(raw) != 9:
            print("Ошибка: нужно 9 чисел")
            return
        R = np.array(raw).reshape(3, 3)
        print("\nВведённая матрица:")
        print(R)

        q = Quaternion.from_rotation_matrix(R)
        print(f"\nПолученный кватернион: {q}")
    except Exception as e:
        print(f"Ошибка: {e}")


def op_euler():
    print_separator("Кватернион → Углы Эйлера (Roll, Pitch, Yaw)")
    q = input_quaternion("Кватернион")

    roll, pitch, yaw = quaternion_to_euler(q)
    print(f"\nУглы Эйлера (радианы):")
    print(f"  Roll  (крен):     {roll:.6f} рад  ({np.degrees(roll):.2f}°)")
    print(f"  Pitch (тангаж):   {pitch:.6f} рад  ({np.degrees(pitch):.2f}°)")
    print(f"  Yaw   (рысканье): {yaw:.6f} рад  ({np.degrees(yaw):.2f}°)")


def op_yaw_only():
    print_separator("Создание кватерниона только по углу рысканья (yaw)")
    yaw_deg = float(input("Угол рысканья в градусах: "))
    q = quaternion_from_yaw(np.radians(yaw_deg))
    print(f"\nКватернион: {q}")
    print(f"Матрица поворота:\n{q.to_rotation_matrix()}")


def show_menu():
    print("\n" + "=" * 60)
    print("  РУЧНАЯ ПРОВЕРКА КВАТЕРНИОННЫХ ОПЕРАЦИЙ")
    print("=" * 60)
    print("1.  Создать кватернион")
    print("2.  Умножить два кватерниона (q1 * q2)")
    print("3.  Сопряжение / Нормализация / Инверсия")
    print("4.  Повернуть вектор кватернионом")
    print("5.  Создать кватернион из оси и угла")
    print("6.  Кватернион → Матрица поворота")
    print("7.  Матрица поворота → Кватернион")
    print("8.  Кватернион → Углы Эйлера (Roll/Pitch/Yaw)")
    print("9.  Создать кватернион по углу рысканья (yaw)")
    print("0.  Выход")
    print("-" * 60)


def main():
    while True:
        show_menu()
        choice = input("Выберите операцию (0-9): ").strip()

        if choice == "1":
            op_create()
        elif choice == "2":
            op_multiply()
        elif choice == "3":
            op_conjugate_inverse_normalize()
        elif choice == "4":
            op_rotate_vector()
        elif choice == "5":
            op_axis_angle()
        elif choice == "6":
            op_to_matrix()
        elif choice == "7":
            op_from_matrix()
        elif choice == "8":
            op_euler()
        elif choice == "9":
            op_yaw_only()
        elif choice == "0":
            print("Выход.")
            break
        else:
            print("Неверный выбор. Введите число от 0 до 9.")

        input("\nНажмите Enter для продолжения...")


if __name__ == "__main__":
    main()