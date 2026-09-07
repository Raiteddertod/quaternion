# examples/basic_scenarios.py
"""
Демонстрация восстановления горизонтальной ориентации.

Сценарий:
- ТТ начинает с наклонённой ориентации
- Контроллер выдаёт угловые скорости, чтобы выровнять аппарат
- a_des = [0, 0, 9.81] (висение)
- Строятся графики углов и команд скоростей
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from quaternion.quaternion import Quaternion
from quaternion.attitude_controller import compute_desired_rates
from quaternion.conversions import quaternion_to_euler


def integrate_quaternion(q: Quaternion, omega: np.ndarray, dt: float) -> Quaternion:
    """
    Интегрирование кинематики кватерниона.
    \dot{q} = 1/2 * q ⊗ [0, ω]
    """
    q_omega = Quaternion(0.0, omega[0], omega[1], omega[2])
    dq = 0.5 * (q * q_omega)          
    q_new = (q + dq * dt).normalize()  # используем __add__ и скалярное умножение
    return q_new


def main():
    # ==================== НАСТРОЙКИ ====================
    initial_roll_deg = -90.0
    initial_pitch_deg = 180.0

    a_des = np.array([0.0, 0.0, 9.81])
    psi_ref = 0.0

    p_rp = 10.0
    p_yaw = 8.0

    dt = 0.01
    T = 5.0
    # ===================================================

    # Начальный кватернион
    q = Quaternion.from_axis_angle([1, 0, 0], np.radians(initial_roll_deg))
    
    print(f"Начальный кватернион {q}")
    q = (q * Quaternion.from_axis_angle([0, 1, 0], np.radians(initial_pitch_deg))).normalize()
    print(f"Нормализация {q}")

    print(f"Начальный наклон: Roll={initial_roll_deg}°, Pitch={initial_pitch_deg}°\n")

    time_log, roll_log, pitch_log, yaw_log = [], [], [], []
    p_log, q_log, r_log = [], [], []

    for i in range(int(T / dt)):
        p_des, q_des, r_des, _ = compute_desired_rates(q, a_des, psi_ref, p_rp, p_yaw)

        omega = np.array([p_des, q_des, r_des])
        q = integrate_quaternion(q, omega, dt)

        roll, pitch, yaw = quaternion_to_euler(q)

        time_log.append(i * dt)
        roll_log.append(np.degrees(roll))
        pitch_log.append(np.degrees(pitch))
        yaw_log.append(np.degrees(yaw))
        p_log.append(p_des)
        q_log.append(q_des)
        r_log.append(r_des)

    # ==================== ГРАФИКИ ====================
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axs[0].plot(time_log, roll_log, label='Roll')
    axs[0].plot(time_log, pitch_log, label='Pitch')
    axs[0].plot(time_log, yaw_log, label='Yaw', linestyle='--')
    axs[0].axhline(0, color='k', linestyle=':', linewidth=1)
    axs[0].set_ylabel('Угол, °')
    axs[0].set_title('Восстановление горизонтальной ориентации')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time_log, p_log, label='p_des')
    axs[1].plot(time_log, q_log, label='q_des')
    axs[1].plot(time_log, r_log, label='r_des')
    axs[1].axhline(0, color='k', linestyle=':', linewidth=1)
    axs[1].set_xlabel('Время, с')
    axs[1].set_ylabel('Угловая скорость, рад/с')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()