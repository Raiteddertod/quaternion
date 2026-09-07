
"""
Улучшенная демонстрация контроллера ориентации.

Возможности:
- Заданный начальный наклон (roll, pitch, yaw)
- Симуляция восстановления горизонтальной ориентации
- Графики углов Эйлера (текущие и желаемые)
- Графики команд угловых скоростей
- Анимация ориентации в 3D (текущая СК vs желаемая СК)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from quaternion.quaternion import Quaternion
from quaternion.attitude_controller import compute_desired_rates
from quaternion.conversions import quaternion_to_euler


# ----------------------------------------------------------------------
# Интегрирование
# ----------------------------------------------------------------------
def integrate_quaternion(q: Quaternion, omega: np.ndarray, dt: float) -> Quaternion:
    """Интегрирование кинематики кватерниона (1-й порядок)."""
    q_omega = Quaternion(0.0, omega[0], omega[1], omega[2])
    dq = 0.5 * (q * q_omega)
    q_new = (q + dq * dt).normalize()
    return q_new


def draw_frame(ax, R: np.ndarray, color='blue', label_prefix='', alpha=1.0, linewidth=2.0):
    """Рисует систему координат по матрице поворота."""
    origin = np.zeros(3)
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']

    for i in range(3):
        vec = R[:, i]
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2],
                  color=colors[i], alpha=alpha, linewidth=linewidth,
                  arrow_length_ratio=0.18,
                  label=f"{label_prefix} {labels[i]}" if i == 0 else "")


# ----------------------------------------------------------------------
# Основная симуляция
# ----------------------------------------------------------------------
def run_simulation(initial_roll_deg=30.0, initial_pitch_deg=15.0, initial_yaw_deg=0.0,
                   a_des=np.array([0.0, 0.0, 9.81]), psi_ref=0.0,
                   p_rp=8.0, p_yaw=4.0, dt=0.01, T=5.0):
    """Запускает симуляцию и возвращает логи."""

    # Начальный кватернион
    q = Quaternion.from_axis_angle([1, 0, 0], np.radians(initial_roll_deg))
    q = q * Quaternion.from_axis_angle([0, 1, 0], np.radians(initial_pitch_deg))
    q = q * Quaternion.from_axis_angle([0, 0, 1], np.radians(initial_yaw_deg))
    q = q.normalize()

    # Логи
    logs = {
        'time': [],
        'roll': [], 'pitch': [], 'yaw': [],
        'roll_des': [], 'pitch_des': [], 'yaw_des': [],
        'p_des': [], 'q_des': [], 'r_des': [],
    }

    steps = int(T / dt)

    for i in range(steps):
        t = i * dt

        p_des, q_des, r_des, q_des_full = compute_desired_rates(
            q_hat=q, a_des=a_des, psi_ref=psi_ref, p_rp=p_rp, p_yaw=p_yaw
        )

        # Интегрируем
        omega = np.array([p_des, q_des, r_des])
        q = integrate_quaternion(q, omega, dt)

        # Текущие углы
        roll, pitch, yaw = quaternion_to_euler(q)
        roll_d, pitch_d, yaw_d = quaternion_to_euler(q_des_full)

        # Сохраняем
        logs['time'].append(t)
        logs['roll'].append(np.degrees(roll))
        logs['pitch'].append(np.degrees(pitch))
        logs['yaw'].append(np.degrees(yaw))
        logs['roll_des'].append(np.degrees(roll_d))
        logs['pitch_des'].append(np.degrees(pitch_d))
        logs['yaw_des'].append(np.degrees(yaw_d))
        logs['p_des'].append(p_des)
        logs['q_des'].append(q_des)
        logs['r_des'].append(r_des)

    return logs, q


# ----------------------------------------------------------------------
# Визуализация
# ----------------------------------------------------------------------
def plot_results(logs):
    """Строит статические графики."""
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    t = logs['time']

    # --- Углы Эйлера ---
    axs[0].plot(t, logs['roll'], label='Roll (текущий)', color='r', linewidth=2)
    axs[0].plot(t, logs['pitch'], label='Pitch (текущий)', color='g', linewidth=2)
    axs[0].plot(t, logs['yaw'], label='Yaw (текущий)', color='b', linewidth=2, linestyle='--')

    axs[0].plot(t, logs['roll_des'], label='Roll (желаемый)', color='r', linewidth=1.5, alpha=0.5, linestyle=':')
    axs[0].plot(t, logs['pitch_des'], label='Pitch (желаемый)', color='g', linewidth=1.5, alpha=0.5, linestyle=':')
    axs[0].plot(t, logs['yaw_des'], label='Yaw (желаемый)', color='b', linewidth=1.5, alpha=0.5, linestyle=':')

    axs[0].axhline(0, color='black', linestyle=':', linewidth=1)
    axs[0].set_ylabel('Угол, °')
    axs[0].set_title('Углы Эйлера (текущие и желаемые)')
    axs[0].legend(loc='upper right', fontsize=9)
    axs[0].grid(True)
    axs[0].set_ylim([-50, 50])

    # --- Команды угловых скоростей ---
    axs[1].plot(t, logs['p_des'], label='p_des (крен)', color='r', linewidth=2)
    axs[1].plot(t, logs['q_des'], label='q_des (тангаж)', color='g', linewidth=2)
    axs[1].plot(t, logs['r_des'], label='r_des (рысканье)', color='b', linewidth=2)

    axs[1].axhline(0, color='black', linestyle=':', linewidth=1)
    axs[1].set_xlabel('Время, с')
    axs[1].set_ylabel('Угловая скорость, рад/с')
    axs[1].set_title('Команды контроллера ориентации')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def animate_orientation(logs, initial_q, a_des, fps=30):
    """
    Анимация ориентации в 3D.
    Показывает текущую и желаемую системы координат.
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Подготовка данных
    times = logs['time']
    n_frames = len(times)

    def update(frame):
        ax.clear()

        t = times[frame]

        # Восстанавливаем текущий кватернион по углам
        roll = np.radians(logs['roll'][frame])
        pitch = np.radians(logs['pitch'][frame])
        yaw = np.radians(logs['yaw'][frame])
        q_current = Quaternion.from_axis_angle([1,0,0], roll) * \
                    Quaternion.from_axis_angle([0,1,0], pitch) * \
                    Quaternion.from_axis_angle([0,0,1], yaw)

        # Желаемый кватернион
        roll_d = np.radians(logs['roll_des'][frame])
        pitch_d = np.radians(logs['pitch_des'][frame])
        yaw_d = np.radians(logs['yaw_des'][frame])
        q_des = Quaternion.from_axis_angle([1,0,0], roll_d) * \
                Quaternion.from_axis_angle([0,1,0], pitch_d) * \
                Quaternion.from_axis_angle([0,0,1], yaw_d)

        R_current = q_current.to_rotation_matrix()
        R_des = q_des.to_rotation_matrix()

        # Рисуем системы координат
        draw_frame(ax, np.eye(3), color='gray', label_prefix='World', alpha=0.4, linewidth=1.5)
        draw_frame(ax, R_current, color='blue', label_prefix='Body (current)', alpha=0.95, linewidth=2.5)
        draw_frame(ax, R_des, color='green', label_prefix='Body (desired)', alpha=0.7, linewidth=2.0, linestyle='--')

        # Желаемое направление тяги (e_z_des)
        ax.quiver(0, 0, 0, 0, 0, 1.2, color='orange', linewidth=2, arrow_length_ratio=0.1, label='a_des direction')

        # Настройки вида
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        p = logs['p_des'][frame]
        q = logs['q_des'][frame]
        r = logs['r_des'][frame]

        ax.set_title(
            f"t = {t:.2f} s\n"
            f"Current:  Roll={logs['roll'][frame]:6.1f}°  Pitch={logs['pitch'][frame]:6.1f}°  Yaw={logs['yaw'][frame]:6.1f}°\n"
            f"Desired:  Roll={logs['roll_des'][frame]:6.1f}°  Pitch={logs['pitch_des'][frame]:6.1f}°  Yaw={logs['yaw_des'][frame]:6.1f}°\n"
            f"Commands: p={p:+6.2f}  q={q:+6.2f}  r={r:+6.2f} rad/s"
        )

        ax.legend(loc='upper left', fontsize=8)
        ax.view_init(elev=25, azim=45 + frame * 0.3)   # лёгкое вращение камеры

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, repeat=True)
    plt.tight_layout()
    plt.show()

    return ani


# ----------------------------------------------------------------------
# Главная функция
# ----------------------------------------------------------------------
def main():
    # ==================== ПАРАМЕТРЫ ====================
    initial_roll_deg  = 35.0
    initial_pitch_deg = 20.0
    initial_yaw_deg   = 45.0          # <-- добавлено

    a_des = np.array([0.0, 0.0, 9.81])
    psi_ref = np.radians(0.0)

    p_rp = 7.0
    p_yaw = 3.5

    dt = 0.01
    T = 6.0
    # ===================================================

    print(f"Начальные углы: Roll={initial_roll_deg}°, Pitch={initial_pitch_deg}°, Yaw={initial_yaw_deg}°")

    logs, final_q = run_simulation(
        initial_roll_deg=initial_roll_deg,
        initial_pitch_deg=initial_pitch_deg,
        initial_yaw_deg=initial_yaw_deg,
        a_des=a_des,
        psi_ref=psi_ref,
        p_rp=p_rp,
        p_yaw=p_yaw,
        dt=dt,
        T=T
    )

    # Финальное состояние
    final_roll, final_pitch, final_yaw = quaternion_to_euler(final_q)
    print(f"\nФинальная ориентация:")
    print(f"  Roll  = {np.degrees(final_roll):.2f}°")
    print(f"  Pitch = {np.degrees(final_pitch):.2f}°")
    print(f"  Yaw   = {np.degrees(final_yaw):.2f}°")

    # Статические графики
    plot_results(logs)

    # Анимация
    print("\nЗапуск анимации...")
    animate_orientation(logs, final_q, a_des, fps=25)


if __name__ == "__main__":
    main()