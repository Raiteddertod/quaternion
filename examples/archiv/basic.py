# examples/basic_scenarios.py
"""
Демонстрация работы контроллера ориентации (по статье Faessler et al., ICRA 2015).

Выходы контроллера:
    p_des, q_des — желаемые угловые скорости по крену и тангажу в СВЯЗАННОЙ СК (ф. 15)
    r_des        — желаемая угловая скорость по рысканью в СВЯЗАННОЙ СК (ф. 20)
    G_des        — полная желаемая ориентация (опционально)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# === Добавь эти импорты в начало файла ===
from matplotlib.widgets import Button
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from quaternion.quaternion import Quaternion
from quaternion.attitude_controller import compute_desired_rates
from quaternion.conversions import quaternion_to_euler


def integrate_quaternion(q: Quaternion, omega: np.ndarray, dt: float) -> Quaternion:
    """Интегрирование кинематики кватерниона."""
    q_omega = Quaternion(0.0, omega[0], omega[1], omega[2])
    dq = 0.5 * (q * q_omega)
    q_new = (q + dq * dt).normalize()
    return q_new


def draw_frame(ax, R, color, label, alpha=1.0, linewidth=2.0, linestyle='-'):
    origin = np.zeros(3)
    colors = ['r', 'g', 'b']
    for i, axis in enumerate(['X', 'Y', 'Z']):
        ax.quiver(*origin, *R[:, i], color=colors[i], alpha=alpha,
                  linewidth=linewidth, arrow_length_ratio=0.15, linestyle=linestyle)


def run_simulation(initial_roll=30.0, initial_pitch=15.0, initial_yaw=0.0,
                   a_des=np.array([0., 0., 9.81]), psi_ref=0.0,
                   p_rp=8.0, p_yaw=4.0, dt=0.01, T=6.0):

    # Начальная ориентация
    q = (Quaternion.from_axis_angle([1, 0, 0], np.radians(initial_roll)) *
         Quaternion.from_axis_angle([0, 1, 0], np.radians(initial_pitch)) *
         Quaternion.from_axis_angle([0, 0, 1], np.radians(initial_yaw))).normalize()

    logs = {
        'time': [],
        'roll': [], 'pitch': [], 'yaw': [],
        'G_des_roll': [], 'G_des_pitch': [], 'G_des_yaw': [],
        'p_des': [], 'q_des': [], 'r_des': [],
    }

    for i in range(int(T / dt)):
        t = i * dt

        p_des, q_des, r_des, G_des = compute_desired_rates(
            q_hat=q, a_des=a_des, psi_ref=psi_ref, p_rp=p_rp, p_yaw=p_yaw
        )

        omega = np.array([p_des, q_des, r_des])
        q = integrate_quaternion(q, omega, dt)

        roll, pitch, yaw = quaternion_to_euler(q)
        g_roll, g_pitch, g_yaw = quaternion_to_euler(G_des)

        logs['time'].append(t)
        logs['roll'].append(np.degrees(roll))
        logs['pitch'].append(np.degrees(pitch))
        logs['yaw'].append(np.degrees(yaw))
        logs['G_des_roll'].append(np.degrees(g_roll))
        logs['G_des_pitch'].append(np.degrees(g_pitch))
        logs['G_des_yaw'].append(np.degrees(g_yaw))
        logs['p_des'].append(p_des)
        logs['q_des'].append(q_des)
        logs['r_des'].append(r_des)

    return logs, q


def plot_results(logs):
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    t = logs['time']

    # Углы Эйлера
    axs[0].plot(t, logs['roll'], 'r-', linewidth=2, label='Roll (текущий)')
    axs[0].plot(t, logs['pitch'], 'g-', linewidth=2, label='Pitch (текущий)')
    axs[0].plot(t, logs['yaw'], 'b--', linewidth=2, label='Yaw (текущий)')

    axs[0].plot(t, logs['G_des_roll'], 'r:', linewidth=1.5, alpha=0.7, label='Roll (G_des)')
    axs[0].plot(t, logs['G_des_pitch'], 'g:', linewidth=1.5, alpha=0.7, label='Pitch (G_des)')
    axs[0].plot(t, logs['G_des_yaw'], 'b:', linewidth=1.5, alpha=0.7, label='Yaw (G_des)')

    axs[0].axhline(0, color='k', linestyle=':', linewidth=1)
    axs[0].set_ylabel('Угол, °')
    axs[0].set_title('Углы Эйлера: текущая ориентация vs G_des')
    axs[0].legend(loc='upper right', fontsize=9)
    axs[0].grid(True)
    axs[0].set_ylim([-55, 55])

    # Команды угловых скоростей (в связанной СК)
    axs[1].plot(t, logs['p_des'], 'r-', linewidth=2, label='p_des (крен, связанная СК)')
    axs[1].plot(t, logs['q_des'], 'g-', linewidth=2, label='q_des (тангаж, связанная СК)')
    axs[1].plot(t, logs['r_des'], 'b-', linewidth=2, label='r_des (рысканье, связанная СК)')

    axs[1].axhline(0, color='k', linestyle=':', linewidth=1)
    axs[1].set_xlabel('Время, с')
    axs[1].set_ylabel('Угловая скорость, рад/с')
    axs[1].set_title('Выходы контроллера: p_des, q_des, r_des (в связанной системе координат)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def animate_orientation_with_controls(logs, fps=25):
    """
    Анимация с элементами управления:
    - Play / Pause
    - Reverse (воспроизведение назад)
    - Restart (с начала)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # === Состояние анимации ===
    n_frames = len(logs['time'])
    frame_idx = [0]                    # используем список, чтобы менять внутри функций
    direction = [1]                    # 1 = вперёд, -1 = назад
    playing = [False]
    interval = 1000 / fps

    # === Создаём таймер ===
    timer = fig.canvas.new_timer(interval=int(interval))

    def get_quaternions(frame):
        """Получить текущий и желаемый кватернионы по номеру кадра"""
        q_cur = (Quaternion.from_axis_angle([1, 0, 0], np.radians(logs['roll'][frame])) *
                 Quaternion.from_axis_angle([0, 1, 0], np.radians(logs['pitch'][frame])) *
                 Quaternion.from_axis_angle([0, 0, 1], np.radians(logs['yaw'][frame])))

        q_des = (Quaternion.from_axis_angle([1, 0, 0], np.radians(logs['G_des_roll'][frame])) *
                 Quaternion.from_axis_angle([0, 1, 0], np.radians(logs['G_des_pitch'][frame])) *
                 Quaternion.from_axis_angle([0, 0, 1], np.radians(logs['G_des_yaw'][frame])))
        return q_cur, q_des

    def update_display(frame):
        """Перерисовывает сцену для текущего кадра"""
        ax.clear()

        q_cur, q_des = get_quaternions(frame)

        R_cur = q_cur.to_rotation_matrix()
        R_des = q_des.to_rotation_matrix()

        # Мировая СК
        draw_frame(ax, np.eye(3), color='gray', label='World', alpha=0.35, linewidth=1.2)
        # Текущая ориентация
        draw_frame(ax, R_cur, color='blue', label='Текущая СК тела', alpha=0.95, linewidth=2.5)
        # Желаемая ориентация G_des
        draw_frame(ax, R_des, color='green', label='G_des (желаемая)', alpha=0.65, linewidth=2.0)

        ax.set_xlim([-1.6, 1.6])
        ax.set_ylim([-1.6, 1.6])
        ax.set_zlim([-1.6, 1.6])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title(
            f"t = {logs['time'][frame]:.2f} с     |     Кадр: {frame}/{n_frames-1}\n\n"
            f"Текущая:   Roll={logs['roll'][frame]:6.1f}°   Pitch={logs['pitch'][frame]:6.1f}°   Yaw={logs['yaw'][frame]:6.1f}°\n"
            f"G_des:     Roll={logs['G_des_roll'][frame]:6.1f}°   Pitch={logs['G_des_pitch'][frame]:6.1f}°   Yaw={logs['G_des_yaw'][frame]:6.1f}°\n\n"
            f"Команды (связанная СК):\n"
            f"p_des = {logs['p_des'][frame]:+6.2f}   q_des = {logs['q_des'][frame]:+6.2f}   r_des = {logs['r_des'][frame]:+6.2f}  рад/с"
        )
        ax.legend(loc='upper left', fontsize=8)
        ax.view_init(elev=22, azim=45)

        fig.canvas.draw_idle()

    def step_forward():
        """Один шаг вперёд/назад"""
        frame_idx[0] += direction[0]

        # Циклическое воспроизведение
        if frame_idx[0] >= n_frames:
            frame_idx[0] = 0
        elif frame_idx[0] < 0:
            frame_idx[0] = n_frames - 1

        update_display(frame_idx[0])

    def _on_timer():
        if playing[0]:
            step_forward()

    timer.add_callback(_on_timer)

    # === Кнопки управления ===
    plt.subplots_adjust(bottom=0.18)

    # Кнопка Play / Pause
    ax_play = fig.add_axes([0.25, 0.05, 0.15, 0.06])
    btn_play = Button(ax_play, '▶ Play', color='lightgreen')

    def toggle_play(event):
        playing[0] = not playing[0]
        if playing[0]:
            btn_play.label.set_text('⏸ Pause')
            btn_play.color = 'lightyellow'
            timer.start()
        else:
            btn_play.label.set_text('▶ Play')
            btn_play.color = 'lightgreen'
            timer.stop()
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    # Кнопка Reverse
    ax_reverse = fig.add_axes([0.42, 0.05, 0.15, 0.06])
    btn_reverse = Button(ax_reverse, '⏪ Reverse')

    def toggle_reverse(event):
        direction[0] *= -1
        if direction[0] == 1:
            btn_reverse.label.set_text('⏪ Reverse')
        else:
            btn_reverse.label.set_text('⏩ Forward')
        fig.canvas.draw_idle()

    btn_reverse.on_clicked(toggle_reverse)

    # Кнопка Restart
    ax_restart = fig.add_axes([0.59, 0.05, 0.15, 0.06])
    btn_restart = Button(ax_restart, '⟲ Restart')

    def restart(event):
        frame_idx[0] = 0
        direction[0] = 1
        btn_reverse.label.set_text('⏪ Reverse')
        playing[0] = True
        btn_play.label.set_text('⏸ Pause')
        btn_play.color = 'lightyellow'
        update_display(0)
        timer.start()

    btn_restart.on_clicked(restart)

    # Первая отрисовка
    update_display(0)

    plt.show()

    return timer, btn_play, btn_reverse, btn_restart

def main():
    # ==================== Параметры ====================
    initial_roll  = 90.
    initial_pitch = 45.0
    initial_yaw   = 0.0          # теперь поддерживается

    a_des = np.array([0.0, 0.0, 9.81])
    psi_ref = 0.0

    p_rp = 7.5
    p_yaw = 3.5
    dt = 0.01
    T = 6.0
    # ===================================================

    print(f"Начальные углы: Roll={initial_roll}°, Pitch={initial_pitch}°, Yaw={initial_yaw}°")

    logs, _ = run_simulation(
        initial_roll=initial_roll,
        initial_pitch=initial_pitch,
        initial_yaw=initial_yaw,
        a_des=a_des,
        psi_ref=psi_ref,
        p_rp=p_rp,
        p_yaw=p_yaw,
        dt=dt,
        T=T
    )

    plot_results(logs)
    print("\nЗапуск анимации с управлением...")
    animate_orientation_with_controls(logs, fps=25)


if __name__ == "__main__":
    main()