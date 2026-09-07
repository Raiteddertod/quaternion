# examples/visualize_quaternion_rotation.py
"""
Интерактивный визуализатор поворота кватернионом.

Возможности:
- Ввод и изменение кватерниона через ползунки и текстовые поля (w, x, y, z)
- Ввод вектора V
- Визуализация мировой и связанной системы координат
- Показ поворота вектора V → V'
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from quaternion.quaternion import Quaternion


class QuaternionVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Визуализатор поворота кватернионом")
        self.root.geometry("1100x750")

        # Текущие значения
        self.q_w = tk.DoubleVar(value=1.0)
        self.q_x = tk.DoubleVar(value=0.0)
        self.q_y = tk.DoubleVar(value=0.0)
        self.q_z = tk.DoubleVar(value=0.0)

        self.v_x = tk.DoubleVar(value=1.0)
        self.v_y = tk.DoubleVar(value=0.0)
        self.v_z = tk.DoubleVar(value=0.0)

        self._create_widgets()
        self.update_visualization()

    def _create_widgets(self):
        # === Левая панель управления ===
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # --- КВАТЕРНИОН ---
        ttk.Label(control_frame, text="КВАТЕРНИОН (scalar-first)", font=("Arial", 11, "bold")).pack(pady=(0, 8))
        self._create_quaternion_controls(control_frame)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=12)

        # --- ВЕКТОР V ---
        ttk.Label(control_frame, text="ВЕКТОР V (в мировой СК)", font=("Arial", 11, "bold")).pack(pady=(0, 8))
        self._create_vector_controls(control_frame)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=12)

        # --- Кнопки ---
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Нормализовать q", command=self.normalize_quaternion).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Сбросить", command=self.reset).pack(side=tk.LEFT, padx=3)

        # --- Информация ---
        self.info_label = ttk.Label(control_frame, text="", justify=tk.LEFT, font=("Consolas", 9))
        self.info_label.pack(pady=10, anchor='w')

        # === Правая часть — 3D график ===
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 7), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Обновление при изменении значений
        for var in [self.q_w, self.q_x, self.q_y, self.q_z, self.v_x, self.v_y, self.v_z]:
            var.trace_add("write", lambda *args: self.update_visualization())

    def _create_quaternion_controls(self, parent):
        params = [
            ("w", self.q_w, -1.0, 1.0),
            ("x", self.q_x, -1.0, 1.0),
            ("y", self.q_y, -1.0, 1.0),
            ("z", self.q_z, -1.0, 1.0),
        ]
        for label, var, vmin, vmax in params:
            frame = ttk.Frame(parent)
            frame.pack(fill='x', pady=2)
            ttk.Label(frame, text=f"{label}:", width=3).pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=var, width=8).pack(side=tk.LEFT, padx=3)
            ttk.Scale(frame, from_=vmin, to=vmax, variable=var,
                      orient=tk.HORIZONTAL, length=200,
                      command=lambda v: self.update_visualization()).pack(side=tk.LEFT, padx=3, fill='x', expand=True)

    def _create_vector_controls(self, parent):
        for label, var in [("x", self.v_x), ("y", self.v_y), ("z", self.v_z)]:
            frame = ttk.Frame(parent)
            frame.pack(fill='x', pady=2)
            ttk.Label(frame, text=f"{label}:", width=3).pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=var, width=10).pack(side=tk.LEFT, padx=3)
            ttk.Button(frame, text="0", width=2, command=lambda v=var: v.set(0)).pack(side=tk.LEFT, padx=1)
            ttk.Button(frame, text="1", width=2, command=lambda v=var: v.set(1)).pack(side=tk.LEFT, padx=1)

    # ------------------------------------------------------------------
    # Основная логика
    # ------------------------------------------------------------------
    def get_current_quaternion(self) -> Quaternion:
        q = Quaternion(self.q_w.get(), self.q_x.get(), self.q_y.get(), self.q_z.get())
        return q.normalize()

    def get_current_vector(self) -> np.ndarray:
        return np.array([self.v_x.get(), self.v_y.get(), self.v_z.get()])

    def update_visualization(self):
        self.ax.clear()

        q = self.get_current_quaternion()
        v = self.get_current_vector()

        # Мировая СК (серая)
        self._draw_frame(self.ax, np.eye(3), color='gray', label_prefix='World', alpha=0.35, linewidth=1.5)

        # Связанная СК (синяя)
        R = q.to_rotation_matrix()
        self._draw_frame(self.ax, R, color='blue', label_prefix='Body', alpha=0.95, linewidth=2.2)

        # Вектор V
        if np.linalg.norm(v) > 1e-6:
            self._draw_vector(self.ax, v, color='green', label='V (original)', linewidth=2.5)

            # Повёрнутый вектор
            v_rot = q.rotate(v)
            self._draw_vector(self.ax, v_rot, color='red', label="V' (rotated)", linewidth=2.5, linestyle='--')

            # Дуга поворота
            self._draw_rotation_arc(self.ax, v, v_rot)

        # Настройка вида
        self.ax.set_xlim([-1.6, 1.6])
        self.ax.set_ylim([-1.6, 1.6])
        self.ax.set_zlim([-1.6, 1.6])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"Поворот вектора кватернионом\nq = [{q.w:.4f}, {q.x:.4f}, {q.y:.4f}, {q.z:.4f}]")
        self.ax.legend(loc='upper left')

        self._update_info(q, v)
        self.canvas.draw()

    def _draw_frame(self, ax, R, color='blue', label_prefix='', alpha=1.0, linewidth=2):
        origin = np.zeros(3)
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            vec = R[:, i]
            ax.quiver(origin[0], origin[1], origin[2],
                      vec[0], vec[1], vec[2],
                      color=colors[i], alpha=alpha, linewidth=linewidth,
                      arrow_length_ratio=0.18, label=f"{label_prefix} {labels[i]}" if i == 0 else "")

    def _draw_vector(self, ax, v, color='green', label='', linewidth=2.5, linestyle='-'):
        ax.quiver(0, 0, 0, v[0], v[1], v[2],
                  color=color, linewidth=linewidth, arrow_length_ratio=0.12,
                  linestyle=linestyle, label=label)

    def _draw_rotation_arc(self, ax, v1, v2, steps=25):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return
        v1n = v1 / n1
        v2n = v2 / n2
        for t in np.linspace(0, 1, steps):
            v = (1 - t) * v1n + t * v2n
            nv = np.linalg.norm(v)
            if nv > 1e-9:
                v = v / nv
                ax.scatter(v[0], v[1], v[2], color='orange', s=6, alpha=0.6)

    def _update_info(self, q: Quaternion, v: np.ndarray):
        v_rot = q.rotate(v)
        nv = np.linalg.norm(v)
        nvr = np.linalg.norm(v_rot)
        dot = np.dot(v, v_rot)
        angle = np.arccos(np.clip(dot / (nv * nvr + 1e-12), -1.0, 1.0))

        info = (
            f"Кватернион (нормализованный):\n"
            f"  w = {q.w:.6f}\n"
            f"  x = {q.x:.6f}\n"
            f"  y = {q.y:.6f}\n"
            f"  z = {q.z:.6f}\n\n"
            f"V     = [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]\n"
            f"V'    = [{v_rot[0]:.3f}, {v_rot[1]:.3f}, {v_rot[2]:.3f}]\n"
            f"Угол поворота ≈ {np.degrees(angle):.2f}°"
        )
        self.info_label.config(text=info)

    def normalize_quaternion(self):
        q = self.get_current_quaternion()
        self.q_w.set(round(q.w, 6))
        self.q_x.set(round(q.x, 6))
        self.q_y.set(round(q.y, 6))
        self.q_z.set(round(q.z, 6))

    def reset(self):
        self.q_w.set(1.0)
        self.q_x.set(0.0)
        self.q_y.set(0.0)
        self.q_z.set(0.0)
        self.v_x.set(1.0)
        self.v_y.set(0.0)
        self.v_z.set(0.0)


def main():
    root = tk.Tk()
    app = QuaternionVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()