# quaternion/attitude_controller.py
"""
Контроллер желаемых угловых скоростей (attitude rate controller).

Реализация контроллера ориентации по статье:
"Automatic Re-Initialization and Failure Recovery for Aggressive Flight 
with a Monocular Vision-Based Quadrotor" (Faessler et al., ICRA 2015)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from .quaternion import Quaternion
from .utils import (
    normalize_vector,
    safe_cross_normalized,
    angle_between_vectors,
    get_signed_rate_from_error,
    handle_zero_acceleration,
    is_singularity_case,
    EPSILON,
)
from .conversions import quaternion_from_axes


def compute_desired_rates(
    q_hat: Quaternion,
    a_des: np.ndarray,
    psi_ref: float,
    p_rp: float = 10.0,
    p_yaw: float = 5.0,
) -> Tuple[float, float, float, Quaternion]:
    """
    Вычисляет желаемые угловые скорости тела (p_des, q_des, r_des).

    Параметры:
        q_hat   : текущий кватернион ориентации (scalar-first)
        a_des   : желаемое ускорение в МИРОВОЙ системе координат [ax, ay, az]
        psi_ref : желаемое рысканье (heading), рад
        p_rp    : коэффициент усиления для roll/pitch
        p_yaw   : коэффициент усиления для yaw

    Возвращает:
        p_des, q_des, r_des : желаемые угловые скорости (рад/с)
        q_des_full          : полная желаемая ориентация
    """
    q_hat = q_hat.normalize()
    a_des = np.asarray(a_des, dtype=float)

    # ============================================================
    # ЧАСТЬ 1: Крен и тангаж (Roll & Pitch) — уравнения 10–15
    # ============================================================
    e_z_B = q_hat.rotate(np.array([0.0, 0.0, 1.0]))
    e_z_des_B, _ = handle_zero_acceleration(a_des, eps=EPSILON)

    alpha = angle_between_vectors(e_z_B, e_z_des_B, eps=EPSILON)
    n, cross_norm = _compute_rotation_axis(e_z_B, e_z_des_B)

    if is_singularity_case(cross_norm):
        q_e_rp = Quaternion.identity()
        p_des = 0.0
        q_des = 0.0
    else:
        B_n = q_hat.conjugate().rotate(n)
        half_alpha = alpha / 2.0
        q_e_rp = Quaternion(
            w=np.cos(half_alpha),
            x=B_n[0] * np.sin(half_alpha),
            y=B_n[1] * np.sin(half_alpha),
            z=B_n[2] * np.sin(half_alpha),
        )
        p_des = get_signed_rate_from_error(q_e_rp.w, q_e_rp.x, p_rp)
        q_des = get_signed_rate_from_error(q_e_rp.w, q_e_rp.y, p_rp)

    # ============================================================
    # ЧАСТЬ 2: Рысканье (Yaw) — уравнения 16–20
    # ============================================================
    e_x_C = np.array([np.cos(psi_ref), np.sin(psi_ref), 0.0])
    e_y_C = np.array([-np.sin(psi_ref), np.cos(psi_ref), 0.0])

    e_x_des_B = safe_cross_normalized(e_y_C, e_z_des_B, eps=EPSILON)
    cross_norm_yz = np.linalg.norm(np.cross(e_y_C, e_z_des_B))

    if is_singularity_case(cross_norm_yz):
        # Невозможно определить yaw (особый случай)
        r_des = 0.0
        q_des_full = _build_q_des_from_z_only(e_z_des_B)
    else:
        # Если желаемое ускорение направлено вниз — инвертируем ось X
        if e_z_des_B[2] < 0.0:
            e_x_des_B = -e_x_des_B

        e_y_des_B = safe_cross_normalized(e_z_des_B, e_x_des_B, eps=EPSILON)
        q_des_full = quaternion_from_axes(e_x_des_B, e_y_des_B, e_z_des_B)

        # Уравнение 20: ошибка по рысканью
        q_e_y = (q_hat * q_e_rp).inverse() * q_des_full
        r_des = get_signed_rate_from_error(q_e_y.w, q_e_y.z, p_yaw)

    return p_des, q_des, r_des, q_des_full.normalize()


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------

def _compute_rotation_axis(
    e_z_B: np.ndarray, e_z_des_B: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Вычисляет ось вращения n = normalize(e_z_B × e_z_des_B)."""
    cross = np.cross(e_z_B, e_z_des_B)
    norm = np.linalg.norm(cross)
    if norm < EPSILON:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return cross / norm, norm


def _build_q_des_from_z_only(e_z_des_B: np.ndarray) -> Quaternion:
    """Строит q_des, когда yaw не определён."""
    if abs(e_z_des_B[2]) < 0.9:
        e_x = np.cross(e_z_des_B, np.array([0.0, 0.0, 1.0]))
    else:
        e_x = np.cross(e_z_des_B, np.array([1.0, 0.0, 0.0]))

    e_x = normalize_vector(e_x)
    e_y = normalize_vector(np.cross(e_z_des_B, e_x))
    return quaternion_from_axes(e_x, e_y, e_z_des_B)