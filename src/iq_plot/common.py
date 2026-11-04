from pathlib import Path

import numpy as np
from qutip import basis, destroy


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


def create_output_path(relative_path) -> str:
    fullpath = OUTPUT_DIR / relative_path
    fullpath.parent.mkdir(parents=True, exist_ok=True)
    return str(fullpath)


# --- 読み出し共振器のパラメータ ---
n_fock = 32
f_r = 1.0                      # 固有周波数 [GHz]
omega_r = 2 * np.pi * f_r      # 固有角周波数 [rad/ns]

# --- 演算子 ---
a_r = destroy(n_fock)          # 消滅演算子
n_r = a_r.dag() * a_r          # 光子数演算子
x_r = a_r + a_r.dag()          # 駆動に結合する演算子

# --- 初期状態 ---
psi_r0 = basis(n_fock, 0)      # 共振器の真空状態

# --- ハミルトニアン ---
h_r = omega_r * n_r            # 共振器の自由ハミルトニアン

# --- マイクロ波駆動 ---
omega = 0.12 * omega_r         # 駆動強度 [rad/ns]

# --- 共振器の減衰 ---
f_kappa = 0.06                  # κ/(2π) [GHz]
kappa = 2 * np.pi * f_kappa    # 光子数の減衰率 [1/ns]
