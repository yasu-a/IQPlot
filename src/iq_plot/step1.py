from .common import *
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, destroy, sesolve, mesolve, tensor
from pprint import pprint
from tqdm.notebook import tqdm


# --- シミュレーション条件 ---
drive_frequencies = [0.8, 0.9, 0.99, 1.0, 1.01, 1.1, 1.2]  # 駆動周波数 [GHz]
times = np.linspace(0, 10, 1001)    # 時間 [ns]

# --- 時間依存ハミルトニアン ---


def drive_coefficient(t, args):
    """マイクロ波駆動の時間依存係数を返す。"""
    return np.cos(args["omega_d"] * t)


h_t = [h_r, [omega * x_r, drive_coefficient]]

# --- 時間発展 ---
def main() -> None:
    results = {}

    for f_d in drive_frequencies:
        result = sesolve(
            h_t,
            psi_r0,
            times,
            e_ops=[n_r],
            args={"omega_d": 2 * np.pi * f_d},
            options={
                "rtol": 1e-9,
                "atol": 1e-11,
                "nsteps": 10000,
                "max_step": 0.02,
            },
        )

        results[f_d] = result.expect[0]

    # --- プロット ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for f_d in drive_frequencies:
        ax.plot(
            times,
            results[f_d],
            label=fr"$f_d = {f_d:.2f}$ GHz",
        )

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Mean photon number [photons]")
    ax.set_title("Driven readout resonator (no damping)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(create_output_path("step1-timeevol.png"))


if __name__ == "__main__":
    main()
