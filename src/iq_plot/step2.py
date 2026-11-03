from .common import *
from .step1 import drive_frequencies, h_t, drive_coefficient
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, destroy, sesolve, mesolve, tensor
from pprint import pprint
from tqdm import tqdm


# --- シミュレーション条件 ---
def main() -> None:
    drive_frequencies_step2 = np.linspace(
        min(drive_frequencies), max(drive_frequencies), 11)
    print(drive_frequencies_step2)
    times_step2 = np.linspace(0, 12 / kappa, 1201)  # 時間 [ns]
    print(times_step2)

    # --- 減衰と打ち切り確認用の演算子 ---
    c_ops = [np.sqrt(kappa) * a_r]
    p_high = sum(
        (basis(n_fock, n).proj() for n in range(n_fock - 4, n_fock)),
        0 * n_r,
    )

    # --- 時間発展 ---
    results_step2 = {}
    high_population_step2 = {}

    for f_d in tqdm(drive_frequencies_step2):
        result = mesolve(
            h_t,
            psi_r0,
            times_step2,
            c_ops=c_ops,
            e_ops=[n_r, p_high],
            args={"omega_d": 2 * np.pi * f_d},
            options={
                "rtol": 1e-9,
                "atol": 1e-11,
                "nsteps": 10000,
                "max_step": 0.02,
            },
        )

        results_step2[f_d] = result.expect[0]
        high_population_step2[f_d] = np.max(result.expect[1])

    # --- 共振付近の収束確認 ---
    f_res = min(drive_frequencies_step2, key=lambda f: abs(f - f_r))
    n_res = results_step2[f_res]

    period = 1 / f_res
    t_end = times_step2[-1]

    last_period = times_step2 > t_end - period
    previous_period = (
        (times_step2 > t_end - 2 * period) & ~last_period
    )

    n_last = np.mean(n_res[last_period])
    n_previous = np.mean(n_res[previous_period])
    relative_change = abs(n_last - n_previous) / max(n_last, 1e-12)

    # --- Hilbert空間の打ち切り確認 ---
    n_fock_check = n_fock + 16
    a_check = destroy(n_fock_check)
    n_check = a_check.dag() * a_check

    h_check = [
        omega_r * n_check,
        [omega * (a_check + a_check.dag()), drive_coefficient],
    ]

    result_check = mesolve(
        h_check,
        basis(n_fock_check, 0),
        [0, t_end],
        c_ops=[np.sqrt(kappa) * a_check],
        e_ops=[n_check],
        args={"omega_d": 2 * np.pi * f_res},
        options={
            "rtol": 1e-9,
            "atol": 1e-11,
            "nsteps": 10000,
            "max_step": 0.02,
        },
    )

    dimension_difference = (
        abs(result_check.expect[0][-1] - n_res[-1])
        / max(abs(n_res[-1]), 1e-12)
    )

    print(f"共振付近: f_d = {f_res:.3f} GHz")
    print(f"最後の2周期の平均光子数の相対差: {relative_change:.3%}")
    print(f"最高4準位の占有確率の最大値: {max(high_population_step2.values()):.2e}")
    print(f"Fock空間次元の変更による相対差: {dimension_difference:.2e}")

    if relative_change > 0.01:
        print("注意: 定常応答への収束を確認してください。")
    if dimension_difference > 1e-3:
        print("注意: Fock空間の次元を増やしてください。")

    # --- プロット ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for f_d in drive_frequencies_step2:
        ax.plot(
            times_step2,
            results_step2[f_d],
            label=fr"$f_d = {f_d:.2f}$ GHz",
        )

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Mean photon number [photons]")
    ax.set_title("Driven readout resonator (with damping)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(create_output_path("step2_timeevol.png"))


if __name__ == "__main__":
    main()
