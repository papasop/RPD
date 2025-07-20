# 安装必要的库
!pip install mpmath sympy

import numpy as np
from mpmath import mp
from sympy import primepi
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

# 设置高精度
mp.dps = 50

# 前10个非平凡零点虚部
lambda_n = [
    mp.mpf('14.134725141734693790457251983562470270784257115699243331281'),
    mp.mpf('21.022039638771554992628479593896902777667930192123466166247'),
    mp.mpf('25.010857580145688763213790992562821818659549672557996672496'),
    mp.mpf('30.424876125859513210311897530584091320181560023662564296976'),
    mp.mpf('32.935061587739189690662368964074903488812715603517039009280'),
    mp.mpf('37.586178158825671257217763480705332821405597350830793468991'),
    mp.mpf('40.918719012147495187398126914633093830429906505984974565387'),
    mp.mpf('43.327073280914999519496122165406805577549168932803343960404'),
    mp.mpf('48.005147980648200151518934618186695997614108796614373797119'),
    mp.mpf('49.773832477672302181916784678563599058765302614207695360604')
]
N = len(lambda_n)

# 模型函数定义
def compute_amplitudes(lambda_n, k, c=1.0):
    return [mp.mpf(c) / (mp.sqrt(lam * n) * k) for n, lam in enumerate(lambda_n, 1)]

def phi(t, lambda_n, An, theta_n):
    t = mp.mpf(t)
    return sum(a * mp.cos(lam * mp.log(t) + theta) for a, lam, theta in zip(An, lambda_n, theta_n)) if t > 1 else mp.mpf('0')

def rho(t, lambda_n, An, theta_n):
    t = mp.mpf(t)
    return mp.mpf('1') / mp.log(t) + phi(t, lambda_n, An, theta_n) if t > 1 else mp.mpf('0')

def delta(t, lambda_n, An, theta_n):
    t = mp.mpf(t)
    pi_t = mp.mpf(primepi(t))
    return pi_t / t - rho(t, lambda_n, An, theta_n)

def H(t, lambda_n, An, theta_n):
    d = delta(t, lambda_n, An, theta_n)
    return mp.log(mp.mpf('1') + d**2)

def Phi(t, lambda_n, An, theta_n):
    return phi(t, lambda_n, An, theta_n)

def K(t, lambda_n, An, theta_n, h=mp.mpf('1e-14')):
    phi_t = Phi(t, lambda_n, An, theta_n)
    phi_th = Phi(t + h, lambda_n, An, theta_n)
    H_t = H(t, lambda_n, An, theta_n)
    H_th = H(t + h, lambda_n, An, theta_n)
    epsilon = mp.mpf('1e-20')
    log_phi_diff = (mp.log(mp.fabs(phi_th) + epsilon) - mp.log(mp.fabs(phi_t) + epsilon)) / h
    log_H_diff = (mp.log(mp.fabs(H_th) + epsilon) - mp.log(mp.fabs(H_t) + epsilon)) / h
    return log_phi_diff / (log_H_diff + epsilon)

# 优化 k 和 theta_n，固定 c = 1
def optimize_k_and_theta(t, lambda_n, c=1.0):
    def objective(params):
        k = params[0]
        theta_n = params[1:]
        An = compute_amplitudes(lambda_n, k, c)
        delta_val = float(delta(t, lambda_n, An, [mp.mpf(th) for th in theta_n]))
        K_val = float(K(t, lambda_n, An, [mp.mpf(th) for th in theta_n]))
        return (K_val - 1.0)**2 + (delta_val * np.log(float(t)))**2
    init_k = 10.0  # 初始 k，相当于 c = 0.1
    init_theta = np.zeros(N)
    init_params = np.concatenate(([init_k], init_theta))
    bounds = [(1.0, 200.0)] + [(0, 2 * np.pi)] * N  # k 范围对应 c 的倒数
    result = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-10})
    return result.x[0], result.x[1:]

# 扫描 x 值
x_vals = list(range(10, 101, 10)) + list(range(200, 1001, 100))
scan_results = []

for x in x_vals:
    k_opt, theta_opt = optimize_k_and_theta(mp.mpf(x), lambda_n, c=1.0)
    An_used = compute_amplitudes(lambda_n, k=k_opt, c=1.0)
    pi_x = primepi(x)
    rho_val = float(rho(x, lambda_n, An_used, [mp.mpf(th) for th in theta_opt]))
    pi_ratio = pi_x / x
    delta_val = pi_ratio - rho_val
    phi_val = rho_val - float(mp.mpf('1') / mp.log(x))
    H_val = float(H(x, lambda_n, An_used, [mp.mpf(th) for th in theta_opt]))
    K_val = float(K(x, lambda_n, An_used, [mp.mpf(th) for th in theta_opt]))
    C_estimate = abs(delta_val) * mp.log(x)
    scan_results.append((x, round(k_opt, 6), pi_x, round(pi_ratio, 6), round(rho_val, 6), 
                        round(phi_val, 6), round(delta_val, 6), round(C_estimate, 6), 
                        round(K_val, 6), round(H_val, 6)))

# 输出 DataFrame 表格
df = pd.DataFrame(scan_results, columns=["x", "k*", "π(x)", "π(x)/x", "ρ(x)", "φ(x)", "δ(x)", "C_estimate", "K(x)", "H(x)"])
print(df)

# 计算 C 的估计上界
C_max = df["C_estimate"].max()
print(f"\n估计的 C 上界: {C_max:.6f}")

# 绘图展示 φ(x), δ(x), K(x), 和 C_estimate
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(df["x"], df["φ(x)"], marker='o', label='φ(x)')
plt.grid(True)
plt.title("Oscillatory Correction φ(x)")
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(df["x"], df["δ(x)"], marker='o', label='δ(x)')
plt.grid(True)
plt.title("Residual δ(x)")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(df["x"], df["K(x)"], marker='o', label='K(x)')
plt.axhline(y=1.0, color='r', linestyle='--', label='K(x)=1')
plt.grid(True)
plt.title("Structure Ratio K(x)")
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(df["x"], df["C_estimate"], marker='o', label='C_estimate')
plt.axhline(y=C_max, color='r', linestyle='--', label=f'C_max={C_max:.6f}')
plt.grid(True)
plt.title("C Estimate (|δ(x)| * log(x))")
plt.legend()

plt.tight_layout()
plt.show()