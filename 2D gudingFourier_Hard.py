import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import math
import numpy as np
import torch
import torch.nn as tn
import matplotlib
import datetime
import shutil

matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path: sys.path.append(project_root)

try:
    from utilizers import DNN_Log_Print

    HAS_CUSTOM_MODULES = True

except ImportError:
    print("Warning: Custom modules not found. Using fallback standard PyTorch NN.")
    HAS_CUSTOM_MODULES = False


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# 1) 数值算子生成
def gamma_fn(x: float) -> float:
    return math.gamma(x)


def grunwald_g(alpha: float, M: int) -> np.ndarray:
    g = np.zeros(M + 1, dtype=float)
    g[0] = 1.0
    for k in range(1, M + 1):
        g[k] = (1.0 - (alpha + 1.0) / k) * g[k - 1]
    return g


def lubich_L2_q(alpha: float, K: int) -> np.ndarray:
    g = grunwald_g(alpha, K)
    q = np.zeros(K + 1, dtype=float)
    c = (1.5) ** alpha
    for k in range(0, K + 1):
        s = 0.0
        for m in range(0, k + 1):
            s += (3.0 ** (-m)) * g[m] * g[k - m]
        q[k] = c * s
    return q


def build_left_Ap_from_weights(alpha: float, Nx: int, h: float, p: int, weights: np.ndarray) -> np.ndarray:
    A = np.zeros((Nx + 1, Nx + 1), dtype=float)
    inv = h ** (-alpha)
    for i in range(Nx + 1):
        kmax = i + p
        if kmax < 0: continue
        kmax = min(kmax, len(weights) - 1)
        for k in range(kmax + 1):
            j = i - k + p
            if 0 <= j <= Nx:
                A[i, j] += inv * weights[k]
    return A


def w4(alpha, p, q, r, s, pb, qb, rb, sb):
    a1 = (r * s) - (p * q)
    a2 = (rb * sb) - (pb * qb)
    b1 = 6.0 * (p * q * r * s) * (r + s - p - q) + 4.0 * alpha * (
            (r * s) * (r + s) - (p * q) * (p + q)) + 9.0 * alpha * ((r * s) - (p * q))
    b2 = 6.0 * (pb * qb * rb * sb) * (rb + sb - pb - qb) + 4.0 * alpha * (
            (rb * sb) * (rb + sb) - (pb * qb) * (pb + qb)) + 9.0 * alpha * ((rb * sb) - (pb * qb))
    den = a1 * b2 - a2 * b1
    return (a1 * b2) / den, 1.0 - ((a1 * b2) / den)


def w2(p: int, q: int):
    return q / (q - p), p / (p - q)


def w3(alpha: float, p: int, q: int, r: int, s: int):
    rs, pq = r * s, p * q
    a = (3.0 * rs + 2.0 * alpha) / (3.0 * (rs - pq))
    b = (3.0 * pq + 2.0 * alpha) / (3.0 * (pq - rs))
    return a, b


def build_A2_left(alpha, Nx, h, p, q, qweights):
    lp, lq = w2(p, q)
    return lp * build_left_Ap_from_weights(alpha, Nx, h, p, qweights) + lq * build_left_Ap_from_weights(alpha, Nx, h, q,
                                                                                                        qweights)


def build_A3_left(alpha, Nx, h, p, q, r, s, qweights):
    a, b = w3(alpha, p, q, r, s)
    return a * build_A2_left(alpha, Nx, h, p, q, qweights) + b * build_A2_left(alpha, Nx, h, r, s, qweights)


def build_A4_left(alpha, Nx, h, p, q, r, s, pb, qb, rb, sb, qweights):
    theta, theta_bar = w4(alpha, p, q, r, s, pb, qb, rb, sb)
    return theta * build_A3_left(alpha, Nx, h, p, q, r, s, qweights) + theta_bar * build_A3_left(alpha, Nx, h, pb, qb,
                                                                                                 rb, sb, qweights)


def build_A_riesz_interior(method: str, gamma: float, Nx: int, h: float) -> np.ndarray:
    coef = -1.0 / (2.0 * math.cos(gamma * math.pi / 2.0))
    qweights = lubich_L2_q(gamma, Nx + 8)
    AL_full = build_A4_left(gamma, Nx, h, 1, 3, 1, 0, 1, 3, 1, -2, qweights)
    AR_full = AL_full.T
    return coef * (AL_full[1:-1, 1:-1] + AR_full[1:-1, 1:-1])


def build_L1_time_matrix(alpha: float, Nt: int, dt: float, device: str, dtype=torch.float32):
    k = np.arange(Nt, dtype=np.float64)
    b = (k + 1) ** (1 - alpha) - k ** (1 - alpha)
    c0 = (dt ** (-alpha)) / gamma_fn(2.0 - alpha)
    L = np.zeros((Nt + 1, Nt + 1), dtype=np.float64)
    n_idx, j_idx = np.indices((Nt + 1, Nt + 1))
    diag_mask = (n_idx == j_idx) & (n_idx > 0)
    L[diag_mask] = c0 * b[0]
    col0_mask = (j_idx == 0) & (n_idx > 0)
    L[col0_mask] = -c0 * b[n_idx[col0_mask] - 1]
    lower_mask = (j_idx > 0) & (j_idx < n_idx)
    diff = n_idx[lower_mask] - j_idx[lower_mask]
    L[lower_mask] = -c0 * (b[diff - 1] - b[diff])
    return torch.tensor(L, device=device, dtype=dtype)


# 2) 网络架构 (经典傅里叶 + 14维 MAE + 门控 + 硬约束)

class Gated_MAE_fPINN_2D(tn.Module):
    def __init__(self, hidden_layer, alpha_val, gamma_val, freqs_np):
        super(Gated_MAE_fPINN_2D, self).__init__()
        self.alpha_val = alpha_val
        self.gamma_val = gamma_val

        if freqs_np is not None:
            self.register_buffer("freqs", torch.tensor(freqs_np, dtype=torch.float32).view(1, -1))
            fourier_dim = self.freqs.shape[1] * 6
        else:
            self.freqs = None
            fourier_dim = 0

        in_smooth = 3 + fourier_dim
        self.net_smooth = self._build_mlp(in_smooth, hidden_layer)

        self.net_singular = self._build_mlp(14, [64, 64, 64, 64])
        self.net_gate = self._build_gate(3, [32, 32])

    def _build_mlp(self, in_dim, hidden_list):
        layers = []
        dims = [in_dim] + list(hidden_list) + [1]
        for i in range(len(dims) - 2):
            layers.append(tn.Linear(dims[i], dims[i + 1]))
            layers.append(tn.Tanh())
            # layers.append(tn.GELU())
        layers.append(tn.Linear(dims[-2], dims[-1]))
        return tn.Sequential(*layers)

    def _build_gate(self, in_dim, hidden_list):
        layers = []
        dims = [in_dim] + list(hidden_list) + [1]
        for i in range(len(dims) - 2):
            layers.append(tn.Linear(dims[i], dims[i + 1]))
            layers.append(tn.Tanh())
            # layers.append(tn.GELU())
        last_layer = tn.Linear(dims[-2], dims[-1])
        torch.nn.init.constant_(last_layer.weight, 0.0)
        torch.nn.init.constant_(last_layer.bias, 0.0)
        layers.append(last_layer)
        layers.append(tn.Sigmoid())
        return tn.Sequential(*layers)

    def forward(self, XYT):
        x = XYT[:, 0:1]
        y = XYT[:, 1:2]
        t = XYT[:, 2:3]

        if self.freqs is not None:
            arg_x = 2.0 * math.pi * x * self.freqs
            arg_y = 2.0 * math.pi * y * self.freqs
            arg_t = 2.0 * math.pi * t * self.freqs

            fourier_feat = torch.cat([
                torch.sin(arg_x), torch.cos(arg_x),
                torch.sin(arg_y), torch.cos(arg_y),
                torch.sin(arg_t), torch.cos(arg_t)
            ], dim=-1)
            feat_smooth = torch.cat([x, y, t, fourier_feat], dim=-1)
        else:
            feat_smooth = torch.cat([x, y, t], dim=-1)

        # ---------------- 提取奇异特征 (保留原版的 14 维) ----------------
        zx = torch.clamp(1.0 - x ** 2, min=1e-8)
        zy = torch.clamp(1.0 - y ** 2, min=1e-8)
        sb_x = zx ** (self.gamma_val / 2.0)
        sb_y = zy ** (self.gamma_val / 2.0)
        sg_x, sg_y = x * sb_x, y * sb_y
        sb_xy = sb_x * sb_y

        t_safe = torch.clamp(t, min=1e-8)
        t_a = t_safe ** self.alpha_val
        t_2a = t_safe ** (2.0 * self.alpha_val)
        t_poly2 = t_safe ** 2.0
        t_1_a = t_safe ** (1.0 + self.alpha_val)
        t_1_2a = t_safe ** (1.0 + 2.0 * self.alpha_val)

        st_coupling = t_a * sb_xy

        feat_singular = torch.cat([x, y, t, sb_x, sg_x, sb_y, sg_y, sb_xy,
                                   t_poly2, t_a, t_2a, t_1_a, t_1_2a, st_coupling], dim=-1)

        # ---------------- 三分支前向传播 ----------------
        U_smooth = self.net_smooth(feat_smooth)
        U_singular = self.net_singular(feat_singular)
        G = self.net_gate(XYT)

        # ---------------- 门控融合 + 2D 时空硬约束 Ansatz ----------------
        U_raw = G * U_singular + (1.0 - G) * U_smooth

        ansatz = t * (1.0 - x ** 2.0) * (1.0 - y ** 2.0)
        U_pred = U_raw * ansatz

        return U_pred


# ============================================================
# 3) 绘图
# ============================================================
def setup_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18
    })


def save_2d_heatmap_triplet_t_end(x_in, y_in, U_true_end, U_pred_end, out_png):
    try:
        setup_publication_style()
        X, Y = np.meshgrid(x_in, y_in, indexing='ij')
        Err = np.abs(U_pred_end - U_true_end)

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        cmap_field, cmap_error = 'RdYlBu_r', 'Reds'

        im1 = axes[0].pcolormesh(X, Y, U_true_end, cmap=cmap_field, shading='gouraud')
        axes[0].set_title(r'Exact Solution $u(x, y, t=1)$', pad=15)
        axes[0].set_xlabel(r'$x$');
        axes[0].set_ylabel(r'$y$')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].pcolormesh(X, Y, U_pred_end, cmap=cmap_field, shading='gouraud')
        axes[1].set_title(r'Predicted $u_\theta(x, y, t=1)$', pad=15)
        axes[1].set_xlabel(r'$x$');
        axes[1].set_ylabel(r'$y$')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        im3 = axes[2].pcolormesh(X, Y, Err, cmap=cmap_error, shading='gouraud')
        axes[2].set_title(r'MAE Absolute Error at $t=1$', pad=15)
        axes[2].set_xlabel(r'$x$');
        axes[2].set_ylabel(r'$y$')
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(out_png, dpi=400, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Plotting failed: {e}")


# ============================================================
# 4) 训练
# ============================================================
def solve_PINN_FODE_MMS_2D(R):
    device = R['opt2device']
    torch.set_default_dtype(torch.float32)

    log_out_path = R['FolderName']
    if not os.path.exists(log_out_path): os.makedirs(log_out_path, exist_ok=True)
    logfile_name = 'log_Training_Details.txt'
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w', encoding='utf-8')

    alpha, gamma = R["alpha"], 2.0 * R["beta0"]
    xL, xR = R["domain_x"]
    yL, yR = R["domain_y"]
    t0, t1 = R["domain_t"]
    Nx, Ny, Nt = R["Nx"], R["Ny"], R["Nt"]

    A_space_np = build_A_riesz_interior('WSLD4', gamma, Nx, (xR - xL) / Nx)
    A_space = torch.tensor(A_space_np, device=device, dtype=torch.float32)
    Lt = build_L1_time_matrix(alpha, Nt, (t1 - t0) / Nt, device=device)

    x_in_np = np.linspace(xL, xR, Nx + 1)[1:-1]
    y_in_np = np.linspace(yL, yR, Ny + 1)[1:-1]
    t_np = np.linspace(t0, t1, Nt + 1)

    x_in_t = torch.tensor(x_in_np, device=device, dtype=torch.float32)
    y_in_t = torch.tensor(y_in_np, device=device, dtype=torch.float32)
    t_grid_t = torch.tensor(t_np, device=device, dtype=torch.float32)

    T_grid, X_grid, Y_grid = torch.meshgrid(t_grid_t, x_in_t, y_in_t, indexing='ij')
    XYT = torch.stack([X_grid.flatten(), Y_grid.flatten(), T_grid.flatten()], dim=-1)

    print("Pre-calculating 2D discrete analytical Forcing term F(x,y,t) directly on the training mesh...")

    U_true = np.zeros((Nt + 1, Nx - 1, Ny - 1), dtype=np.float64)
    beta0 = gamma / 2.0
    for n in range(Nt + 1):
        u1 = (t_np[n] ** (1.0 + alpha)) * np.outer((1.0 - x_in_np ** 2) ** (1.0 + beta0),
                                                   (1.0 - y_in_np ** 2) ** (1.0 + beta0))
        u2 = (t_np[n] ** 2.0) * np.outer(np.sin(math.pi * x_in_np), np.sin(math.pi * y_in_np)) * np.cos(
            4.0 * math.pi * t_np[n])
        U_true[n, :, :] = u1 + u2

    U_true_tensor = torch.tensor(U_true, device=device, dtype=torch.float32)

    with torch.no_grad():
        DtU_true = (Lt @ U_true_tensor.view(Nt + 1, -1)).view(Nt + 1, Nx - 1, Ny - 1)
        DxU_true = torch.einsum('ik, tkj -> tij', A_space, U_true_tensor)
        DyU_true = torch.einsum('jk, tik -> tij', A_space, U_true_tensor)

        F_t = DtU_true - R["kappa"] * (DxU_true + DyU_true) + R["nu"] * U_true_tensor

    print("Initializing 2D Network (Classical Fourier + Gated MAE + Hard-Constraint Ansatz)...")
    mscalednn = Gated_MAE_fPINN_2D(
        hidden_layer=R['hidden_layers'],
        alpha_val=R['alpha'],
        gamma_val=2.0 * R['beta0'],
        freqs_np=R.get('freq', None)
    )
    if R['use_gpu']: mscalednn = mscalednn.to(device)

    optimizer = torch.optim.Adam(mscalednn.parameters(), lr=R['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)
    mse = tn.MSELoss()

    loss_history = []
    global_t0 = time.time()
    interval_start_str = f"--- Starting 2D Discrete-MMS Validation: Nx={Nx}, Ny={Ny}, Nt={Nt} ---"
    print(interval_start_str)
    log_fileout.write(interval_start_str + '\n')

    for epoch in range(1, R['max_epoch'] + 1):
        optimizer.zero_grad()

        U_pred_flat = mscalednn(XYT)
        U_pred = U_pred_flat.view(Nt + 1, Nx - 1, Ny - 1)

        DtU = (Lt @ U_pred.view(Nt + 1, -1)).view(Nt + 1, Nx - 1, Ny - 1)
        DxU = torch.einsum('ik, tkj -> tij', A_space, U_pred)
        DyU = torch.einsum('jk, tik -> tij', A_space, U_pred)

        Res = DtU - R["kappa"] * (DxU + DyU) + R["nu"] * U_pred - F_t
        scale = torch.mean(torch.abs(F_t[1:, :, :])) + 1.0
        ResN = Res / scale
        loss_pde = mse(ResN[1:, :, :], torch.zeros_like(ResN[1:, :, :]))

        # Ansatz 硬约束，初边界误差严格恒为 0
        loss_ic = torch.tensor(0.0, device=device)
        loss_bc = torch.tensor(0.0, device=device)

        loss_total = loss_pde
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append([epoch, loss_total.item(), loss_pde.item(), loss_ic.item(), loss_bc.item()])

        if epoch % 500 == 0:
            run_times = time.time() - global_t0
            print_msg = (f"  Epoch: {epoch:<5} Time: {run_times:<.1f}s, Total/PDE Loss: {loss_total.item():<.4e}")
            print(print_msg)
            log_fileout.write(print_msg + '\n')

    print(f"\n=== Training Finished === Total Time: {time.time() - global_t0:.2f}s")

    with torch.no_grad():
        U_pred_final = mscalednn(XYT).view(Nt + 1, Nx - 1, Ny - 1).detach().cpu().numpy()

    # =========================================================================
    # 数据存储
    # =========================================================================
    loss_history_np = np.array(loss_history)
    np.save(os.path.join(log_out_path, f"loss_fPINN_WSLD4_2D.npy"), loss_history_np)

    np.save(os.path.join(log_out_path, f"U_true_2D.npy"), U_true)
    np.save(os.path.join(log_out_path, f"U_pred_fPINN_2D.npy"), U_pred_final)

    np.save(os.path.join(log_out_path, f"x_grid_2D.npy"), x_in_np)
    np.save(os.path.join(log_out_path, f"y_grid_2D.npy"), y_in_np)
    np.save(os.path.join(log_out_path, f"t_grid_2D.npy"), t_np)

    print(f"Data successfully saved to: {log_out_path}")

    e_L2 = np.linalg.norm(U_pred_final[-1] - U_true[-1]) / (np.linalg.norm(U_true[-1]) + 1e-14)
    e_Linf = np.max(np.abs(U_pred_final[-1] - U_true[-1]))
    error_msg = f"Final step (t=1.0) dynamic evolution error: \n RelL2(T)={e_L2:.6e} \n Linf(T)={e_Linf:.6e}"
    print(error_msg)
    log_fileout.write("\n" + error_msg + "\n")
    log_fileout.close()

    try:
        plt.figure(figsize=(10, 6))
        plt.semilogy(loss_history_np[:, 0], loss_history_np[:, 1], 'b-', linewidth=1.5, label='Total Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss (Log Scale)")
        plt.title("2D fPINN Training Loss")
        plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='grey', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(log_out_path, f"loss_curve_WSLD4_2D.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Loss plot failed: {e}")

    save_2d_heatmap_triplet_t_end(x_in_np, y_in_np, U_true[-1], U_pred_final[-1],
                                  os.path.join(log_out_path, f"2D_Heatmap_fPINN_t1.png"))


# ============================================================
# 5) 底部字典配置
# ============================================================
if __name__ == "__main__":
    set_seed(42)

    R = {}
    R['gpuNo'] = 0
    R['use_gpu'] = torch.cuda.is_available()
    R['opt2device'] = 'cuda:' + str(R['gpuNo']) if R['use_gpu'] else 'cpu'

    R['domain_x'] = [-1.0, 1.0]
    R['domain_y'] = [-1.0, 1.0]
    R['domain_t'] = [0.0, 1.0]
    R['alpha'] = 0.5
    R['beta0'] = 0.75
    R['kappa'] = 1.0
    R['nu'] = 0.0

    R['Nx'] = 50
    R['Ny'] = 50
    R['Nt'] = 150

    R['hidden_layers'] = (128, 128, 128, 128, 64)
    R['learning_rate'] = 5e-4
    R['max_epoch'] = 20000

    R['freq'] = np.arange(1, 25)

    store_file = f'MMS_Model_fPINN_2D_H_GS_FF_Discrete'
    OUT_DIR = os.path.join(project_root, 'Results_MMS', store_file)
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR, exist_ok=True)
    date_time_dir = datetime.datetime.now().strftime('%m%d_%H%M%S')
    R['FolderName'] = os.path.join(OUT_DIR, date_time_dir)
    if not os.path.exists(R['FolderName']): os.makedirs(R['FolderName'], exist_ok=True)

    try:
        shutil.copy(__file__, os.path.join(R['FolderName'], os.path.basename(__file__)))
    except:
        pass

    solve_PINN_FODE_MMS_2D(R)