import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Fix Chinese font rendering in matplotlib charts
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="4QAM完整通信链路仿真", page_icon="📡", layout="wide")

# ============================
# SIDEBAR: Global Parameters
# ============================
st.sidebar.header("⚙️ 全局参数")
noise_std = st.sidebar.slider("噪声标准差 σ", 0.0, 0.8, 0.15, step=0.05)
fc = st.sidebar.slider("载波频率 f_c (Hz)", 5, 30, 10, step=1)
seed = st.sidebar.number_input("随机种子", 0, 9999, 42)
st.sidebar.markdown("---")

# ============================
# SIGNAL CHAIN CONSTANTS
# ============================
Tsym = 1.0
fs = 400
num_symbols = 8
sps = int(fs * Tsym)  # samples per symbol = 400
total_duration = num_symbols * Tsym  # 8 seconds
num_samples = num_symbols * sps  # 3200

# ============================
# STEP 1: Generate Input Bits (cycling 00→01→10→11)
# ============================
pattern = [0, 0, 0, 1, 1, 0, 1, 1]
cycle_len = len(pattern)
bits = [pattern[i % cycle_len] for i in range(num_symbols * 2)]

# ============================
# STEP 2: Symbol Mapping (Gray coding)
# ============================
gray_map = {
    (0, 0): complex(1, -1),
    (0, 1): complex(1, 1),
    (1, 0): complex(-1, -1),
    (1, 1): complex(-1, 1),
}
symbols = [gray_map[(bits[2 * i], bits[2 * i + 1])] for i in range(num_symbols)]
I_vals = np.array([s.real for s in symbols])
Q_vals = np.array([s.imag for s in symbols])

# ============================
# TIME ARRAY & BASEBAND
# ============================
t = np.arange(0, total_duration, 1 / fs)
symbol_idx = np.clip(np.floor(t / Tsym).astype(int), 0, num_symbols - 1)
I_bb = I_vals[symbol_idx]
Q_bb = Q_vals[symbol_idx]

# ============================
# STEP 3: Up-conversion
# s_RF(t) = I(t)·cos(2πfc·t) - Q(t)·sin(2πfc·t)
# ============================
carrier_cos = np.cos(2 * np.pi * fc * t)
carrier_sin = np.sin(2 * np.pi * fc * t)
s_rf = I_bb * carrier_cos - Q_bb * carrier_sin

# ============================
# STEP 4: AWGN Channel
# r(t) = s_RF(t) + w(t),  w ~ N(0, σ²)
# ============================
rng = np.random.default_rng(seed)
noise = rng.normal(0, noise_std, num_samples)
s_rf_noisy = s_rf + noise

# ============================
# STEP 5: Down-conversion & LPF
# I_mixed = 2·r(t)·cos(2πfc·t)
# Q_mixed = -2·r(t)·sin(2πfc·t)
# LPF: moving average (window = Tsym/2)
# ============================
I_mixed = 2 * s_rf_noisy * carrier_cos
Q_mixed = -2 * s_rf_noisy * carrier_sin

window_size = sps // 2
lpf_kernel = np.ones(window_size) / window_size
I_rec = np.convolve(I_mixed, lpf_kernel, mode="same")
Q_rec = np.convolve(Q_mixed, lpf_kernel, mode="same")

# ============================
# STEP 6: Sampling & ML Decision
# ============================
sample_times_centers = np.arange(0.5, total_duration, Tsym)
sample_indices = (sample_times_centers * fs).astype(int)
I_sampled = I_rec[sample_indices]
Q_sampled = Q_rec[sample_indices]

const_pts = [complex(1, -1), complex(1, 1), complex(-1, -1), complex(-1, 1)]
inv_map = {
    (1, -1): (0, 0),
    (1, 1): (0, 1),
    (-1, -1): (1, 0),
    (-1, 1): (1, 1),
}

rx_bits = []
rx_decisions = []
for i in range(num_symbols):
    s_rx = complex(I_sampled[i], Q_sampled[i])
    diffs = [abs(s_rx - c) for c in const_pts]
    idx_closest = int(np.argmin(diffs))
    c_chosen = const_pts[idx_closest]
    rx_decisions.append(c_chosen)
    bit_pair = inv_map[(int(c_chosen.real), int(c_chosen.imag))]
    rx_bits.extend(bit_pair)

# ============================
# BER
# ============================
bit_errors = sum(1 for b, r in zip(bits, rx_bits) if b != r)
ber = bit_errors / len(bits) if len(bits) > 0 else 0

# ============================
# ZOOM WINDOWS
# ============================
ZOOM_DURATION = 2.0
zoom_mask = (t >= 0) & (t < ZOOM_DURATION)
MIX_ZOOM_DURATION = 0.4
mix_zoom_mask = (t >= 0) & (t < MIX_ZOOM_DURATION)

# ============================
# SIDEBAR: Module Navigation
# ============================
st.sidebar.header("📋 模块导航")
module = st.sidebar.radio(
    "选择查看步骤",
    [
        "📥 1. 输入比特流",
        "📊 2. 符号映射与基带波形",
        "⬆️ 3. 上变频 (通带调制)",
        "🌊 4. AWGN信道",
        "⬇️ 5. 下变频与低通滤波",
        "📍 6. 解调判决 (ML检测)",
        "✅ 7. 比特恢复与误码率",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"符号速率: 1 Baud | 采样率: {fs} Hz\n"
    f"载波: {fc} Hz | 总时长: {total_duration}s\n"
    f"当前 σ = {noise_std:.2f}, BER = {ber:.4f}"
)


# ============================
# HELPER
# ============================
def section_header(title, formula=None):
    st.markdown("---")
    st.subheader(title)
    if formula:
        st.latex(formula)


# ============================
# MAIN TITLE
# ============================
st.title("📡 4QAM 完整通信链路仿真")
st.caption(
    "输入信号: 00→01→10→11 循环 · 符号周期 1s · "
    f"载波 f_c={fc} Hz · 全链路包含 上变频→AWGN→下变频→LPF→ML判决"
)

# ============================
# MODULE 1: Input Bits
# ============================
if module == "📥 1. 输入比特流":
    section_header("📥 步骤1: 输入比特流")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**比特序列 (00→01→10→11 循环2遍)**")
        bit_str = "  ".join(str(b) for b in bits)
        st.code(bit_str, language=None)
        st.markdown(f"共 __{len(bits)}__ 比特 = __{num_symbols}__ 个符号")

    with col_b:
        st.markdown("**比特按符号分组**")
        pairs = [f"({bits[2 * i]}{bits[2 * i + 1]})" for i in range(num_symbols)]
        st.code(" → ".join(pairs), language=None)

    st.markdown(
        """
    **4QAM (Quadrature Amplitude Modulation)** 是一种数字调制方式，
    每个符号用载波的 **2 种正交状态**（I 同相 & Q 正交）传输 2 比特信息，共 4 种组合。

    后续步骤将完整展示这 16 个比特依次经过:
    ```
    映射 → 基带 → 上变频 → 信道传输 → 下变频 → LPF → 采样判决 → 恢复
    ```
    """
    )

# ============================
# MODULE 2: Symbol Mapping & Baseband
# ============================
elif module == "📊 2. 符号映射与基带波形":
    section_header("📊 步骤2: 符号映射 (Gray编码)", formula=None)

    col_tab, col_math = st.columns([1, 2])
    with col_tab:
        st.markdown(
            """
        | 比特对 | I | Q |
        |--------|---|---|
        | **00** | +1 | −1 |
        | **01** | +1 | +1 |
        | **10** | −1 | −1 |
        | **11** | −1 | +1 |
        """
        )
        st.caption("相邻星座点仅差 1 比特 → 最小化误码")

    with col_math:
        st.latex(r"s_k = I_k + j Q_k,\quad I_k, Q_k \in \{\pm 1\}")
        st.latex(
            r"""
        \begin{aligned}
        \text{00} &\rightarrow 1 - j \\
        \text{01} &\rightarrow 1 + j \\
        \text{10} &\rightarrow -1 - j \\
        \text{11} &\rightarrow -1 + j
        \end{aligned}
        """
        )

    st.markdown("### 星座图 与 基带 I/Q 波形")

    fig_bb, (ax_c, ax_w) = plt.subplots(1, 2, figsize=(13, 5))
    fig_bb.patch.set_facecolor("white")

    # --- Constellation ---
    const_arr = np.array(const_pts)
    labels_bits = ["00", "01", "10", "11"]
    ax_c.scatter(
        const_arr.real, const_arr.imag, c="#1f77b4", s=180, zorder=5,
        edgecolors="black", linewidths=0.6,
    )
    for idx_pt, pt in enumerate(const_pts):
        ax_c.annotate(
            labels_bits[idx_pt], (pt.real, pt.imag),
            textcoords="offset points", xytext=(14, 14),
            fontsize=11, fontweight="bold", color="#1f77b4",
        )
    # Symbol sequence arrows
    for idx_sym in range(num_symbols - 1):
        ax_c.annotate(
            "", xy=(I_vals[idx_sym + 1], Q_vals[idx_sym + 1]),
            xytext=(I_vals[idx_sym], Q_vals[idx_sym]),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.5, lw=1.2,
                            connectionstyle="arc3,rad=0.2"),
        )
    ax_c.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax_c.axvline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax_c.set_xlim(-1.8, 1.8)
    ax_c.set_ylim(-1.8, 1.8)
    ax_c.set_xlabel("I (同相分量)", fontsize=11)
    ax_c.set_ylabel("Q (正交分量)", fontsize=11)
    ax_c.set_title("4QAM 星座图 + 符号轨迹", fontsize=12, fontweight="bold")
    ax_c.set_aspect("equal")
    ax_c.grid(True, alpha=0.3)

    # --- Baseband Waveforms ---
    t_step = np.arange(num_symbols + 1) * Tsym
    i_step = np.append(I_vals, I_vals[-1])
    q_step = np.append(Q_vals, Q_vals[-1])
    ax_w.step(t_step, i_step, where="post", color="blue", linewidth=2, label="I(t)")
    ax_w.step(t_step, q_step, where="post", color="red", linewidth=2, label="Q(t)")
    ax_w.set_xlabel("时间 (s)", fontsize=11)
    ax_w.set_ylabel("幅度", fontsize=11)
    ax_w.set_title("基带 I(t) / Q(t) 波形", fontsize=12, fontweight="bold")
    ax_w.set_ylim(-1.8, 1.8)
    ax_w.set_xlim(0, total_duration)
    ax_w.legend(fontsize=10, loc="upper right")
    ax_w.grid(True, alpha=0.3)
    # Annotate symbol labels
    for idx_sym in range(num_symbols):
        bit_label = f"{bits[2*idx_sym]}{bits[2*idx_sym+1]}"
        ax_w.axvline(idx_sym * Tsym, color="gray", linestyle=":", linewidth=0.5)
        ax_w.text(idx_sym * Tsym + 0.5, 1.65, bit_label, ha="center", fontsize=8, color="gray")

    fig_bb.tight_layout()
    st.pyplot(fig_bb)
    plt.close(fig_bb)

    st.info(
        "💡 基带信号是 **矩形脉冲** (零阶保持): 每个符号周期内 I(t)、Q(t) 保持恒定。"
        "这两个基带信号将分别调制到正交载波 cos 和 sin 上。"
    )

# ============================
# MODULE 3: Up-conversion
# ============================
elif module == "⬆️ 3. 上变频 (通带调制)":
    section_header(
        "⬆️ 步骤3: 上变频 (Up-conversion)",
        formula=(
            r"s_{\mathrm{RF}}(t) = I(t) \cdot \cos(2\pi f_c t) \;-\; "
            r"Q(t) \cdot \sin(2\pi f_c t)"
        ),
    )

    st.markdown(
        f"""
    基带 I/Q 信号被调制到载波频率 $f_c = {fc}$ Hz 上:
    - **I 路** 调制到 $\\cos(2\\pi f_c t)$ (余弦载波)
    - **Q 路** 调制到 $-\\sin(2\\pi f_c t)$ (负正弦载波，与余弦正交)

    两步乘积相加后得到 **通带射频信号** $s_{{\\mathrm{{RF}}}}(t)$，
    可以被天线发射。
    """
    )

    # --- ZOOM VIEW ---
    st.markdown("#### 🔍 放大视图 (前 2 个符号，可见载波细节)")

    t_z = t[zoom_mask]
    I_z = I_bb[zoom_mask]
    Q_z = Q_bb[zoom_mask]
    ccos_z = carrier_cos[zoom_mask]
    csin_z = carrier_sin[zoom_mask]
    srf_z = s_rf[zoom_mask]

    fig_zoom, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    fig_zoom.patch.set_facecolor("white")

    # Subplot 1: I(t)·cos
    i_mod = I_z * ccos_z
    ax1.fill_between(t_z, 0, I_z, color="blue", alpha=0.12)
    ax1.plot(t_z, i_mod, color="blue", linewidth=0.7, label=r"$I(t)\cos(2\pi f_c t)$")
    ax1.plot(
        t_z, I_z, color="blue", linewidth=1.5, alpha=0.45, linestyle="--",
        label="I(t) 基带包络",
    )
    ax1.set_ylabel("I 路幅度", fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-2.3, 2.3)

    # Subplot 2: -Q(t)·sin
    q_mod = -Q_z * csin_z
    ax2.fill_between(t_z, 0, Q_z, color="red", alpha=0.12)
    ax2.plot(t_z, q_mod, color="red", linewidth=0.7, label=r"$-Q(t)\sin(2\pi f_c t)$")
    ax2.plot(
        t_z, Q_z, color="red", linewidth=1.5, alpha=0.45, linestyle="--",
        label="Q(t) 基带包络",
    )
    ax2.set_ylabel("Q 路幅度", fontsize=10)
    ax2.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-2.3, 2.3)

    # Subplot 3: s_RF = sum
    ax3.plot(t_z, srf_z, color="purple", linewidth=1.0, label=r"$s_{\mathrm{RF}}(t)$")
    # Highlight symbol regions
    for sym_i in range(int(ZOOM_DURATION / Tsym)):
        ax3.axvspan(
            sym_i * Tsym, (sym_i + 1) * Tsym,
            alpha=0.05, color="gray",
        )
        bit_lbl = f"{bits[2*sym_i]}{bits[2*sym_i+1]}"
        ax3.text(
            sym_i * Tsym + Tsym / 2, 2.15, bit_lbl,
            ha="center", fontsize=9, color="gray",
        )
    ax3.set_xlabel("时间 (s)", fontsize=10)
    ax3.set_ylabel("RF 信号", fontsize=10)
    ax3.legend(fontsize=7.5, loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-2.5, 2.5)

    fig_zoom.suptitle(
        f"上变频细节 — I/Q 分量叠加形成 RF 信号 (f_c={fc} Hz, {int(ZOOM_DURATION)}s 窗)",
        fontsize=13, fontweight="bold",
    )
    fig_zoom.tight_layout()
    st.pyplot(fig_zoom)
    plt.close(fig_zoom)

    # --- PANORAMIC VIEW ---
    st.markdown("#### 🌐 全景视图 (全部 8 个符号)")

    fig_pan, ax_pan = plt.subplots(figsize=(13, 3.5))
    fig_pan.patch.set_facecolor("white")
    ax_pan.plot(t, s_rf, color="purple", linewidth=0.5)
    for sym_i in range(num_symbols):
        ax_pan.axvline(sym_i * Tsym, color="gray", linestyle=":", linewidth=0.4)
        bit_lbl = f"{bits[2*sym_i]}{bits[2*sym_i+1]}"
        ax_pan.text(
            sym_i * Tsym + Tsym / 2, 2.4, bit_lbl,
            ha="center", fontsize=8, color="gray",
        )
    ax_pan.set_xlabel("时间 (s)", fontsize=11)
    ax_pan.set_ylabel("幅度", fontsize=11)
    ax_pan.set_title(
        f"调制后 RF 信号 (全部 {num_symbols} 符号, f_c={fc} Hz)",
        fontsize=12, fontweight="bold",
    )
    ax_pan.set_xlim(0, total_duration)
    ax_pan.set_ylim(-2.8, 2.8)
    ax_pan.grid(True, alpha=0.3)
    fig_pan.tight_layout()
    st.pyplot(fig_pan)
    plt.close(fig_pan)

    st.info(
        f"💡 放大图中每秒可见 **{fc} 个载波周期**。I 和 Q 分别调制在相位差 90° 的正交载波上，"
        "两路信号在频谱上重叠但在相位空间上正交，可以同时传输且互不干扰。"
    )

# ============================
# MODULE 4: AWGN Channel
# ============================
elif module == "🌊 4. AWGN信道":
    section_header(
        "🌊 步骤4: AWGN信道",
        formula=r"r(t) = s_{\mathrm{RF}}(t) + w(t),\quad w(t) \sim \mathcal{N}(0, \sigma^2)",
    )

    st.markdown(
        f"""
    经过上变频的 RF 信号在信道中叠加 **加性高斯白噪声 (AWGN)**。
    当前噪声标准差 $\\sigma = {noise_std:.2f}$。

    噪声独立地作用在每一个采样点上，模拟真实的传输信道。
    """
    )

    # --- ZOOM: Clean vs Noisy comparison ---
    st.markdown("#### 🔍 放大对比 (前 2 个符号): 理想 vs 含噪")

    t_z = t[zoom_mask]
    srf_z = s_rf[zoom_mask]
    srf_n_z = s_rf_noisy[zoom_mask]
    noise_z = noise[zoom_mask]
    noise_power = np.var(noise)
    snr_db = 10 * np.log10(np.var(s_rf) / max(noise_power, 1e-12))

    fig_comp, (ax_cl, ax_ns) = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
    fig_comp.patch.set_facecolor("white")

    # Clean
    ax_cl.plot(t_z, srf_z, color="purple", linewidth=0.8, label=r"$s_{\mathrm{RF}}(t)$ 理想")
    ax_cl.set_ylabel("理想 RF", fontsize=10)
    ax_cl.legend(fontsize=8, loc="upper right")
    ax_cl.grid(True, alpha=0.3)
    ax_cl.set_ylim(-2.5, 2.5)

    # Noisy
    ax_ns.plot(t_z, srf_n_z, color="#d62728", linewidth=0.6, alpha=0.85,
               label=r"$r(t)$ 含噪")
    ax_ns.plot(t_z, srf_z, color="purple", linewidth=0.5, alpha=0.3,
               label="理想 (半透明对比)")
    ax_ns.set_xlabel("时间 (s)", fontsize=10)
    ax_ns.set_ylabel("含噪 RF", fontsize=10)
    ax_ns.legend(fontsize=8, loc="upper right")
    ax_ns.grid(True, alpha=0.3)
    ax_ns.set_ylim(-2.5, 2.5)
    # Symbol boundaries
    for sym_i in range(int(ZOOM_DURATION / Tsym)):
        ax_ns.axvline(sym_i * Tsym, color="gray", linestyle=":", linewidth=0.4)

    fig_comp.suptitle(
        f"AWGN 信道效果 (σ={noise_std:.2f}, SNR≈{snr_db:.1f} dB, 前{int(ZOOM_DURATION)}秒)",
        fontsize=13, fontweight="bold",
    )
    fig_comp.tight_layout()
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    # --- PANORAMIC: Noisy RF ---
    st.markdown("#### 🌐 全景: 含噪 RF 信号 (全部 8 符号)")

    fig_np, ax_np = plt.subplots(figsize=(13, 3.5))
    fig_np.patch.set_facecolor("white")
    ax_np.plot(t, s_rf_noisy, color="#d62728", linewidth=0.4, alpha=0.8,
               label=r"$r(t)$ 含噪声接收信号")
    ax_np.plot(t, s_rf, color="purple", linewidth=0.4, alpha=0.3,
               label="理想 (对比)")
    for sym_i in range(num_symbols):
        ax_np.axvline(sym_i * Tsym, color="gray", linestyle=":", linewidth=0.4)
        bit_lbl = f"{bits[2*sym_i]}{bits[2*sym_i+1]}"
        ax_np.text(sym_i * Tsym + 0.5, 2.5, bit_lbl, ha="center", fontsize=8, color="gray")
    ax_np.set_xlabel("时间 (s)", fontsize=11)
    ax_np.set_ylabel("幅度", fontsize=11)
    ax_np.set_title(f"含噪 RF 全景 (σ={noise_std:.2f})", fontsize=12, fontweight="bold")
    ax_np.set_xlim(0, total_duration)
    ax_np.set_ylim(-2.8, 2.8)
    ax_np.legend(fontsize=8, loc="upper right")
    ax_np.grid(True, alpha=0.3)
    fig_np.tight_layout()
    st.pyplot(fig_np)
    plt.close(fig_np)

    # Noise statistics
    col_n1, col_n2, col_n3 = st.columns(3)
    col_n1.metric("噪声功率 σ²", f"{noise_power:.4f}")
    col_n2.metric("信号功率", f"{np.var(s_rf):.4f}")
    col_n3.metric("信噪比 SNR", f"{snr_db:.1f} dB")

    st.info(
        "💡 调节侧边栏的 **噪声标准差 σ** 可以观察不同噪声强度下的信号质量变化。"
        "σ 越大，接收波形越偏离原始信号，后续解调的错误也越多。"
    )

# ============================
# MODULE 5: Down-conversion & LPF
# ============================
elif module == "⬇️ 5. 下变频与低通滤波":
    section_header(
        "⬇️ 步骤5: 下变频与低通滤波 (Down-conversion & LPF)",
        formula=(
            r"\hat{I}(t) = \mathrm{LPF}\big\{2 \cdot r(t) \cdot \cos(2\pi f_c t)\big\}"
            r",\quad "
            r"\hat{Q}(t) = \mathrm{LPF}\big\{-2 \cdot r(t) \cdot \sin(2\pi f_c t)\big\}"
        ),
    )

    st.markdown(
        f"""
    接收端将收到的 RF 信号分别乘以与发送端同频同相的载波 $\\cos(2\\pi f_c t)$
    和 $-\\sin(2\\pi f_c t)$，再经过低通滤波器去除 $2f_c$ 高频分量，
    即可恢复出原始的基带信号。

    **滤波器**: 滑动平均 (窗口 = $T_{{\\mathrm{{sym}}}}/2 = {int(window_size)}$ 样点 =
    {window_size / fs:.2f}秒)
    """
    )

    # --- Before LPF: mixed signals (short zoom to show 2fc ripple) ---
    st.markdown("#### 🔬 混频后 (LPF前) — 短窗视图，可见 2f_c 高频纹波")

    t_mz = t[mix_zoom_mask]
    I_mz = I_mixed[mix_zoom_mask]
    Q_mz = Q_mixed[mix_zoom_mask]

    fig_mix, (ax_im, ax_qm) = plt.subplots(2, 1, figsize=(13, 4.5), sharex=True)
    fig_mix.patch.set_facecolor("white")

    ax_im.plot(t_mz, I_mz, color="blue", linewidth=0.7)
    ax_im.plot(t_mz, I_bb[mix_zoom_mask], color="blue", linewidth=1.5, alpha=0.4,
               linestyle="--", label="原始 I(t) 基带")
    ax_im.set_ylabel(r"$I_{\mathrm{mixed}}(t)$", fontsize=10)
    ax_im.legend(fontsize=7.5, loc="upper right")
    ax_im.grid(True, alpha=0.3)
    ax_im.set_ylim(-3.5, 3.5)

    ax_qm.plot(t_mz, Q_mz, color="red", linewidth=0.7)
    ax_qm.plot(t_mz, Q_bb[mix_zoom_mask], color="red", linewidth=1.5, alpha=0.4,
               linestyle="--", label="原始 Q(t) 基带")
    ax_qm.set_xlabel("时间 (s)", fontsize=10)
    ax_qm.set_ylabel(r"$Q_{\mathrm{mixed}}(t)$", fontsize=10)
    ax_qm.legend(fontsize=7.5, loc="upper right")
    ax_qm.grid(True, alpha=0.3)
    ax_qm.set_ylim(-3.5, 3.5)

    fig_mix.suptitle(
        f"混频输出 (前{MIX_ZOOM_DURATION:.1f}秒, 含直流分量 + 2f_c={2*fc}Hz 高频分量)",
        fontsize=12, fontweight="bold",
    )
    fig_mix.tight_layout()
    st.pyplot(fig_mix)
    plt.close(fig_mix)

    # --- After LPF: recovered vs original ---
    st.markdown("#### 🎯 LPF 输出 — 恢复基带 vs 原始基带 对比")

    fig_rec, (ax_ir, ax_qr) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    fig_rec.patch.set_facecolor("white")

    # I recovery
    ax_ir.plot(t, I_rec, color="blue", linewidth=1.2, label=r"$\hat{I}(t)$ 恢复")
    ax_ir.plot(t, I_bb, color="blue", linewidth=1.5, alpha=0.35, linestyle="--",
               label="I(t) 原始")
    ax_ir.scatter(
        sample_times_centers, I_sampled, color="green", s=50, zorder=5,
        marker="o", edgecolors="black", linewidths=0.5, label="采样点",
    )
    for sym_i in range(num_symbols):
        ax_ir.axvline(sym_i * Tsym, color="gray", linestyle=":", linewidth=0.4)
    ax_ir.set_ylabel(r"$\hat{I}(t)$", fontsize=10)
    ax_ir.legend(fontsize=7.5, loc="upper right", ncol=3)
    ax_ir.grid(True, alpha=0.3)
    ax_ir.set_ylim(-2, 2)

    # Q recovery
    ax_qr.plot(t, Q_rec, color="red", linewidth=1.2, label=r"$\hat{Q}(t)$ 恢复")
    ax_qr.plot(t, Q_bb, color="red", linewidth=1.5, alpha=0.35, linestyle="--",
               label="Q(t) 原始")
    ax_qr.scatter(
        sample_times_centers, Q_sampled, color="green", s=50, zorder=5,
        marker="o", edgecolors="black", linewidths=0.5, label="采样点",
    )
    for sym_i in range(num_symbols):
        ax_qr.axvline(sym_i * Tsym, color="gray", linestyle=":", linewidth=0.4)
    ax_qr.set_xlabel("时间 (s)", fontsize=10)
    ax_qr.set_ylabel(r"$\hat{Q}(t)$", fontsize=10)
    ax_qr.legend(fontsize=7.5, loc="upper right", ncol=3)
    ax_qr.grid(True, alpha=0.3)
    ax_qr.set_ylim(-2, 2)

    fig_rec.suptitle(
        "LPF 输出: 恢复的基带信号 vs 原始基带信号 (绿点 = 采样判决时刻)",
        fontsize=12, fontweight="bold",
    )
    fig_rec.tight_layout()
    st.pyplot(fig_rec)
    plt.close(fig_rec)

    st.info(
        "💡 混频后信号包含 **直流分量** (原基带) + **2fc 高频分量**。"
        "滑动平均 LPF 滤除高频后，恢复波形基本与原始基带重合。"
        "绿点为每符号中间的采样时刻，用于后续星座判决。"
    )

# ============================
# MODULE 6: Demodulation & ML Decision
# ============================
elif module == "📍 6. 解调判决 (ML检测)":
    section_header(
        "📍 步骤6: 解调判决 — 最大似然 (ML) 检测",
        formula=r"\hat{s} = \arg\min_{c \in \mathcal{C}} |r - c|",
    )

    st.markdown(
        """
    从 LPF 输出的 $\\hat{I}(t)$ 和 $\\hat{Q}(t)$ 波形上，在**每个符号的中间时刻**进行采样。
    得到采样值 $(I_n, Q_n)$ 后，计算它到 4 个理想星座点的 **欧氏距离**，
    选择距离最近的点作为判决结果。

    $$d_k = \\sqrt{(I_n - I_k)^2 + (Q_n - Q_k)^2}, \\quad k \\in \\{00,01,10,11\\}$$
    """
    )

    # --- Constellation: ideal + received + decision arrows ---
    st.markdown("### 接收星座图 — 理想点、接收点、判决轨迹")

    fig_cst, ax_cst = plt.subplots(figsize=(8, 8))
    fig_cst.patch.set_facecolor("white")

    # Ideal constellation
    const_arr = np.array(const_pts)
    ax_cst.scatter(
        const_arr.real, const_arr.imag, c="#1f77b4", s=220, zorder=5,
        edgecolors="black", linewidths=0.6, marker="s", label="理想星座点",
    )
    labels_bits = ["00", "01", "10", "11"]
    for idx_pt, pt in enumerate(const_pts):
        ax_cst.annotate(
            labels_bits[idx_pt], (pt.real, pt.imag),
            textcoords="offset points", xytext=(16, 16),
            fontsize=12, fontweight="bold", color="#1f77b4",
        )

    # Decision boundaries
    ax_cst.axhline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.5)
    ax_cst.axvline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.5)

    # Received samples
    rx_real = np.array([s.real for s in rx_decisions])
    rx_imag = np.array([s.imag for s in rx_decisions])
    ax_cst.scatter(
        I_sampled, Q_sampled, c="#ff7f0e", s=120, zorder=4,
        edgecolors="black", linewidths=0.5, marker="o", alpha=0.8,
        label="接收采样点",
    )
    # Arrows from received to decision
    for idx_sym in range(num_symbols):
        ax_cst.annotate(
            "", xy=(rx_real[idx_sym], rx_imag[idx_sym]),
            xytext=(I_sampled[idx_sym], Q_sampled[idx_sym]),
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", alpha=0.6,
                            lw=1.2, connectionstyle="arc3,rad=0"),
        )
        # Label sample index
        ax_cst.annotate(
            str(idx_sym + 1),
            (I_sampled[idx_sym], Q_sampled[idx_sym]),
            textcoords="offset points", xytext=(8, -12),
            fontsize=8, color="#ff7f0e", alpha=0.7,
        )

    ax_cst.set_xlim(-2.2, 2.2)
    ax_cst.set_ylim(-2.2, 2.2)
    ax_cst.set_xlabel("I (同相分量)", fontsize=12)
    ax_cst.set_ylabel("Q (正交分量)", fontsize=12)
    ax_cst.set_title("接收星座图 — ML判决 (箭头指向判决星座点)", fontsize=13, fontweight="bold")
    ax_cst.set_aspect("equal")
    ax_cst.legend(fontsize=9, loc="upper right")
    ax_cst.grid(True, alpha=0.3)
    fig_cst.tight_layout()
    st.pyplot(fig_cst)
    plt.close(fig_cst)

    # Decision detail table
    st.markdown("### 判决详情")

    decision_data = []
    for idx_sym in range(num_symbols):
        tx_bits = f"{bits[2*idx_sym]}{bits[2*idx_sym+1]}"
        rx_pt = complex(I_sampled[idx_sym], Q_sampled[idx_sym])
        decided_pt = rx_decisions[idx_sym]
        decided_bits = f"{rx_bits[2*idx_sym]}{rx_bits[2*idx_sym+1]}"
        dist = abs(rx_pt - decided_pt)
        correct = "✓" if tx_bits == decided_bits else "✗"
        decision_data.append({
            "符号": idx_sym + 1,
            "发送比特": tx_bits,
            "接收 (I,Q)": f"({I_sampled[idx_sym]:+.3f}, {Q_sampled[idx_sym]:+.3f})",
            "判决 (I,Q)": f"({int(decided_pt.real):+d}, {int(decided_pt.imag):+d})",
            "判决比特": decided_bits,
            "距离": f"{dist:.3f}",
            "结果": correct,
        })

    st.dataframe(decision_data, use_container_width=True, hide_index=True)

    st.info(
        "💡 判决边界是 I=0 和 Q=0 两条线，将平面分成 4 个象限。"
        "接收点落在哪个象限，就判决为对应的星座点。"
        "当噪声较大时，接收点可能越过边界 → 误判 → 比特错误。"
    )

# ============================
# MODULE 7: Bit Recovery & BER
# ============================
elif module == "✅ 7. 比特恢复与误码率":
    section_header("✅ 步骤7: 比特恢复与误码率 (BER)")

    st.markdown(
        f"""
    将 ML 判决得到的星座点通过 **逆 Gray 映射** 恢复为比特对，
    再拼接成完整的接收比特流，与原始发送比特逐一对比。
    """
    )

    # --- BER metric ---
    col_ber1, col_ber2, col_ber3 = st.columns(3)
    col_ber1.metric(
        "误比特率 (BER)", f"{ber:.4f}",
        delta=f"{bit_errors} 个错误" if bit_errors > 0 else "无误码",
        delta_color="off" if bit_errors == 0 else "inverse",
    )
    col_ber2.metric("发送总比特数", len(bits))
    col_ber3.metric(
        "噪声 σ", f"{noise_std:.2f}",
        delta="无误码条件" if bit_errors == 0 else None,
    )

    # --- Bit comparison display ---
    st.markdown("### 逐比特对比")

    tx_str = ".".join(str(b) for b in bits)
    rx_str = ".".join(str(b) for b in rx_bits)

    col_tx, col_rx = st.columns(2)
    with col_tx:
        st.markdown("**📤 发送比特:**")
        st.code(tx_str, language=None)

    with col_rx:
        st.markdown("**📥 接收比特:**")
        st.code(rx_str, language=None)

    # Visual bit comparison with color
    st.markdown("**逐位校验:**")

    diff_html = ""
    for idx_b, (bb, rb) in enumerate(zip(bits, rx_bits)):
        if bb == rb:
            diff_html += f'<span style="color:green;font-weight:bold">✓</span>'
        else:
            diff_html += f'<span style="color:red;font-weight:bold">✗</span>'
        if (idx_b + 1) % 8 == 0 and idx_b < len(bits) - 1:
            diff_html += " &nbsp;|&nbsp; "

    st.markdown(f'<p style="font-size:18px;font-family:monospace">{diff_html}</p>',
                unsafe_allow_html=True)

    # --- Symbol-level summary ---
    st.markdown("### 符号级汇总")

    sym_summary = []
    for idx_sym in range(num_symbols):
        tx_pair = f"{bits[2*idx_sym]}{bits[2*idx_sym+1]}"
        rx_pair = f"{rx_bits[2*idx_sym]}{rx_bits[2*idx_sym+1]}"
        ok = tx_pair == rx_pair
        sym_summary.append({
            "符号": f"#{idx_sym + 1}",
            "发送": tx_pair,
            "映射": f"({int(I_vals[idx_sym]):+d}, {int(Q_vals[idx_sym]):+d})",
            "接收": rx_pair,
            "正确": "✓" if ok else "✗",
        })

    st.dataframe(sym_summary, use_container_width=True, hide_index=True)

    # --- Summary judgment ---
    st.markdown("---")
    if bit_errors == 0:
        st.success("🎉 完美传输! 所有比特正确恢复，BER = 0。")
    elif ber < 0.1:
        st.warning(
            f"⚠️ 存在 {bit_errors} 个传输错误 (BER={ber:.4f})。"
            "噪声导致部分接收点越过了判决边界。"
        )
    else:
        st.error(
            f"❌ 高误码率! {bit_errors} 个错误 (BER={ber:.4f})。"
            "噪声过大，建议减小 σ 重新观察。"
        )

# ============================
# FOOTER
# ============================
st.markdown("---")
st.caption(
    "💡 提示: 使用左侧边栏切换不同步骤，调节噪声 σ 和载波频率 f_c 观察效果变化。"
    "|\t  4QAM 通信链路全流程: 比特 → 映射 → 基带 → 上变频 → AWGN → 下变频 → LPF → ML判决 → 恢复"
)
