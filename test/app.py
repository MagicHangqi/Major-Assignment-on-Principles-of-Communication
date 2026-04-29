import io
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.io import wavfile
from scipy.io.wavfile import read as wav_read

fm.fontManager.addfont(
    os.path.join(os.path.dirname(__file__), "fonts", "wqy-microhei.ttc")
)

plt.rcParams["font.sans-serif"] = [
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="调制通信仿真", page_icon="🎵", layout="wide")

# ============================================================================
# CONSTANTS
# ============================================================================
AUDIO_SR = 8000
AUDIO_BITS = 8
PRESET_DURATION = 2.0
MAX_AUDIO_SECONDS = 20
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * AUDIO_SR
FC = 20000
FS = int(FC * 10)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def generate_preset_melody():
    dur = PRESET_DURATION
    total_samples = int(dur * AUDIO_SR)
    t_audio = np.arange(total_samples) / AUDIO_SR
    notes = [262, 294, 330, 349, 392, 440, 494, 523]
    note_dur = dur / len(notes)
    mel = np.zeros(total_samples, dtype=np.float64)
    for i, f in enumerate(notes):
        start = int(i * note_dur * AUDIO_SR)
        end = int((i + 1) * note_dur * AUDIO_SR)
        if end > total_samples:
            end = total_samples
        seg = t_audio[start:end] - t_audio[start]
        env = np.exp(-2 * (seg / (note_dur / 2)))
        mel[start:end] = np.sin(2 * np.pi * f * seg) * env * 0.6
    pcm = np.clip(((mel + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
    return pcm


def pcm_to_bits(pcm):
    bits = []
    for val in pcm:
        for bit_pos in range(7, -1, -1):
            bits.append((val >> bit_pos) & 1)
    return bits


def bits_to_pcm(bits, num_samples):
    pcm = np.zeros(num_samples, dtype=np.uint8)
    for i in range(num_samples):
        val = 0
        for bit_pos in range(8):
            val = (val << 1) | bits[i * 8 + bit_pos]
        pcm[i] = val
    return pcm


def pcm_to_wav_bytes(pcm, sample_rate=AUDIO_SR):
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, pcm)
    buf.seek(0)
    return buf.getvalue()


def section_header(title, formula=None):
    st.markdown("---")
    st.subheader(title)
    if formula:
        st.latex(formula)


# ============================================================================
# SIDEBAR — MAIN PAGE SELECTION
# ============================================================================
st.sidebar.header("📋 选择查看")
page = st.sidebar.radio(
    "",
    [
        "📡 1. 16QAM原理（音频）",
        "📻 2. DSB-SC原理（音频）",
        "🔊 3. 16QAM vs DSB-SC 对比",
    ],
)

PAGE_QAM16 = page.startswith("📡")
PAGE_DSBSC = page.startswith("📻")
PAGE_COMPARE = page.startswith("🔊")

# --- Audio source ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎵 音频源")
audio_source = st.sidebar.radio("音频来源", ["📦 预置旋律 (C大调音阶)", "📤 上传 WAV 文件 (≤20秒)"])
if "上传" in audio_source:
    uploaded_file = st.sidebar.file_uploader("选择 WAV 文件", type=["wav", "wave"])
else:
    uploaded_file = None

# --- Channel parameters ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ 信道参数")
noise_std = st.sidebar.slider("噪声标准差 σ", 0.0, 2.0, 0.15, step=0.05)
seed = st.sidebar.number_input("随机种子", 0, 9999, 42)
st.sidebar.markdown("---")

# ============================================================================
# LOAD AUDIO → PCM → BITS
# ============================================================================
audio_loaded = False
pcm_original = None
audio_duration = None

if "上传" in audio_source:
    if uploaded_file is not None:
        try:
            sr, data = wav_read(uploaded_file)
            if data.ndim > 1:
                data = data[:, 0]
            if sr != AUDIO_SR:
                data = np.interp(
                    np.linspace(0, len(data) - 1, int(len(data) * AUDIO_SR / sr)),
                    np.arange(len(data)), data,
                ).astype(np.float32)
            if len(data) > MAX_AUDIO_SAMPLES:
                data = data[:MAX_AUDIO_SAMPLES]
                st.sidebar.warning(f"⚠️ 音频过长，已截取前 {MAX_AUDIO_SECONDS} 秒")
            if data.dtype != np.uint8:
                dmin, dmax = data.min(), data.max()
                if dmax > dmin:
                    data = np.clip(((data - dmin) / (dmax - dmin) * 255.0), 0, 255)
                else:
                    data = np.full(len(data), 128, dtype=np.uint8)
                data = data.astype(np.uint8)
            pcm_original = data.astype(np.uint8)
            audio_duration = len(pcm_original) / AUDIO_SR
            audio_loaded = True
        except Exception as e:
            st.sidebar.error(f"文件读取失败: {e}")
else:
    pcm_original = generate_preset_melody()
    audio_duration = len(pcm_original) / AUDIO_SR
    audio_loaded = True

if audio_loaded and len(pcm_original) == 0:
    audio_loaded = False

if not audio_loaded:
    st.warning("⚠️ 请先从侧边栏选择预置旋律或上传 WAV 文件")
    st.stop()

# ============================================================================
# COMPUTE BITS
# ============================================================================
bits = pcm_to_bits(pcm_original)
num_bits = len(bits)
num_samples_pcm = len(pcm_original)
total_duration = audio_duration

# ============================================================================
# TIME ARRAY & CARRIER (shared by 16QAM and DSB-SC)
# ============================================================================
t = np.arange(0, total_duration, 1 / FS)
num_samples = min(len(t), int(FS * total_duration))
t = t[:num_samples]

carrier_cos = np.cos(2 * np.pi * FC * t)
carrier_sin = np.sin(2 * np.pi * FC * t)

# ============================================================================
# 16QAM MODULATION CHAIN
# ============================================================================
BITS_PER_SYM = 4
num_symbols = num_bits // BITS_PER_SYM

# 4-PAM Gray-coded lookup (I and Q independently)
PAM4_LABEL_TO_LEVEL = {0b00: -3, 0b01: -1, 0b11: 1, 0b10: 3}
PAM4_LEVEL_TO_LABEL = {v: k for k, v in PAM4_LABEL_TO_LEVEL.items()}
pam4_levels = np.array(sorted(PAM4_LABEL_TO_LEVEL.values()))

# Symbol mapping
I_vals = np.zeros(num_symbols, dtype=np.int8)
Q_vals = np.zeros(num_symbols, dtype=np.int8)
for i in range(num_symbols):
    b4 = bits[i * 4:(i + 1) * 4]
    I_vals[i] = PAM4_LABEL_TO_LEVEL[(b4[0] << 1) | b4[1]]
    Q_vals[i] = PAM4_LABEL_TO_LEVEL[(b4[2] << 1) | b4[3]]

# Baseband (ZOH)
sym_idx = np.clip(
    np.floor(t * num_symbols / total_duration).astype(int), 0, num_symbols - 1,
)
I_bb = I_vals[sym_idx].astype(np.float64)
Q_bb = Q_vals[sym_idx].astype(np.float64)

# Up-conversion
s_rf = I_bb * carrier_cos - Q_bb * carrier_sin

# AWGN
rng = np.random.default_rng(seed)
noise = rng.normal(0, noise_std, num_samples)
s_rf_noisy = s_rf + noise

# Down-conversion + LPF
I_mixed = 2 * s_rf_noisy * carrier_cos
Q_mixed = -2 * s_rf_noisy * carrier_sin

spb = max(2, int(FS * total_duration / num_symbols))
lpf_kernel = np.ones(spb // 2) / (spb // 2)
I_rec = np.convolve(I_mixed, lpf_kernel, mode="same")
Q_rec = np.convolve(Q_mixed, lpf_kernel, mode="same")

# Sampling at symbol centers
sample_times = np.arange(
    total_duration / num_symbols / 2, total_duration, total_duration / num_symbols,
)
sample_idx = np.clip((sample_times * FS).astype(int), 0, num_samples - 1)
I_sampled = I_rec[sample_idx]
Q_sampled = Q_rec[sample_idx]

# ML detection (independent I/Q 4-PAM)
def detect_pam4(x):
    return pam4_levels[np.argmin(np.abs(x - pam4_levels))]

rx_bits = []
for i in range(num_symbols):
    i_det = int(detect_pam4(I_sampled[i]))
    q_det = int(detect_pam4(Q_sampled[i]))
    i_label = PAM4_LEVEL_TO_LABEL[i_det]
    q_label = PAM4_LEVEL_TO_LABEL[q_det]
    rx_bits.extend([(i_label >> 1) & 1, i_label & 1, (q_label >> 1) & 1, q_label & 1])

# BER & Audio reconstruction
bit_errors = sum(1 for b, r in zip(bits, rx_bits) if b != r)
ber = bit_errors / num_bits
pcm_recovered = bits_to_pcm(rx_bits, num_samples_pcm)
qam_wav = pcm_to_wav_bytes(pcm_recovered)

# ============================================================================
# DSB-SC MODULATION CHAIN
# ============================================================================
t_audio = np.arange(num_samples_pcm) / AUDIO_SR
m_vals = pcm_original.astype(np.float64) / 255.0 * 2.0 - 1.0
m_t = np.interp(t, t_audio, m_vals)

s_dsbsc = m_t * carrier_cos
s_dsbsc_noisy = s_dsbsc + noise

dsbsc_mixed = 2.0 * s_dsbsc_noisy * carrier_cos
dsbsc_lpf = np.ones(5) / 5
dsbsc_demod = np.convolve(dsbsc_mixed, dsbsc_lpf, mode="same")
m_hat_raw = dsbsc_demod

dsbsc_sample_idx = np.clip(
    (np.arange(num_samples_pcm) / AUDIO_SR * FS).astype(int), 0, num_samples - 1,
)
m_hat_audio = m_hat_raw[dsbsc_sample_idx]
pcm_dsbsc_recovered = np.clip(
    ((m_hat_audio + 1.0) / 2.0 * 255.0), 0, 255,
).astype(np.uint8)
dsbsc_wav = pcm_to_wav_bytes(pcm_dsbsc_recovered)

dsbsc_mse = np.mean((m_hat_audio - m_vals) ** 2)
dsbsc_sig_pwr = max(np.var(m_vals), 1e-12)
dsbsc_snr_db = 10 * np.log10(dsbsc_sig_pwr / max(dsbsc_mse, 1e-12))

# ============================================================================
# ZOOM WINDOWS
# ============================================================================
Z_SYMBOLS = min(8, num_symbols)
Z_DURATION = Z_SYMBOLS * total_duration / max(num_symbols, 1)
zoom_mask = (t >= 0) & (t < Z_DURATION)
MIX_DURATION = min(0.1 * total_duration, total_duration / max(num_symbols, 1) * 4)
mix_mask = (t >= 0) & (t < MIX_DURATION)
CARRIER_DURATION = min(0.04, total_duration)
carrier_mask = (t >= 0) & (t < CARRIER_DURATION)

# ============================================================================
# SIDEBAR FOOTER
# ============================================================================
st.sidebar.markdown("---")
if PAGE_QAM16:
    st.sidebar.caption(
        f"音频: {audio_duration:.1f}s · 音频SR: {AUDIO_SR} Hz\n"
        f"载波: {FC} Hz · 仿真SR: {FS} Hz · {num_symbols}符号\n"
        f"16QAM · σ = {noise_std:.2f} · BER = {ber:.4f}"
    )
elif PAGE_DSBSC:
    st.sidebar.caption(
        f"音频: {audio_duration:.1f}s · 音频SR: {AUDIO_SR} Hz\n"
        f"载波: {FC} Hz · 仿真SR: {FS} Hz\n"
        f"DSB-SC · σ = {noise_std:.2f} · SNR = {dsbsc_snr_db:.1f} dB"
    )
else:
    st.sidebar.caption(
        f"音频: {audio_duration:.1f}s · 音频SR: {AUDIO_SR} Hz\n"
        f"载波: {FC} Hz · 仿真SR: {FS} Hz · σ = {noise_std:.2f}\n"
        f"16QAM BER = {ber:.4f}  |  DSB-SC SNR = {dsbsc_snr_db:.1f} dB"
    )

# ============================================================================
# MAIN TITLE
# ============================================================================
if PAGE_QAM16:
    st.title("📡 16QAM 数字调制原理 — 音频通信链路仿真")
    st.caption(
        "音频 PCM → 比特 → 16QAM 调制 → 上变频 → AWGN 信道 → "
        "下变频 → LPF → ML 判决 → 比特恢复 → PCM → 音频播放"
    )
elif PAGE_DSBSC:
    st.title("📻 DSB-SC 模拟调制原理 — 音频通信链路仿真")
    st.caption(
        "音频 PCM → m(t) 模拟幅度 → DSB-SC 调制 → AWGN 信道 → "
        "相干解调 → PCM → 音频播放"
    )
else:
    st.title("🔊 16QAM vs DSB-SC — 数字 vs 模拟对比")
    st.caption(
        f"同一段音频，同步经过 16QAM 和 DSB-SC → "
        f"相同载波 ({FC} Hz)、相同噪声 → 听感对比"
    )

# ============================================================================
# ============================================================================
#  16QAM PRINCIPLE PAGE
# ============================================================================
# ============================================================================
if PAGE_QAM16:
    # ------------------------------------------------------------------
    # MODULE 1: Input Bits
    # ------------------------------------------------------------------
    with st.expander("📥 1. 输入比特流 — PCM → 比特分解", expanded=False):
        section_header("📥 步骤1: 输入比特流")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**PCM 音频统计**")
            st.metric("采样率", f"{AUDIO_SR} Hz")
            st.metric("位深", f"{AUDIO_BITS} bit")
            st.metric("时长", f"{audio_duration:.2f} 秒")
            st.metric("PCM 样本数", num_samples_pcm)
        with col_b:
            st.markdown("**比特流统计**")
            st.metric("总比特数", num_bits)
            st.metric("16QAM 符号数", num_symbols)
            st.metric("每符号比特数", "4 bit")

        st.markdown("**PCM 样本 → 比特 示例 (前 4 字节)**")
        if num_samples_pcm >= 4:
            rows = "".join(
                f"PCM[{i}] = {pcm_original[i]:3d} → "
                f"{''.join(str((pcm_original[i]>>bit)&1) for bit in range(7,-1,-1))}\n"
                for i in range(min(4, num_samples_pcm))
            )
            st.code(rows, language=None)

        st.info(
            "💡 每个 8-bit PCM 音频样本被拆分为 8 个比特。"
            f"这 {num_bits} 个比特将分成 {num_symbols} 组 (每组 4 bit) 进行 16QAM 调制。"
        )

    # ------------------------------------------------------------------
    # MODULE 2: Symbol Mapping & Constellation
    # ------------------------------------------------------------------
    with st.expander("📊 2. 符号映射与星座图 — 16QAM Gray编码", expanded=False):
        section_header("📊 步骤2: 符号映射 (16QAM Gray编码)")

        col_tab, col_math = st.columns([1, 2])
        with col_tab:
            st.markdown(
                """
            | I bits | I 电平 | Q bits | Q 电平 |
            |--------|--------|--------|--------|
            | **00** |  −3 | **00** |  −3 |
            | **01** |  −1 | **01** |  −1 |
            | **11** |  +1 | **11** |  +1 |
            | **10** |  +3 | **10** |  +3 |
            """
            )
            st.caption("4-PAM Gray映射 → 4×4 = 16 星座点")

        with col_math:
            st.latex(r"s_k = I_k + j Q_k,\quad I_k, Q_k \in \{\pm 3, \pm 1\}")
            st.latex(r"\text{Gray: 相邻星座点仅差 1 bit}")

        st.markdown("### 16QAM 星座图 与 基带 I/Q 波形")

        fig_bb, (ax_c, ax_w) = plt.subplots(1, 2, figsize=(13, 5))
        fig_bb.patch.set_facecolor("white")

        # Constellation
        const_pts = [(i, q) for i in [-3, -1, 1, 3] for q in [-3, -1, 1, 3]]
        for pt_i, pt_q in const_pts:
            label_i = PAM4_LEVEL_TO_LABEL[pt_i]
            label_q = PAM4_LEVEL_TO_LABEL[pt_q]
            bit_label = f"{label_i>>1}{label_i&1}{label_q>>1}{label_q&1}"
            ax_c.scatter(pt_i, pt_q, c="#1f77b4", s=80, zorder=5, edgecolors="black", linewidths=0.4)
            ax_c.annotate(bit_label, (pt_i, pt_q), textcoords="offset points",
                          xytext=(8, 8), fontsize=6.5, color="#1f77b4", fontweight="bold")
        ax_c.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_c.axvline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_c.set_xlim(-4.5, 4.5)
        ax_c.set_ylim(-4.5, 4.5)
        ax_c.set_xlabel("I (同相分量)", fontsize=10)
        ax_c.set_ylabel("Q (正交分量)", fontsize=10)
        ax_c.set_title("16QAM 星座图", fontsize=12, fontweight="bold")
        ax_c.set_aspect("equal")
        ax_c.grid(True, alpha=0.3)

        # I/Q waveform
        disp_syms = min(Z_SYMBOLS, num_symbols)
        sym_dur = total_duration / num_symbols
        t_wave = np.arange(disp_syms + 1) * sym_dur
        i_wave = np.append(I_vals[:disp_syms], I_vals[disp_syms - 1])
        q_wave = np.append(Q_vals[:disp_syms], Q_vals[disp_syms - 1])
        ax_w.step(t_wave, i_wave, where="post", color="blue", linewidth=1.5, label="I(t)")
        ax_w.step(t_wave, q_wave, where="post", color="red", linewidth=1.5, label="Q(t)")
        ax_w.set_xlabel("时间 (s)", fontsize=10)
        ax_w.set_ylabel("幅度", fontsize=10)
        ax_w.set_title(f"基带 I(t)/Q(t) 波形 (前 {disp_syms} 符号)", fontsize=12, fontweight="bold")
        ax_w.set_ylim(-4.5, 4.5)
        ax_w.set_xlim(0, t_wave[-1])
        ax_w.legend(fontsize=9, loc="upper right")
        ax_w.grid(True, alpha=0.3)
        for i in range(disp_syms):
            b4 = bits[i*4:(i+1)*4]
            bit_label = f"{b4[0]}{b4[1]},{b4[2]}{b4[3]}"
            ax_w.axvline(i * sym_dur, color="gray", linestyle=":", linewidth=0.5)
            ax_w.text((i + 0.5) * sym_dur, 4.0, bit_label, ha="center", fontsize=7, color="gray")

        fig_bb.tight_layout()
        st.pyplot(fig_bb)
        plt.close(fig_bb)

        st.info(
            "💡 16QAM 每个符号携带 4 bit。I 路和 Q 路各自使用 4-PAM (4电平)，"
            "组合成 4×4=16 个星座点。Gray 编码确保相邻点仅差 1 bit。"
        )

    # ------------------------------------------------------------------
    # MODULE 3: Up-conversion
    # ------------------------------------------------------------------
    with st.expander("⬆️ 3. 上变频 (通带调制) — I/Q载波混合", expanded=False):
        section_header(
            "⬆️ 步骤3: 上变频 (Up-conversion)",
            formula=r"s_{\mathrm{RF}}(t) = I(t) \cdot \cos(2\pi f_c t) \;-\; Q(t) \cdot \sin(2\pi f_c t)",
        )

        st.markdown(
            f"""
        基带 I/Q 信号被调制到载波频率 $f_c = {FC}$ Hz 上:
        - **I 路** 调制到 $\\cos(2\\pi f_c t)$ (余弦载波)
        - **Q 路** 调制到 $-\\sin(2\\pi f_c t)$ (正交载波)

        两步乘积相加后得到 **通带射频信号** $s_{{\\mathrm{{RF}}}}(t)$。
        """
        )

        st.markdown("#### 🔍 放大视图 (前几个符号，可见载波细节)")

        t_z = t[zoom_mask]
        I_z = I_bb[zoom_mask]
        Q_z = Q_bb[zoom_mask]
        ccos_z = carrier_cos[zoom_mask]
        csin_z = carrier_sin[zoom_mask]
        srf_z = s_rf[zoom_mask]

        fig_zoom, (ax_i, ax_q, ax_rf) = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
        fig_zoom.patch.set_facecolor("white")

        i_mod = I_z * ccos_z
        ax_i.plot(t_z, i_mod, color="blue", linewidth=0.7, label=r"$I(t)\cos(2\pi f_c t)$")
        ax_i.plot(t_z, I_z, color="blue", linewidth=1.5, alpha=0.35, linestyle="--", label="I(t) 包络")
        ax_i.set_ylabel("I 路", fontsize=9)
        ax_i.legend(fontsize=7, loc="upper right", ncol=2)
        ax_i.grid(True, alpha=0.3)
        ax_i.set_ylim(-4.8, 4.8)

        q_mod = -Q_z * csin_z
        ax_q.plot(t_z, q_mod, color="red", linewidth=0.7, label=r"$-Q(t)\sin(2\pi f_c t)$")
        ax_q.plot(t_z, Q_z, color="red", linewidth=1.5, alpha=0.35, linestyle="--", label="Q(t) 包络")
        ax_q.set_ylabel("Q 路", fontsize=9)
        ax_q.legend(fontsize=7, loc="upper right", ncol=2)
        ax_q.grid(True, alpha=0.3)
        ax_q.set_ylim(-4.8, 4.8)

        ax_rf.plot(t_z, srf_z, color="purple", linewidth=0.8, label=r"$s_{\mathrm{RF}}(t)$")
        ax_rf.plot(t_z, I_z, color="blue", linewidth=1.2, alpha=0.3, linestyle="--", label="I(t) 包络")
        ax_rf.set_xlabel("时间 (s)", fontsize=9)
        ax_rf.set_ylabel("RF", fontsize=9)
        ax_rf.legend(fontsize=7, loc="upper right", ncol=2)
        ax_rf.grid(True, alpha=0.3)
        ax_rf.set_ylim(-8, 8)
        ax_rf.set_title("通带信号 $s_{\\mathrm{RF}}(t)$ — I 路包络叠加可见", fontsize=10, fontweight="bold")

        fig_zoom.tight_layout()
        st.pyplot(fig_zoom)
        plt.close(fig_zoom)

        st.info(
            "💡 I 路调制到 cos 载波，Q 路调制到 −sin 载波 (正交)。"
            "二者相加得到单一通带信号，同时保持 I/Q 正交性以利接收端分离。"
        )

    # ------------------------------------------------------------------
    # MODULE 4: AWGN Channel
    # ------------------------------------------------------------------
    with st.expander("🌊 4. AWGN信道 — 高斯噪声叠加", expanded=False):
        section_header(
            "🌊 步骤4: AWGN信道",
            formula=r"r(t) = s_{\mathrm{RF}}(t) + w(t),\quad w(t) \sim \mathcal{N}(0, \sigma^2)",
        )

        st.markdown(
            f"""
        通带信号通过 **加性高斯白噪声 (AWGN)** 信道。当前噪声标准差 $\\sigma = {noise_std:.2f}$。

        噪声叠加后，信号波形出现随机起伏，星座图上的点将偏离理想位置。
        """
        )

        st.markdown("#### 🔍 放大对比: 理想 vs 含噪")

        t_z = t[zoom_mask]
        srf_z = s_rf[zoom_mask]
        noisy_z = s_rf_noisy[zoom_mask]

        fig_noise, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
        fig_noise.patch.set_facecolor("white")

        ax1.plot(t_z, srf_z, color="blue", linewidth=0.6)
        ax1.set_ylabel("理想", fontsize=9)
        ax1.set_title("理想通带信号 $s_{\\mathrm{RF}}(t)$", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_z, noisy_z, color="#e67e22", linewidth=0.6, alpha=0.9)
        ax2.plot(t_z, srf_z, color="blue", linewidth=0.6, alpha=0.25, label="理想 (半透明)")
        ax2.set_xlabel("时间 (s)", fontsize=9)
        ax2.set_ylabel("含噪", fontsize=9)
        ax2.set_title(f"含噪信号 ($\\sigma={noise_std:.2f}$)", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.3)

        for ax in [ax1, ax2]:
            ax.set_ylim(-8, 8)

        fig_noise.tight_layout()
        st.pyplot(fig_noise)
        plt.close(fig_noise)

        st.info(
            f"💡 噪声 σ = {noise_std:.2f} 叠加到通带信号上。"
            "数字调制的优势在于: 只要噪声不跨越判决边界，比特信息完好无损。"
        )

    # ------------------------------------------------------------------
    # MODULE 5: Down-conversion & LPF
    # ------------------------------------------------------------------
    with st.expander("⬇️ 5. 下变频与低通滤波 — 相干解调前端", expanded=False):
        section_header(
            "⬇️ 步骤5: 下变频与低通滤波",
            formula=r"\hat{I}(t) = \mathrm{LPF}\big\{2 r(t) \cos(2\pi f_c t)\big\},\;"
                    r"\hat{Q}(t) = \mathrm{LPF}\big\{-2 r(t) \sin(2\pi f_c t)\big\}",
        )

        st.markdown(
            f"""
        接收端使用与发送端**同频同相**的本地载波进行混频 (相干解调)，再用低通滤波器去除高频。

        **滤波器**: 滑动平均 LPF (窗口 = {int(spb // 2)} 样点, 采样率 {FS} Hz)
        """
        )

        st.markdown("#### 🔬 混频后波形 (LPF前 — 可见 2fc 高频纹波)")
        t_mz = t[mix_mask]
        I_mz = I_mixed[mix_mask]
        Q_mz = Q_mixed[mix_mask]

        fig_mix, (ax_im, ax_qm) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
        fig_mix.patch.set_facecolor("white")

        ax_im.plot(t_mz, I_mz, color="blue", linewidth=0.6, label=r"$2 r(t) \cos(2\pi f_c t)$")
        ax_im.plot(t_mz, I_bb[mix_mask], color="blue", linewidth=1.5, alpha=0.35,
                   linestyle="--", label="理想 I(t) 包络")
        ax_im.set_ylabel("I 混频输出", fontsize=9)
        ax_im.legend(fontsize=7, loc="upper right")
        ax_im.grid(True, alpha=0.3)
        ax_im.set_title("I 路混频输出 (含基带 + 2fc 高频)", fontsize=11, fontweight="bold")

        ax_qm.plot(t_mz, Q_mz, color="red", linewidth=0.6, label=r"$-2 r(t) \sin(2\pi f_c t)$")
        ax_qm.plot(t_mz, Q_bb[mix_mask], color="red", linewidth=1.5, alpha=0.35,
                   linestyle="--", label="理想 Q(t) 包络")
        ax_qm.set_xlabel("时间 (s)", fontsize=9)
        ax_qm.set_ylabel("Q 混频输出", fontsize=9)
        ax_qm.legend(fontsize=7, loc="upper right")
        ax_qm.grid(True, alpha=0.3)
        ax_qm.set_title("Q 路混频输出", fontsize=11, fontweight="bold")

        fig_mix.tight_layout()
        st.pyplot(fig_mix)
        plt.close(fig_mix)

        st.markdown("#### 🎯 LPF 输出 — 恢复基带 vs 原始基带对比")

        t_z = t[zoom_mask]
        I_rec_z = I_rec[zoom_mask]
        I_bb_z = I_bb[zoom_mask]
        Q_rec_z = Q_rec[zoom_mask]
        Q_bb_z = Q_bb[zoom_mask]

        fig_lpf, (ax_ir, ax_qr) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
        fig_lpf.patch.set_facecolor("white")

        ax_ir.plot(t_z, I_rec_z, color="blue", linewidth=1.0, label=r"LPF输出 $\hat{I}(t)$")
        ax_ir.plot(t_z, I_bb_z, color="green", linewidth=1.2, alpha=0.4,
                   linestyle="--", label="原始 I(t)")
        ax_ir.set_ylabel("I 幅度", fontsize=9)
        ax_ir.legend(fontsize=7, loc="upper right")
        ax_ir.grid(True, alpha=0.3)
        ax_ir.set_title("LPF 后恢复的 Î(t) vs 原始 I(t)", fontsize=11, fontweight="bold")

        ax_qr.plot(t_z, Q_rec_z, color="red", linewidth=1.0, label=r"LPF输出 $\hat{Q}(t)$")
        ax_qr.plot(t_z, Q_bb_z, color="green", linewidth=1.2, alpha=0.4,
                   linestyle="--", label="原始 Q(t)")
        ax_qr.set_xlabel("时间 (s)", fontsize=9)
        ax_qr.set_ylabel("Q 幅度", fontsize=9)
        ax_qr.legend(fontsize=7, loc="upper right")
        ax_qr.grid(True, alpha=0.3)
        ax_qr.set_title("LPF 后恢复的 Q̂(t) vs 原始 Q(t)", fontsize=11, fontweight="bold")

        fig_lpf.tight_layout()
        st.pyplot(fig_lpf)
        plt.close(fig_lpf)

        st.info(
            f"💡 滑动平均 LPF (窗口 {int(spb // 2)} 样点) 有效滤除了 2fc={2*FC} Hz 高频分量，"
            "恢复出基带 Î(t) 和 Q̂(t)。"
        )

    # ------------------------------------------------------------------
    # MODULE 6: ML Detection
    # ------------------------------------------------------------------
    with st.expander("📍 6. 解调判决 (ML检测) — 星座点判定", expanded=False):
        section_header(
            "📍 步骤6: 解调判决 — 最大似然 (ML) 检测",
            formula=r"\hat{s}_k = \arg\min_{c \in \mathcal{C}} |r_k - c|",
        )

        st.markdown(
            f"""
        在每个符号周期中心采样，得到 {num_symbols} 个接收点 $(\\hat{{I}}_k, \\hat{{Q}}_k)$。
        对每个点，在 16QAM 星座中找到最近的星座点——**ML 判决**。

        由于 I/Q 正交且独立，判决可拆分为两个独立的 4-PAM 检测。
        """
        )

        st.markdown("#### 🎯 接收星座图 (含判决边界)")

        fig_det, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(13, 5))

        # Raw sampled scatter
        ax_s.scatter(I_sampled, Q_sampled, c="purple", s=12, alpha=0.6, edgecolors="none")
        ax_s.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_s.axvline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        for lvl in [-2, 0, 2]:
            ax_s.axhline(lvl, color="orange", linestyle=":", linewidth=0.4, alpha=0.4)
            ax_s.axvline(lvl, color="orange", linestyle=":", linewidth=0.4, alpha=0.4)
        ax_s.set_xlabel("I", fontsize=10)
        ax_s.set_ylabel("Q", fontsize=10)
        ax_s.set_title(f"接收采样点 (σ={noise_std:.2f})", fontsize=11, fontweight="bold")
        ax_s.set_aspect("equal")
        ax_s.grid(True, alpha=0.3)
        ax_s.set_xlim(-5, 5)
        ax_s.set_ylim(-5, 5)

        # Detected constellation
        const_pts = [(i, q) for i in [-3, -1, 1, 3] for q in [-3, -1, 1, 3]]
        ax_d.scatter(*zip(*const_pts), c="#1f77b4", s=60, zorder=5,
                     edgecolors="black", linewidths=0.5)
        ax_d.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_d.axvline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_d.set_xlabel("I", fontsize=10)
        ax_d.set_ylabel("Q", fontsize=10)
        ax_d.set_title("16QAM 判决星座点", fontsize=11, fontweight="bold")
        ax_d.set_aspect("equal")
        ax_d.grid(True, alpha=0.3)
        ax_d.set_xlim(-5, 5)
        ax_d.set_ylim(-5, 5)

        fig_det.tight_layout()
        st.pyplot(fig_det)
        plt.close(fig_det)

        st.markdown("#### 📊 4-PAM 判决区域 (I路示例)")
        fig_pam, ax_pam = plt.subplots(figsize=(10, 2.5))
        fig_pam.patch.set_facecolor("white")
        ax_pam.scatter(I_sampled[:min(80, num_symbols)],
                       np.zeros(min(80, num_symbols)), c="purple", s=15, alpha=0.5)
        for bnd in [-2, 0, 2]:
            ax_pam.axvline(bnd, color="orange", linestyle="--", linewidth=1.0)
        for lvl, lbl in zip([-3, -1, 1, 3], ["00", "01", "11", "10"]):
            ax_pam.axvline(lvl, color="#1f77b4", linestyle="-", linewidth=2, alpha=0.4)
            ax_pam.text(lvl, 0.03, lbl, ha="center", fontsize=10, fontweight="bold", color="#1f77b4")
        ax_pam.set_xlabel("I 幅度", fontsize=10)
        ax_pam.set_xlim(-5, 5)
        ax_pam.set_ylim(-0.1, 0.1)
        ax_pam.set_title("I 路 4-PAM 判决边界 (I∈{−3,−1,+1,+3})", fontsize=11, fontweight="bold")
        ax_pam.set_yticks([])
        ax_pam.grid(True, alpha=0.3, axis="x")
        fig_pam.tight_layout()
        st.pyplot(fig_pam)
        plt.close(fig_pam)

        st.info(
            "💡 虚线为判决边界 (−2, 0, +2)。接收值落入哪个区间，就判为对应的星座电平。"
            "Q 路同理。独立判决使得 ML 检测高效。"
        )

    # ------------------------------------------------------------------
    # MODULE 7: Bit Recovery & BER
    # ------------------------------------------------------------------
    with st.expander("✅ 7. 比特恢复与误码率 — BER计算", expanded=False):
        section_header("✅ 步骤7: 比特恢复与误码率 (BER)")

        st.markdown(
            f"""
        判决得到的星座点通过 Gray 反查表恢复为 4 bit，再拼接成完整的接收比特流，
        与原始发送比特逐一对比。
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("发送比特数", num_bits)
            st.metric("16QAM 符号数", num_symbols)
        with col2:
            st.metric("比特错误数", bit_errors)
            st.metric("BER", f"{ber:.6f}")

        st.markdown("### 逐比特对比 (前 40 bit)")
        compare_rows = []
        for i in range(min(40, num_bits)):
            marker = " ✓" if bits[i] == rx_bits[i] else " ✗"
            compare_rows.append(
                f"{i:3d}: 发送={bits[i]} 接收={rx_bits[i]}{marker}"
            )
        st.code("\n".join(compare_rows), language=None)

        if bit_errors == 0:
            st.success(f"🎉 **BER = 0** — 完美恢复! 噪声未跨越判决边界。")
        elif ber < 0.01:
            st.warning(f"⚠️ BER = {ber:.4f} — 轻微误码")
        else:
            st.error(f"❌ BER = {ber:.4f} — 误码较多")

        st.info(
            "💡 **门限效应**: 数字调制的比特错误率在噪声达到判决边界之前保持极低，"
            "一旦跨越边界则迅速恶化。这是数字通信的\"悬崖效应\"。"
        )

    # ------------------------------------------------------------------
    # MODULE 8: Audio Listening
    # ------------------------------------------------------------------
    with st.expander("🔊 8. 音频听感对比 — 16QAM原始 vs 解调", expanded=False):
        section_header("🔊 步骤8: 音频听感对比 — 原始 vs 解调后")

        st.markdown(
            f"""
        将恢复的 {num_bits} 个比特重新组装为 PCM 音频样本，对比原始音频和经过
        **16QAM 调制 → AWGN 信道 → 解调** 之后的音频。
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📤 原始音频")
            st.audio(pcm_to_wav_bytes(pcm_original), format="audio/wav")
            st.caption("无调制 · 原始 PCM")
        with col2:
            st.markdown("### 📡 16QAM 解调后")
            st.audio(qam_wav, format="audio/wav")
            if bit_errors == 0:
                st.success("完美恢复 😊")
            else:
                st.error(f"BER = {ber:.4f}")

        st.markdown("### 📊 波形对比")

        fig_wcmp, axes = plt.subplots(2, 1, figsize=(13, 4.5), sharex=True)
        fig_wcmp.patch.set_facecolor("white")
        t_audio_plt = np.arange(num_samples_pcm) / AUDIO_SR

        axes[0].plot(t_audio_plt, pcm_original.astype(int), color="blue", linewidth=0.5)
        axes[0].set_ylabel("原始 PCM", fontsize=9)
        axes[0].set_title("原始音频", fontsize=11, fontweight="bold")
        axes[0].set_ylim(0, 255)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_audio_plt, pcm_recovered.astype(int), color="purple", linewidth=0.5)
        axes[1].set_xlabel("时间 (s)", fontsize=9)
        axes[1].set_ylabel("恢复 PCM", fontsize=9)
        axes[1].set_title(f"16QAM 解调 (BER={ber:.4f})", fontsize=11, fontweight="bold")
        axes[1].set_ylim(0, 255)
        axes[1].grid(True, alpha=0.3)

        fig_wcmp.tight_layout()
        st.pyplot(fig_wcmp)
        plt.close(fig_wcmp)

        st.info(
            f"💡 调节噪声 σ 观察: σ≈0 → BER=0 完美; "
            f"σ≈0.5~1.0 → 少量误码; σ>1.5 → 严重失真。"
        )

# ============================================================================
# ============================================================================
#  DSB-SC PRINCIPLE PAGE
# ============================================================================
# ============================================================================
elif PAGE_DSBSC:
    # ------------------------------------------------------------------
    # DSB-SC MODULE 1: Input Audio & m(t)
    # ------------------------------------------------------------------
    with st.expander("📥 1. 输入音频与调制信号 m(t) — PCM→调制信号", expanded=False):
        section_header(
            "📥 步骤1: 输入音频与调制信号 m(t)",
            formula=r"m(t) = 2 \cdot \frac{\mathrm{PCM}(t)}{255} - 1,\quad m(t) \in [-1, 1]",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**PCM 音频统计**")
            st.metric("采样率", f"{AUDIO_SR} Hz")
            st.metric("位深", f"{AUDIO_BITS} bit")
            st.metric("时长", f"{audio_duration:.2f} 秒")
            st.metric("PCM 样本数", num_samples_pcm)
        with col_b:
            st.markdown("**调制参数**")
            st.metric("载波频率", f"{FC} Hz")
            st.metric("采样率 (仿真)", f"{FS} Hz")
            st.metric("调制方式", "DSB-SC")
            st.metric("载波抑制", "是 (无直流)")

        st.markdown("#### m(t) 波形 (前 0.1 秒)")
        t_short = t_audio[:int(0.1 * AUDIO_SR)]
        m_short = m_vals[:len(t_short)]

        fig_m, ax_m = plt.subplots(figsize=(13, 2.5))
        fig_m.patch.set_facecolor("white")
        ax_m.plot(t_short, m_short, color="blue", linewidth=0.7)
        ax_m.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax_m.set_xlabel("时间 (s)", fontsize=10)
        ax_m.set_ylabel("m(t)", fontsize=10)
        ax_m.set_title("调制信号 m(t) ∈ [−1, +1]", fontsize=12, fontweight="bold")
        ax_m.grid(True, alpha=0.3)
        fig_m.tight_layout()
        st.pyplot(fig_m)
        plt.close(fig_m)

        st.info(
            "💡 m(t) 是归一化到 [−1, +1] 的音频幅度。DSB-SC 直接用它调制载波，"
            "没有直流偏置——这与 AM 不同 (AM 需要 1+m(t) 保证包络非负)。"
        )

    # ------------------------------------------------------------------
    # DSB-SC MODULE 2: DSB-SC Modulation
    # ------------------------------------------------------------------
    with st.expander("📻 2. DSB-SC调制 — 抑制载波双边带", expanded=False):
        section_header(
            "📻 步骤2: DSB-SC 调制",
            formula=r"s_{\mathrm{DSB}}(t) = m(t) \cdot \cos(2\pi f_c t)",
        )

        st.markdown(
            f"""
        **DSB-SC (Double Sideband Suppressed Carrier)** 将消息 m(t) 直接乘以载波:
        
        $$s_{{\\mathrm{{DSB}}}}(t) = m(t) \\cdot \\cos(2\\pi f_c t)$$
        
        注意: **没有** 1+ 项，载波被完全抑制。与 AM 的关键区别:
        - AM: $[1+m(t)]\\cos(2\\pi f_c t)$ — 载波消耗大量功率
        - DSB-SC: $m(t)\\cos(2\\pi f_c t)$ — 全部功率用于信息传输
        
        代价是包络不再反映 m(t) 形状，接收端必须用相干解调。
        """
        )

        st.markdown("#### 🔍 调制信号对比 (前几个符号周期)")

        t_z = t[zoom_mask]
        m_tz = m_t[zoom_mask]
        s_dsbsc_z = s_dsbsc[zoom_mask]
        carrier_z = carrier_cos[zoom_mask]

        fig_mod, (ax_m, ax_s) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
        fig_mod.patch.set_facecolor("white")

        ax_m.plot(t_z, m_tz, color="blue", linewidth=1.0, label="m(t)")
        ax_m.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax_m.set_ylabel("m(t)", fontsize=9)
        ax_m.set_title("调制信号 m(t)", fontsize=11, fontweight="bold")
        ax_m.legend(fontsize=8)
        ax_m.grid(True, alpha=0.3)

        ax_s.plot(t_z, s_dsbsc_z, color="purple", linewidth=0.7, label=r"$m(t)\cos(2\pi f_c t)$")
        ax_s.plot(t_z, m_tz, color="blue", linewidth=1.2, alpha=0.35,
                  linestyle="--", label="m(t) 包络")
        ax_s.plot(t_z, -m_tz, color="blue", linewidth=1.2, alpha=0.35, linestyle="--")
        ax_s.set_xlabel("时间 (s)", fontsize=9)
        ax_s.set_ylabel("已调信号", fontsize=9)
        ax_s.set_title(f"DSB-SC 已调信号 ($f_c={FC}$ Hz)", fontsize=11, fontweight="bold")
        ax_s.legend(fontsize=8, loc="upper right")
        ax_s.grid(True, alpha=0.3)

        fig_mod.tight_layout()
        st.pyplot(fig_mod)
        plt.close(fig_mod)

        st.info(
            "💡 图中可见 m(t) 作为包络 (±m(t))。当 m(t) 过零时已调信号相位翻转 180°。"
            "这与 AM 截然不同——AM 的包络始终 ≥0。"
        )

    # ------------------------------------------------------------------
    # DSB-SC MODULE 3: AWGN Channel
    # ------------------------------------------------------------------
    with st.expander("🌊 3. AWGN信道 — 高斯噪声叠加", expanded=False):
        section_header(
            "🌊 步骤3: AWGN信道",
            formula=r"r(t) = s_{\mathrm{DSB}}(t) + w(t),\quad w(t) \sim \mathcal{N}(0, \sigma^2)",
        )

        st.markdown(
            f"""
        DSB-SC 信号通过 AWGN 信道，叠加高斯白噪声 ($\\sigma = {noise_std:.2f}$)。
        噪声均匀分布在整个频谱，影响接收端解调质量。
        """
        )

        st.markdown("#### 🔍 放大对比: 理想 DSB-SC vs 含噪 DSB-SC")

        t_z = t[zoom_mask]
        s_dsbsc_z = s_dsbsc[zoom_mask]
        noisy_z = s_dsbsc_noisy[zoom_mask]

        fig_n, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
        fig_n.patch.set_facecolor("white")

        ax1.plot(t_z, s_dsbsc_z, color="purple", linewidth=0.6)
        ax1.set_ylabel("理想", fontsize=9)
        ax1.set_title("理想 DSB-SC 信号", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_z, noisy_z, color="#e67e22", linewidth=0.6, alpha=0.9)
        ax2.plot(t_z, s_dsbsc_z, color="purple", linewidth=0.6, alpha=0.25,
                 label="理想 (半透明)")
        ax2.set_xlabel("时间 (s)", fontsize=9)
        ax2.set_ylabel("含噪", fontsize=9)
        ax2.set_title(f"含噪 DSB-SC ($\\sigma={noise_std:.2f}$)", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig_n.tight_layout()
        st.pyplot(fig_n)
        plt.close(fig_n)

        st.info(
            f"💡 噪声 σ = {noise_std:.2f} 叠加后信号波形出现随机起伏。"
            "模拟调制对噪声的响应是渐进的——噪声越大，底噪越强，但信号不会完全丢失。"
        )

    # ------------------------------------------------------------------
    # DSB-SC MODULE 4: Coherent Demodulation & Recovery
    # ------------------------------------------------------------------
    with st.expander("⬇️ 4. 相干解调与恢复 — 混频 → LPF → m̂(t)", expanded=False):
        section_header(
            "⬇️ 步骤4: 相干解调与 m(t) 恢复",
            formula=r"\hat{m}(t) = \mathrm{LPF}\big\{2 r(t) \cos(2\pi f_c t)\big\}",
        )

        st.markdown(
            f"""
        接收端用同频同相载波与接收信号混频:
        
        $$2 \\cdot r(t) \\cdot \\cos(2\\pi f_c t) = 2 m(t)\\cos^2(2\\pi f_c t) + \\text{{噪声项}}$$
        
        利用 $2\\cos^2(x) = 1 + \\cos(2x)$:
        
        $$= m(t) + m(t)\\cos(4\\pi f_c t) + \\cdots$$
        
        **关键**: 混频后直接得到 m(t) 基带分量 + 2fc 高频。无需去直流!
        经 LPF 滤除 2fc 项即恢复 $\\hat{{m}}(t)$。

        **滤波器**: 滑动平均 (5 样点, 精确抑制 2fc={2*FC} Hz)
        """
        )

        st.markdown("#### 🔬 混频后 (LPF前) — 可见 2fc 高频纹波")
        t_mz = t[mix_mask]
        dsbsc_mixed_z = dsbsc_mixed[mix_mask]
        m_t_mz = m_t[mix_mask]

        fig_mix, ax_mix = plt.subplots(figsize=(13, 3.5))
        fig_mix.patch.set_facecolor("white")
        ax_mix.plot(t_mz, dsbsc_mixed_z, color="#e67e22", linewidth=0.7,
                    label=r"$2 r(t) \cos(2\pi f_c t)$")
        ax_mix.plot(t_mz, m_t_mz, color="green", linewidth=1.5, alpha=0.5,
                    linestyle="--", label="理想 m(t)")
        ax_mix.set_xlabel("时间 (s)", fontsize=10)
        ax_mix.set_ylabel("混频输出", fontsize=10)
        ax_mix.legend(fontsize=8, loc="upper right")
        ax_mix.grid(True, alpha=0.3)
        ax_mix.set_ylim(-3.5, 3.5)
        ax_mix.set_title(
            f"混频输出 (含 m(t) 基带 + 2fc={2*FC} Hz 高频)",
            fontsize=12, fontweight="bold",
        )
        fig_mix.tight_layout()
        st.pyplot(fig_mix)
        plt.close(fig_mix)

        st.markdown("#### 🎯 LPF 输出 — 恢复 m̂(t) vs 原始 m(t)")
        fig_lpf, ax_lpf = plt.subplots(figsize=(13, 3.5))
        fig_lpf.patch.set_facecolor("white")

        t_z = t[zoom_mask]
        dsbsc_demod_z = dsbsc_demod[zoom_mask]
        m_t_z = m_t[zoom_mask]

        ax_lpf.plot(t_z, dsbsc_demod_z, color="#e67e22", linewidth=1.0,
                    label=r"恢复 $\hat{m}(t)$")
        ax_lpf.plot(t_z, m_t_z, color="green", linewidth=1.2, alpha=0.45,
                    linestyle="--", label="原始 m(t)")
        ax_lpf.set_xlabel("时间 (s)", fontsize=10)
        ax_lpf.set_ylabel("幅度", fontsize=10)
        ax_lpf.legend(fontsize=8, loc="upper right")
        ax_lpf.grid(True, alpha=0.3)
        ax_lpf.set_ylim(-1.5, 1.5)
        ax_lpf.set_title(
            "LPF 输出: 恢复的 m̂(t) vs 原始 m(t) (无需去直流)",
            fontsize=12, fontweight="bold",
        )
        fig_lpf.tight_layout()
        st.pyplot(fig_lpf)
        plt.close(fig_lpf)

        st.info(
            "💡 与 AM 不同，DSB-SC 混频后直接得到 m(t) 而非 1+m(t)，"
            "所以**无需去直流**。LPF 输出就是最终恢复信号。"
            "这是 DSB-SC 解调比 AM 更简洁的原因之一。"
        )

    # ------------------------------------------------------------------
    # DSB-SC MODULE 5: Audio Listening
    # ------------------------------------------------------------------
    with st.expander("🔊 5. 音频听感对比 — DSB-SC解调效果", expanded=False):
        section_header("🔊 步骤5: 音频听感对比 — DSB-SC 解调效果")

        st.markdown(
            f"""
        将恢复的 $\\hat{{m}}(t)$ 重新映射为 PCM 音频样本，
        对比原始音频和经过 **DSB-SC 调制 → AWGN 信道 → 相干解调** 之后的音频。
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📤 原始音频")
            st.audio(pcm_to_wav_bytes(pcm_original), format="audio/wav")
            st.caption("无调制 · 原始 PCM")
        with col2:
            st.markdown("### 📻 DSB-SC 解调后")
            st.audio(dsbsc_wav, format="audio/wav")
            if dsbsc_snr_db > 30:
                st.success(f"SNR = {dsbsc_snr_db:.1f} dB — 高质量")
            elif dsbsc_snr_db > 15:
                st.warning(f"SNR = {dsbsc_snr_db:.1f} dB — 中等质量")
            else:
                st.error(f"SNR = {dsbsc_snr_db:.1f} dB — 低质量")

        st.markdown("### 📊 波形对比")

        fig_wcmp2, axes = plt.subplots(2, 1, figsize=(13, 4.5), sharex=True)
        fig_wcmp2.patch.set_facecolor("white")
        t_audio_plt = np.arange(num_samples_pcm) / AUDIO_SR

        axes[0].plot(t_audio_plt, pcm_original.astype(int), color="blue", linewidth=0.5)
        axes[0].set_ylabel("原始 PCM", fontsize=9)
        axes[0].set_title("原始音频", fontsize=11, fontweight="bold")
        axes[0].set_ylim(0, 255)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_audio_plt, pcm_dsbsc_recovered.astype(int),
                     color="#e67e22", linewidth=0.5)
        axes[1].set_xlabel("时间 (s)", fontsize=9)
        axes[1].set_ylabel("DSB-SC 恢复", fontsize=9)
        axes[1].set_title(
            f"DSB-SC 解调 (SNR={dsbsc_snr_db:.1f} dB)", fontsize=11, fontweight="bold",
        )
        axes[1].set_ylim(0, 255)
        axes[1].grid(True, alpha=0.3)

        fig_wcmp2.tight_layout()
        st.pyplot(fig_wcmp2)
        plt.close(fig_wcmp2)

        st.info(
            "💡 DSB-SC 对噪声的退化是**渐进的**: SNR 随 σ 增大而平滑下降，"
            "总能听到一些旋律痕迹。与 16QAM 的「悬崖效应」形成有趣对比。"
        )

# ============================================================================
# ============================================================================
#  COMPARE PAGE: 16QAM vs DSB-SC
# ============================================================================
# ============================================================================
else:
    section_header("🔊 16QAM vs DSB-SC — 数字与模拟对比")

    st.markdown(
        f"""
    同一段音频，同步经过 **16QAM 数字调制** 和 **DSB-SC 模拟调制**。
    
    两者使用完全相同的载波 ($f_c = {FC}$ Hz) 和相同的高斯噪声 ($\\sigma = {noise_std:.2f}$)。
    **公平对比**: 相同载波、相同噪声。16QAM 带宽约 16 kHz，DSB-SC 带宽约 8 kHz——前者占用 2 倍带宽但凭借判决边界在低噪声下可实现完美恢复。这是数字调制以带宽换质量的经典体现。
    
    点击下方播放器试听三种音频，调节侧边栏的 **噪声标准差 σ** 来感受不同噪声水平下的效果差异。
    """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📤 原始音频")
        st.audio(pcm_to_wav_bytes(pcm_original), format="audio/wav")
        st.caption(f"无噪声 · {audio_duration:.1f}s")

    with col2:
        st.markdown("### 📡 16QAM 解调")
        st.audio(qam_wav, format="audio/wav")
        if bit_errors == 0:
            st.success(f"BER = 0 — 完美恢复")
        elif ber < 0.01:
            st.warning(f"BER = {ber:.4f} — 轻微失真")
        else:
            st.error(f"BER = {ber:.4f} — 误码较多")

    with col3:
        st.markdown("### 📻 DSB-SC 解调")
        st.audio(dsbsc_wav, format="audio/wav")
        if dsbsc_snr_db > 30:
            st.success(f"SNR = {dsbsc_snr_db:.1f} dB")
        elif dsbsc_snr_db > 15:
            st.warning(f"SNR = {dsbsc_snr_db:.1f} dB")
        else:
            st.error(f"SNR = {dsbsc_snr_db:.1f} dB")

    st.markdown("---")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("噪声 σ", f"{noise_std:.2f}")
    col_m2.metric("16QAM BER", f"{ber:.4f}")
    col_m3.metric("DSB-SC SNR", f"{dsbsc_snr_db:.1f} dB")

    st.markdown("---")
    st.markdown("### 📊 音频波形对比")

    fig_wcmp, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    fig_wcmp.patch.set_facecolor("white")
    t_audio_plt = np.arange(num_samples_pcm) / AUDIO_SR

    axes[0].plot(t_audio_plt, pcm_original.astype(int), color="blue", linewidth=0.5)
    axes[0].set_ylabel("原始 PCM", fontsize=9)
    axes[0].set_title("原始音频", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 255)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_audio_plt, pcm_recovered.astype(int), color="purple", linewidth=0.5)
    axes[1].set_ylabel("16QAM 恢复", fontsize=9)
    axes[1].set_title(f"16QAM 解调 (BER={ber:.4f})", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 255)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_audio_plt, pcm_dsbsc_recovered.astype(int),
                 color="#e67e22", linewidth=0.5)
    axes[2].set_xlabel("时间 (s)", fontsize=9)
    axes[2].set_ylabel("DSB-SC 恢复", fontsize=9)
    axes[2].set_title(
        f"DSB-SC 解调 (SNR={dsbsc_snr_db:.1f} dB)", fontsize=11, fontweight="bold",
    )
    axes[2].set_ylim(0, 255)
    axes[2].grid(True, alpha=0.3)

    fig_wcmp.tight_layout()
    st.pyplot(fig_wcmp)
    plt.close(fig_wcmp)

    st.markdown("---")
    st.markdown("### 🧪 抗干扰行为总结")

    if noise_std < 0.01:
        st.success("**σ≈0**: 两者均完美。DSB-SC 和 16QAM 都能无损恢复。")
    elif noise_std < 0.4:
        st.info(
            f"**σ={noise_std:.2f} (低噪声)**: DSB-SC 已出现轻微底噪 (SNR={dsbsc_snr_db:.0f} dB)，"
            f"16QAM 通过判决边界保护保持完美 (BER={ber:.4f})。"
            "**数字调制展示了门限效应前的优势**。"
        )
    elif noise_std < 1.0:
        st.warning(
            f"**σ={noise_std:.2f} (中等噪声)**: DSB-SC 嘶嘶声明显但仍能辨认旋律 "
            f"(SNR={dsbsc_snr_db:.0f} dB)。"
            f"16QAM 开始出现比特错误 (BER={ber:.4f})，产生咔嚓声。"
        )
    else:
        st.error(
            f"**σ={noise_std:.2f} (高噪声)**: DSB-SC 仍能听到模糊旋律痕迹 "
            f"(SNR={dsbsc_snr_db:.0f} dB)。"
            f"16QAM BER={ber:.4f}，音频几乎崩溃。"
            "**模拟调制的渐进退化优势在高噪声下显现**。"
        )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(
    "💡 提示: 使用左侧边栏切换查看页面、调节噪声 σ 观察效果变化。"
    " | 调制通信原理可视化: 16QAM 数字调制 | DSB-SC 模拟调制 | 噪声对比"
)
