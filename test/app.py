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
NUM_SUBCARRIERS = 4
BITS_PER_SYM_16QAM = 4

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
        "📡 1. 4QAM原理（音频）",
        "📻 2. DSB-SC原理（音频）",
        "📡 3. FM原理（音频）",
        "🔊 4. 4QAM vs DSB-SC vs FM 对比",
        "📡 5. 16QAM vs OFDM 多径",
    ],
)

PAGE_QAM = page.startswith("📡") and "4QAM" in page
PAGE_DSBSC = page.startswith("📻")
PAGE_FM = page.startswith("📡") and "FM" in page
PAGE_COMPARE = page.startswith("🔊")
PAGE_16QAM_OFDM = page.startswith("📡") and "16QAM vs OFDM" in page

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
noise_std = st.sidebar.slider("噪声标准差 σ", 0.0, 1.0, 0.10, step=0.02)
seed = st.sidebar.number_input("随机种子", 0, 9999, 42)
st.sidebar.markdown("---")

if PAGE_16QAM_OFDM:
    st.sidebar.header("🌊 多径信道")
    multipath_delay = st.sidebar.slider(
        "多径延迟 (仿真采样点)", 0, 100, 10, step=1,
        help="第二径相对于主径的延迟"
    )
    multipath_atten = st.sidebar.slider(
        "第二径衰减", 0.0, 1.0, 0.6, step=0.05,
        help="第二径幅度衰减系数, 1=等强, 0=无多径"
    )
    st.sidebar.markdown("---")
else:
    multipath_delay = 10
    multipath_atten = 0.6



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
# TIME ARRAY & CARRIER (shared by 4QAM and DSB-SC)
# ============================================================================
t = np.arange(0, total_duration, 1 / FS)
num_samples = min(len(t), int(FS * total_duration))
t = t[:num_samples]

carrier_cos = np.cos(2 * np.pi * FC * t)
carrier_sin = np.sin(2 * np.pi * FC * t)

# ============================================================================
# 4QAM (QPSK) MODULATION CHAIN
# ============================================================================
BITS_PER_SYM = 2
num_symbols = num_bits // BITS_PER_SYM

# Symbol mapping: bit 0 → I, bit 1 → Q  (0 → −1, 1 → +1)
I_vals = np.zeros(num_symbols, dtype=np.int32)
Q_vals = np.zeros(num_symbols, dtype=np.int32)
for i in range(num_symbols):
    I_vals[i] = 1 if bits[i * 2] else -1
    Q_vals[i] = 1 if bits[i * 2 + 1] else -1

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

lpf_kernel = np.ones(5) / 5
I_rec = np.convolve(I_mixed, lpf_kernel, mode="same")
Q_rec = np.convolve(Q_mixed, lpf_kernel, mode="same")

# Sampling at symbol centers
sample_times = np.arange(
    total_duration / num_symbols / 2, total_duration, total_duration / num_symbols,
)
sample_idx = np.clip((sample_times * FS).astype(int), 0, num_samples - 1)
I_sampled = I_rec[sample_idx]
Q_sampled = Q_rec[sample_idx]

# ML detection: threshold at 0 for 2-PAM
rx_bits = []
for i in range(num_symbols):
    rx_bits.append(1 if I_sampled[i] >= 0 else 0)
    rx_bits.append(1 if Q_sampled[i] >= 0 else 0)

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
# FM MODULATION CHAIN
# ============================================================================
KF = 12000

s_fm = np.cos(2 * np.pi * FC * t + 2 * np.pi * KF * np.cumsum(m_t) / FS)
s_fm_noisy = s_fm + noise

fm_lpf = np.ones(5) / 5
diff5 = np.array([-1, 8, 0, -8, 1]) * FS / 12.0

I_fm = 2 * s_fm_noisy * carrier_cos
Q_fm = -2 * s_fm_noisy * carrier_sin
I_fm_lpf = np.convolve(np.convolve(I_fm, fm_lpf, mode="same"), fm_lpf, mode="same")
Q_fm_lpf = np.convolve(np.convolve(Q_fm, fm_lpf, mode="same"), fm_lpf, mode="same")

fm_phase = np.unwrap(np.arctan2(Q_fm_lpf, I_fm_lpf))
m_hat_fm_raw = np.convolve(fm_phase, diff5, mode="same") / (2 * np.pi * KF)

m_hat_fm_audio = m_hat_fm_raw[dsbsc_sample_idx]
pcm_fm_recovered = np.clip(
    ((m_hat_fm_audio + 1.0) / 2.0 * 255.0), 0, 255,
).astype(np.uint8)
fm_wav = pcm_to_wav_bytes(pcm_fm_recovered)

fm_mse = np.mean((m_hat_fm_audio - m_vals) ** 2)
fm_snr_db = 10 * np.log10(dsbsc_sig_pwr / max(fm_mse, 1e-12))
input_snr_db = -20 * np.log10(max(noise_std, 1e-12))

# ============================================================================
# 16QAM / OFDM CHAINS (conditional on page selection)
# ============================================================================
PAM4_MAP = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
PAM4_INV = {-3: (0, 0), -1: (0, 1), 1: (1, 1), 3: (1, 0)}
SCALE16Q = np.sqrt(10.0)
PAM4_LEVELS_NORM = np.array([-3, -1, 1, 3]) / SCALE16Q
PAM4_BITS = [(0, 0), (0, 1), (1, 1), (1, 0)]

CP_RATIO = 0.25
N_FFT = NUM_SUBCARRIERS
CP_LEN_FS = 0

def _demap_4pam(val):
    idx = np.argmin(np.abs(val - PAM4_LEVELS_NORM))
    return PAM4_BITS[idx]

need_new_chains = PAGE_16QAM_OFDM

if need_new_chains:
    # ========================================================================
    # SC-16QAM (single carrier)
    # ========================================================================
    num_sym_16qam = num_bits // BITS_PER_SYM_16QAM
    bits_16qam = bits[:num_sym_16qam * BITS_PER_SYM_16QAM]

    I_vals_16qam = np.zeros(num_sym_16qam, dtype=np.float64)
    Q_vals_16qam = np.zeros(num_sym_16qam, dtype=np.float64)
    for i in range(num_sym_16qam):
        b0, b1, b2, b3 = bits_16qam[i * 4:(i + 1) * 4]
        I_vals_16qam[i] = PAM4_MAP[(b0, b1)] / SCALE16Q
        Q_vals_16qam[i] = PAM4_MAP[(b2, b3)] / SCALE16Q

    sym_idx_16qam = np.clip(
        np.floor(t * num_sym_16qam / total_duration).astype(int),
        0, num_sym_16qam - 1,
    )
    I_bb_16qam = I_vals_16qam[sym_idx_16qam]
    Q_bb_16qam = Q_vals_16qam[sym_idx_16qam]
    s_rf_16qam = I_bb_16qam * carrier_cos - Q_bb_16qam * carrier_sin

    delay_samps = max(1, int(multipath_delay))
    if multipath_delay > 0 and multipath_atten > 1e-6:
        s_dly = np.zeros(num_samples)
        s_dly[delay_samps:] = s_rf_16qam[:num_samples - delay_samps]
        s_rf_16qam_mp = s_rf_16qam + multipath_atten * s_dly
    else:
        s_rf_16qam_mp = s_rf_16qam

    noise_16qam = rng.normal(0, noise_std, num_samples)
    s_rf_16qam_n = s_rf_16qam_mp + noise_16qam

    I_mix_16qam = 2 * s_rf_16qam_n * carrier_cos
    Q_mix_16qam = -2 * s_rf_16qam_n * carrier_sin
    lpf_k = np.ones(5) / 5
    I_rec_16qam = np.convolve(I_mix_16qam, lpf_k, mode="same")
    Q_rec_16qam = np.convolve(Q_mix_16qam, lpf_k, mode="same")

    T_sym16 = total_duration / num_sym_16qam
    samp_t_16qam = np.arange(T_sym16 / 2, total_duration, T_sym16)
    samp_idx_16qam = np.clip((samp_t_16qam * FS).astype(int), 0, num_samples - 1)
    I_sam_16qam = I_rec_16qam[samp_idx_16qam]
    Q_sam_16qam = Q_rec_16qam[samp_idx_16qam]

    rx_bits_16qam = []
    for i in range(num_sym_16qam):
        bi0, bi1 = _demap_4pam(I_sam_16qam[i])
        bq0, bq1 = _demap_4pam(Q_sam_16qam[i])
        rx_bits_16qam.extend([bi0, bi1, bq0, bq1])

    sc_16qam_ber = sum(
        1 for b, r in zip(bits_16qam, rx_bits_16qam) if b != r
    ) / len(bits_16qam)
    pcm_sc16qam = bits_to_pcm(rx_bits_16qam[:num_bits], num_samples_pcm)
    sc16_wav = pcm_to_wav_bytes(pcm_sc16qam)

    # ========================================================================
    # OFDM-16QAM  (baseband-only – no RF/LPF distortion)
    # ========================================================================
    bpo = N_FFT * BITS_PER_SYM_16QAM
    num_ofdm = len(bits_16qam) // bpo
    bits_ofdm = bits_16qam[:num_ofdm * bpo]

    T_ofdm_total = total_duration / num_ofdm
    T_useful = T_ofdm_total / (1.0 + CP_RATIO)
    T_cp = T_ofdm_total - T_useful
    delta_f = 1.0 / T_useful

    sps_ofdm = int(num_samples / num_ofdm)
    cp_fs = int(sps_ofdm * CP_RATIO / (1.0 + CP_RATIO))
    ufs = sps_ofdm - cp_fs
    CP_LEN_FS = cp_fs

    # Baseband-equivalent multipath channel
    tau_sec = multipath_delay / FS
    delay_samps = max(1, int(multipath_delay))
    h_dly = multipath_atten * np.exp(-1j * 2.0 * np.pi * FC * tau_sec)

    # Per-subcarrier channel frequency response (for equalisation)
    H_k_raw = 1.0 + h_dly * np.exp(
        -1j * 2.0 * np.pi * np.arange(N_FFT) * delta_f * tau_sec
    )

    # ---- OFDM transmitter ----
    ofdm_bb = np.zeros(num_samples, dtype=complex)
    X_ofdm_all = np.zeros((num_ofdm, N_FFT), dtype=complex)

    for sym in range(num_ofdm):
        idx0 = sym * sps_ofdm
        idx1 = idx0 + sps_ofdm
        bs = sym * bpo
        X_sym = np.zeros(N_FFT, dtype=complex)
        for k in range(N_FFT):
            b0, b1, b2, b3 = bits_ofdm[bs + k * 4:bs + k * 4 + 4]
            X_sym[k] = (PAM4_MAP[(b0, b1)] + 1j * PAM4_MAP[(b2, b3)]) / SCALE16Q
        X_ofdm_all[sym, :] = X_sym

        t_u = np.arange(ufs) / FS
        sig_u = np.zeros(ufs, dtype=complex)
        for k in range(N_FFT):
            sig_u += X_sym[k] * np.exp(1j * 2.0 * np.pi * k * delta_f * t_u)
        sig_u /= np.sqrt(N_FFT)
        sig_cp = np.concatenate([sig_u[ufs - cp_fs:], sig_u])
        ofdm_bb[idx0:idx1] = sig_cp[:sps_ofdm]

    # ---- Baseband multipath + complex AWGN ----
    if multipath_delay > 0 and multipath_atten > 1e-6:
        ofdm_bb_dly = np.zeros(num_samples, dtype=complex)
        ofdm_bb_dly[delay_samps:] = ofdm_bb[:num_samples - delay_samps]
        orx_bb = ofdm_bb + h_dly * ofdm_bb_dly
    else:
        orx_bb = ofdm_bb.copy()

    orx_bb += (rng.normal(0, noise_std / np.sqrt(2), num_samples) +
               1j * rng.normal(0, noise_std / np.sqrt(2), num_samples))

    # ---- OFDM receiver ----
    def _demap_16qam_sym(z):
        i0, i1 = _demap_4pam(z.real)
        q0, q1 = _demap_4pam(z.imag)
        return [i0, i1, q0, q1]

    rx_X_all = np.zeros((num_ofdm, N_FFT), dtype=complex)
    rx_X_noeq = np.zeros((num_ofdm, N_FFT), dtype=complex)
    ofdm_rx_bits = []
    ofdm_per_sc_errors = np.zeros(N_FFT, dtype=int)
    ofdm_per_sc_total = np.zeros(N_FFT, dtype=int)

    t_demod = np.arange(ufs) / FS

    for sym in range(num_ofdm):
        idx0 = sym * sps_ofdm
        sym_bb = orx_bb[idx0:idx0 + sps_ofdm]
        sym_u = sym_bb[cp_fs:cp_fs + ufs]
        rx_X = np.zeros(N_FFT, dtype=complex)
        for k in range(N_FFT):
            rx_X[k] = np.sum(sym_u * np.exp(-1j * 2.0 * np.pi * k * delta_f * t_demod)) / ufs
        rx_X_noeq[sym, :] = rx_X

        rx_X_eq = np.zeros(N_FFT, dtype=complex)
        for k in range(N_FFT):
            if abs(H_k_raw[k]) > 1e-6:
                rx_X_eq[k] = rx_X[k] / H_k_raw[k] * np.sqrt(N_FFT)
            else:
                rx_X_eq[k] = rx_X[k] * np.sqrt(N_FFT)
        rx_X_all[sym, :] = rx_X_eq

        for k in range(N_FFT):
            tx_bits = bits_ofdm[sym * bpo + k * 4:sym * bpo + k * 4 + 4]
            rx_bits_k = _demap_16qam_sym(rx_X_eq[k])
            ofdm_rx_bits.extend(rx_bits_k)
            for bi in range(4):
                ofdm_per_sc_total[k] += 1
                if tx_bits[bi] != rx_bits_k[bi]:
                    ofdm_per_sc_errors[k] += 1

    ofdm_ber = sum(
        1 for b, r in zip(bits_ofdm, ofdm_rx_bits) if b != r
    ) / len(bits_ofdm)
    ofdm_per_sc_ber = ofdm_per_sc_errors / np.maximum(ofdm_per_sc_total, 1)
    pcm_ofdm = bits_to_pcm(ofdm_rx_bits[:num_bits], num_samples_pcm)
    ofdm_wav = pcm_to_wav_bytes(pcm_ofdm)


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
if PAGE_QAM:
    st.sidebar.caption(
        f"音频: {audio_duration:.1f}s · 音频SR: {AUDIO_SR} Hz\n"
        f"载波: {FC} Hz · 仿真SR: {FS} Hz · {num_symbols}符号\n"
        f"4QAM (QPSK) · σ = {noise_std:.2f} · BER = {ber:.4f}"
    )
elif PAGE_DSBSC:
    st.sidebar.caption(
        f"音频: {audio_duration:.1f}s · 音频SR: {AUDIO_SR} Hz\n"
        f"载波: {FC} Hz · 仿真SR: {FS} Hz\n"
        f"DSB-SC · σ = {noise_std:.2f} · SNR = {dsbsc_snr_db:.1f} dB"
    )
elif PAGE_FM:
    st.sidebar.caption(
        f"音频: {audio_duration:.1f}s · 音频SR: {AUDIO_SR} Hz\n"
        f"载波: {FC} Hz · 仿真SR: {FS} Hz\n"
        f"FM (β=3) · σ = {noise_std:.2f} · SNR = {fm_snr_db:.1f} dB"
    )
elif PAGE_16QAM_OFDM:
    st.sidebar.caption(
        f"音频: {audio_duration:.1f}s · 音频SR: {AUDIO_SR} Hz\n"
        f"载波: {FC} Hz · 仿真SR: {FS} Hz\n"
        f"16QAM vs OFDM · 多径延迟={multipath_delay} · 衰减={multipath_atten:.2f} · σ={noise_std:.2f}"
    )
else:
    st.sidebar.caption(
        f"音频: {audio_duration:.1f}s · 音频SR: {AUDIO_SR} Hz\n"
        f"载波: {FC} Hz · 仿真SR: {FS} Hz · σ = {noise_std:.2f} · 输入SNR = {input_snr_db:.1f} dB\n"
        f"4QAM BER = {ber:.4f}  |  DSB-SC SNR = {dsbsc_snr_db:.1f} dB  |  FM SNR = {fm_snr_db:.1f} dB"
    )

# ============================================================================
# MAIN TITLE
# ============================================================================
if PAGE_QAM:
    st.title("📡 4QAM (QPSK) 数字调制原理 — 音频通信链路仿真")
    st.caption(
        "音频 PCM → 比特 → 4QAM 调制 → 上变频 → AWGN 信道 → "
        "下变频 → 5点LPF → ML 判决 → 比特恢复 → PCM → 音频播放"
    )
elif PAGE_DSBSC:
    st.title("📻 DSB-SC 模拟调制原理 — 音频通信链路仿真")
    st.caption(
        "音频 PCM → m(t) 模拟幅度 → DSB-SC 调制 → AWGN 信道 → "
        "相干解调 → PCM → 音频播放"
    )
elif PAGE_FM:
    st.title("📡 FM 频率调制原理 — 音频通信链路仿真")
    st.caption(
        "音频 PCM → m(t) 模拟幅度 → FM 调制 (β=3) → AWGN 信道 → "
        "正交鉴频 → 5点LPF → PCM → 音频播放"
    )
elif PAGE_16QAM_OFDM:
    st.title("📡 16QAM单载波 vs OFDM — 多径信道对比")
    st.caption(
        f"16QAM(4bit/符号) → 单载波 vs OFDM({N_FFT}子载波) → "
        "多径信道 → AWGN → 解调 → 对比ISI与均衡效果"
    )


def show_input_audio_module(mod_name, extra_metrics, info_text):
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
            st.metric("调制方式", mod_name)
            for label, val in extra_metrics:
                st.metric(label, val)
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
        st.info(info_text)

# ============================================================================
# ============================================================================
#  4QAM PRINCIPLE PAGE
# ============================================================================
# ============================================================================
if PAGE_QAM:
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
            st.metric("4QAM 符号数", num_symbols)
            st.metric("每符号比特数", "2 bit")

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
            f"这 {num_bits} 个比特将分成 {num_symbols} 组 (每组 2 bit) 进行 4QAM (QPSK) 调制。"
        )

    # ------------------------------------------------------------------
    # MODULE 2: Symbol Mapping & Constellation
    # ------------------------------------------------------------------
    with st.expander("📊 2. 符号映射与星座图 — 4QAM/QPSK", expanded=False):
        section_header("📊 步骤2: 符号映射 (4QAM / QPSK)")

        col_tab, col_math = st.columns([1, 2])
        with col_tab:
            st.markdown(
                """
            | I bit | I 电平 | Q bit | Q 电平 |
            |--------|--------|--------|--------|
            | **0** |  −1 | **0** |  −1 |
            | **1** |  +1 | **1** |  +1 |
            """
            )
            st.caption("2-PAM 映射 → 2×2 = 4 星座点")

        with col_math:
            st.latex(r"s_k = I_k + j Q_k,\quad I_k, Q_k \in \{\pm 1\}")
            st.latex(r"\text{每个符号携带 2 bit: I路bit + Q路bit}")

        st.markdown("### 4QAM 星座图 与 基带 I/Q 波形")

        fig_bb, (ax_c, ax_w) = plt.subplots(1, 2, figsize=(13, 5))
        fig_bb.patch.set_facecolor("white")

        # Constellation
        const_pts = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        bit_labels = ["00", "01", "10", "11"]
        for (pt_i, pt_q), bl in zip(const_pts, bit_labels):
            ax_c.scatter(pt_i, pt_q, c="#1f77b4", s=120, zorder=5, edgecolors="black", linewidths=0.5)
            ax_c.annotate(bl, (pt_i, pt_q), textcoords="offset points",
                          xytext=(10, 10), fontsize=8, color="#1f77b4", fontweight="bold")
        ax_c.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_c.axvline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_c.set_xlim(-2.5, 2.5)
        ax_c.set_ylim(-2.5, 2.5)
        ax_c.set_xlabel("I (同相分量)", fontsize=10)
        ax_c.set_ylabel("Q (正交分量)", fontsize=10)
        ax_c.set_title("4QAM (QPSK) 星座图", fontsize=12, fontweight="bold")
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
        ax_w.set_ylim(-2.5, 2.5)
        ax_w.set_xlim(0, t_wave[-1])
        ax_w.legend(fontsize=9, loc="upper right")
        ax_w.grid(True, alpha=0.3)
        for i in range(disp_syms):
            b2 = bits[i*2:(i+1)*2]
            bit_label = f"{b2[0]},{b2[1]}"
            ax_w.axvline(i * sym_dur, color="gray", linestyle=":", linewidth=0.5)
            ax_w.text((i + 0.5) * sym_dur, 2.0, bit_label, ha="center", fontsize=7, color="gray")

        fig_bb.tight_layout()
        st.pyplot(fig_bb)
        plt.close(fig_bb)

        st.info(
            "💡 4QAM 每个符号携带 2 bit。I 路和 Q 路各自使用 2-PAM (2电平, ±1)，"
            "组合成 2×2=4 个星座点。判决只需判断符号在 0 上/下方。"
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

        **滤波器**: 5点滑动平均 LPF (采样率 {FS} Hz)
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
            f"💡 5点滑动平均 LPF 有效滤除了 2fc={2*FC} Hz 高频分量，"
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
        对每个点，在 4QAM 星座中找到最近的星座点——**ML 判决**。

        由于 I/Q 正交且独立，判决简化为独立 2-PAM: I ≥ 0 → 1, I < 0 → 0 (Q 同理)。
        """
        )

        st.markdown("#### 🎯 接收星座图 (含判决边界)")

        fig_det, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(13, 5))

        # Raw sampled scatter
        ax_s.scatter(I_sampled, Q_sampled, c="purple", s=12, alpha=0.6, edgecolors="none")
        ax_s.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_s.axvline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_s.set_xlabel("I", fontsize=10)
        ax_s.set_ylabel("Q", fontsize=10)
        ax_s.set_title(f"接收采样点 (σ={noise_std:.2f})", fontsize=11, fontweight="bold")
        ax_s.set_aspect("equal")
        ax_s.grid(True, alpha=0.3)
        ax_s.set_xlim(-3, 3)
        ax_s.set_ylim(-3, 3)

        # Detected constellation
        const_pts = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        ax_d.scatter(*zip(*const_pts), c="#1f77b4", s=100, zorder=5,
                     edgecolors="black", linewidths=0.5)
        ax_d.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_d.axvline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_d.set_xlabel("I", fontsize=10)
        ax_d.set_ylabel("Q", fontsize=10)
        ax_d.set_title("4QAM 判决星座点", fontsize=11, fontweight="bold")
        ax_d.set_aspect("equal")
        ax_d.grid(True, alpha=0.3)
        ax_d.set_xlim(-3, 3)
        ax_d.set_ylim(-3, 3)

        fig_det.tight_layout()
        st.pyplot(fig_det)
        plt.close(fig_det)

        st.markdown("#### 📊 2-PAM 判决区域 (I路示例)")
        fig_pam, ax_pam = plt.subplots(figsize=(10, 2.5))
        fig_pam.patch.set_facecolor("white")
        ax_pam.scatter(I_sampled[:min(80, num_symbols)],
                       np.zeros(min(80, num_symbols)), c="purple", s=15, alpha=0.5)
        ax_pam.axvline(0, color="orange", linestyle="--", linewidth=1.5)
        for lvl, lbl in zip([-1, 1], ["0", "1"]):
            ax_pam.axvline(lvl, color="#1f77b4", linestyle="-", linewidth=2, alpha=0.4)
            ax_pam.text(lvl, 0.03, lbl, ha="center", fontsize=12, fontweight="bold", color="#1f77b4")
        ax_pam.set_xlabel("I 幅度", fontsize=10)
        ax_pam.set_xlim(-3, 3)
        ax_pam.set_ylim(-0.1, 0.1)
        ax_pam.set_title("I 路 2-PAM 判决边界 (I∈{−1,+1}, 阈值 0)", fontsize=11, fontweight="bold")
        ax_pam.set_yticks([])
        ax_pam.grid(True, alpha=0.3, axis="x")
        fig_pam.tight_layout()
        st.pyplot(fig_pam)
        plt.close(fig_pam)

        st.info(
            "💡 判决边界为 0。接收值 ≥ 0 判为 +1 (bit=1), < 0 判为 −1 (bit=0)。"
            "Q 路同理。4QAM 的 2-PAM 判决简单高效。"
        )

    # ------------------------------------------------------------------
    # MODULE 7: Bit Recovery & BER
    # ------------------------------------------------------------------
    with st.expander("✅ 7. 比特恢复与误码率 — BER计算", expanded=False):
        section_header("✅ 步骤7: 比特恢复与误码率 (BER)")

        st.markdown(
            f"""
        判决得到的星座点通过判决阈值恢复为 2 bit，再拼接成完整的接收比特流，
        与原始发送比特逐一对比。
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("发送比特数", num_bits)
            st.metric("4QAM 符号数", num_symbols)
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
    with st.expander("🔊 8. 音频听感对比 — 4QAM原始 vs 解调", expanded=False):
        section_header("🔊 步骤8: 音频听感对比 — 原始 vs 解调后")

        st.markdown(
            f"""
        将恢复的 {num_bits} 个比特重新组装为 PCM 音频样本，对比原始音频和经过
        **4QAM 调制 → AWGN 信道 → 解调** 之后的音频。
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📤 原始音频")
            st.audio(pcm_to_wav_bytes(pcm_original), format="audio/wav")
            st.caption("无调制 · 原始 PCM")
        with col2:
            st.markdown("### 📡 4QAM 解调后")
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
        axes[1].set_title(f"4QAM 解调 (BER={ber:.4f})", fontsize=11, fontweight="bold")
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
    show_input_audio_module(
        "DSB-SC",
        [("载波抑制", "是 (无直流)")],
        "💡 m(t) 是归一化到 [−1, +1] 的音频幅度。DSB-SC 直接用它调制载波，"
        "没有直流偏置——这与 AM 不同 (AM 需要 1+m(t) 保证包络非负)。",
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
            "总能听到一些旋律痕迹。与 4QAM 的「悬崖效应」形成有趣对比。"
        )


# ============================================================================
# ============================================================================
#  FM PRINCIPLE PAGE
# ============================================================================
# ============================================================================
elif PAGE_FM:
    show_input_audio_module(
        "FM (Frequency Modulation)",
        [("调频系数 β", "3.0")],
        "💡 m(t) 是归一化到 [−1, +1] 的音频幅度。FM 用 m(t) 的积分控制相位，"
        "瞬时频率随 m(t) 线性变化: f(t) = fc + kf·m(t)。",
    )

    # ------------------------------------------------------------------
    # FM MODULE 2: FM Modulation
    # ------------------------------------------------------------------
    with st.expander("📡 2. FM频率调制 — 相位积分与载波合成", expanded=False):
        section_header(
            "📡 步骤2: FM 频率调制",
            formula=r"s_{\mathrm{FM}}(t) = \cos\!\Big(2\pi f_c t + 2\pi k_f\!\int_0^t m(\tau)d\tau\Big)",
        )

        st.markdown(
            f"""
        **FM (Frequency Modulation)** 用消息信号 m(t) 控制载波的瞬时频率:

        $$\\begin{{aligned}}
        f_i(t) &= f_c + k_f \\cdot m(t) \\\\
        \\phi(t) &= 2\\pi \\int_0^t f_i(\\tau)d\\tau
                  = 2\\pi f_c t + 2\\pi k_f \\int_0^t m(\\tau)d\\tau \\\\
        s_{{\\mathrm{{FM}}}}(t) &= \\cos(\\phi(t))
        \\end{{aligned}}$$

        **参数设定**: $f_c = {FC}$ Hz, $k_f = {KF}$ Hz/V, |m(t)| ≤ 1

        **频率偏移**: $\\Delta f = k_f \\cdot \\max|m(t)| = {KF}$ Hz
        **调频指数**: $\\beta = \\Delta f / f_{{\\max}} \\approx 12000 / 4000 = 3$
        **卡森带宽**: $B \\approx 2(\\Delta f + f_{{\\max}}) = 2(12000+4000) = 32\\text{{ kHz}}$

        瞬时频率范围: $[f_c - k_f, f_c + k_f] = [{FC-KF}, {FC+KF}]$ Hz
        """
        )

        st.markdown("#### 🔍 调制过程 (前几个符号周期)")

        t_z = t[zoom_mask]
        m_tz = m_t[zoom_mask]
        s_fm_z = s_fm[zoom_mask]
        carrier_z = carrier_cos[zoom_mask]

        fig_mod, (ax_m, ax_s) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
        fig_mod.patch.set_facecolor("white")

        ax_m.plot(t_z, m_tz, color="blue", linewidth=1.0, label="m(t)")
        ax_m.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax_m.set_ylabel("m(t)", fontsize=9)
        ax_m.set_title("调制信号 m(t)", fontsize=11, fontweight="bold")
        ax_m.legend(fontsize=8)
        ax_m.grid(True, alpha=0.3)

        ax_s.plot(t_z, s_fm_z, color="purple", linewidth=0.7, label=r"s_FM(t)")
        ax_s.plot(t_z, m_tz, color="blue", linewidth=1.2, alpha=0.25,
                  linestyle="--")
        ax_s.set_xlabel("时间 (s)", fontsize=9)
        ax_s.set_ylabel("已调信号", fontsize=9)
        ax_s.set_title(f"FM 已调信号 ($f_c={FC}$ Hz, $k_f={KF}$ Hz/V)", fontsize=11, fontweight="bold")
        ax_s.legend(fontsize=8, loc="upper right")
        ax_s.grid(True, alpha=0.3)

        fig_mod.tight_layout()
        st.pyplot(fig_mod)
        plt.close(fig_mod)

        st.info(
            "💡 m(t)>0 时，载波频率升高(波形变密); m(t)<0 时，载波频率降低(波形变疏)。"
            "FM 信号包络恒定，功率集中。信息蕴含在频率变化而非幅度变化中。"
        )

    with st.expander("🌊 3. AWGN信道 — 高斯噪声叠加", expanded=False):
        section_header(
            "🌊 步骤3: AWGN信道",
            formula=r"r(t) = s_{\mathrm{FM}}(t) + w(t),\quad w(t) \sim \mathcal{N}(0, \sigma^2)",
        )

        st.markdown(
            f"""
        FM 信号通过 AWGN 信道，叠加高斯白噪声 ($\\sigma = {noise_std:.2f}$)。
        FM 有「门限效应」: 当输入 SNR 高于门限值时，解调 SNR 大幅优于 AM；
        低于门限时性能急剧恶化。
        """
        )

        st.markdown("#### 🔍 放大对比: 理想 FM vs 含噪 FM")

        t_z = t[zoom_mask]
        s_fm_z = s_fm[zoom_mask]
        noisy_z = s_fm_noisy[zoom_mask]

        fig_n, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
        fig_n.patch.set_facecolor("white")

        ax1.plot(t_z, s_fm_z, color="purple", linewidth=0.6)
        ax1.set_ylabel("理想", fontsize=9)
        ax1.set_title("理想 FM 信号", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_z, noisy_z, color="#e67e22", linewidth=0.6, alpha=0.9)
        ax2.plot(t_z, s_fm_z, color="purple", linewidth=0.6, alpha=0.25,
                 label="理想 (半透明)")
        ax2.set_xlabel("时间 (s)", fontsize=9)
        ax2.set_ylabel("含噪", fontsize=9)
        ax2.set_title(f"含噪 FM ($\\sigma={noise_std:.2f}$)", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig_n.tight_layout()
        st.pyplot(fig_n)
        plt.close(fig_n)

        st.info(
            f"💡 噪声 σ = {noise_std:.2f} 叠加后，波形在时间轴上出现随机抖动。"
            "FM 对幅度噪声有较强抗性，但对相位(频率)噪声敏感。"
        )

    with st.expander("⬇️ 4. 正交鉴频解调 — I/Q混频 → LPF → arctan → 微分 → m̂(t)", expanded=False):
        section_header(
            "⬇️ 步骤4: 正交鉴频解调 (Quadrature Detector)",
            formula=r"\hat{m}(t) = \frac{1}{2\pi k_f}\cdot\frac{d}{dt}\arctan\!\Big(\frac{\hat{Q}(t)}{\hat{I}(t)}\Big)",
        )

        st.markdown(
            f"""
        接收端使用 **正交鉴频器 (Quadrature FM Detector)** 解调 FM 信号:

        $$\\begin{{aligned}}
        I(t) &= 2r(t)\\cos(2\\pi f_c t) = \\cos\\phi_m(t) + \\cos(4\\pi f_c t + \\phi_m) \\\\
        Q(t) &= -2r(t)\\sin(2\\pi f_c t) = \\sin\\phi_m(t) - \\sin(4\\pi f_c t + \\phi_m) \\\\
        \\hat{{I}}(t) &= \\mathrm{{LPF}}\\{{I(t)\\}} \\approx \\cos\\phi_m(t) \\\\
        \\hat{{Q}}(t) &= \\mathrm{{LPF}}\\{{Q(t)\\}} \\approx \\sin\\phi_m(t) \\\\
        \\phi_m(t) &= \\arctan(\\hat{{Q}}/\\hat{{I}}) \\;\\xrightarrow{{\\text{{微分}}}}\\; 2\\pi k_f\\cdot\\hat{{m}}(t)
        \\end{{aligned}}$$

        **关键**: β=3 时 FM 的 2fc 分量带宽较宽 (16~64 kHz)，I/Q 通道各经 5点LPF **两次** 充分滤除，
        arctan 在已滤噪的基带信号上计算相位，再用5点中心差分化器提取瞬时频率。
        **滤波器**: I/Q 各 5点LPF ×2 + 微分器 5点，统一 5点设计。
        """
        )

        st.markdown("#### 🔬 I/Q 混频后 (LPF前) — 可见基带 + 2fc 高频纹波")

        t_mz = t[mix_mask]
        I_fm_z = I_fm[mix_mask]

        fig_mix, ax_mix = plt.subplots(figsize=(13, 3.5))
        fig_mix.patch.set_facecolor("white")
        ax_mix.plot(t_mz, I_fm_z, color="#e67e22", linewidth=0.7,
                    label=r"$2r(t)\cos(2\pi f_c t)$ I路混频")
        ax_mix.plot(t_mz, I_fm_lpf[mix_mask], color="green", linewidth=1.2, alpha=0.7,
                    linestyle="--", label="LPF后 cos(φ_m)")
        ax_mix.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax_mix.set_xlabel("时间 (s)", fontsize=10)
        ax_mix.set_ylabel("幅度", fontsize=10)
        ax_mix.legend(fontsize=8, loc="upper right")
        ax_mix.grid(True, alpha=0.3)
        ax_mix.set_title(
            "I 路混频输出 (含 cos(φ_m) 基带 + 2fc 高频)", fontsize=12, fontweight="bold",
        )
        fig_mix.tight_layout()
        st.pyplot(fig_mix)
        plt.close(fig_mix)

        st.markdown("#### 🎯 LPF 输出 — 恢复 m̂(t) vs 原始 m(t)")
        fig_lpf, ax_lpf = plt.subplots(figsize=(13, 3.5))
        fig_lpf.patch.set_facecolor("white")

        t_z = t[zoom_mask]
        fm_demod_z = m_hat_fm_raw[zoom_mask]
        m_t_z = m_t[zoom_mask]

        ax_lpf.plot(t_z, fm_demod_z, color="#e67e22", linewidth=1.0,
                    label=r"恢复 $\hat{m}(t)$")
        ax_lpf.plot(t_z, m_t_z, color="green", linewidth=1.2, alpha=0.45,
                    linestyle="--", label="原始 m(t)")
        ax_lpf.set_xlabel("时间 (s)", fontsize=10)
        ax_lpf.set_ylabel("幅度", fontsize=10)
        ax_lpf.legend(fontsize=8, loc="upper right")
        ax_lpf.grid(True, alpha=0.3)
        ax_lpf.set_ylim(-1.5, 1.5)
        ax_lpf.set_title(
            "LPF 输出: 恢复的 m̂(t) vs 原始 m(t)",
            fontsize=12, fontweight="bold",
        )
        fig_lpf.tight_layout()
        st.pyplot(fig_lpf)
        plt.close(fig_lpf)

        st.info(
            "💡 正交鉴频器先滤波再算相位，避免了噪声在相位域的放大。"
            "I/Q 通道 5点LPF 恰好零点化 2fc=40kHz，arctan 在基带运行，数值稳定。"
        )

    with st.expander("🔊 5. 音频听感对比 — FM 解调效果", expanded=False):
        section_header("🔊 步骤5: 音频听感对比 — FM 解调效果")

        st.markdown(
            f"""
        将恢复的 $\\hat{{m}}(t)$ 重新映射为 PCM 音频样本，
         对比原始音频和经过 **FM 调制 → AWGN 信道 → 正交鉴频解调** 之后的音频。
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📤 原始音频")
            st.audio(pcm_to_wav_bytes(pcm_original), format="audio/wav")
            st.caption("无调制 · 原始 PCM")
        with col2:
            st.markdown("### 📡 FM 解调后")
            st.audio(fm_wav, format="audio/wav")
            if fm_snr_db > 30:
                st.success(f"SNR = {fm_snr_db:.1f} dB — 高质量")
            elif fm_snr_db > 15:
                st.warning(f"SNR = {fm_snr_db:.1f} dB — 中等质量")
            else:
                st.error(f"SNR = {fm_snr_db:.1f} dB — 低质量")

        st.markdown("### 📊 波形对比")

        fig_wcmp3, axes = plt.subplots(2, 1, figsize=(13, 4.5), sharex=True)
        fig_wcmp3.patch.set_facecolor("white")
        t_audio_plt = np.arange(num_samples_pcm) / AUDIO_SR

        axes[0].plot(t_audio_plt, pcm_original.astype(int), color="blue", linewidth=0.5)
        axes[0].set_ylabel("原始 PCM", fontsize=9)
        axes[0].set_title("原始音频", fontsize=11, fontweight="bold")
        axes[0].set_ylim(0, 255)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_audio_plt, pcm_fm_recovered.astype(int),
                     color="#e67e22", linewidth=0.5)
        axes[1].set_xlabel("时间 (s)", fontsize=9)
        axes[1].set_ylabel("FM 恢复", fontsize=9)
        axes[1].set_title(
            f"FM 解调 (SNR={fm_snr_db:.1f} dB)", fontsize=11, fontweight="bold",
        )
        axes[1].set_ylim(0, 255)
        axes[1].grid(True, alpha=0.3)

        fig_wcmp3.tight_layout()
        st.pyplot(fig_wcmp3)
        plt.close(fig_wcmp3)

        st.info(
            "💡 FM 对幅度噪声有天然抑制 (包络恒定)，"
            "但对频率噪声较敏感。在中等噪声下 FM 音质通常优于 DSB-SC。"
            "注意门限效应: 噪声过大时，FM 解调会突然崩溃。"
        )

# ============================================================================
# ============================================================================
#  PAGE 5: 16QAM vs OFDM 多径
# ============================================================================
# ============================================================================
elif PAGE_16QAM_OFDM:
    # ------------------------------------------------------------------
    # MODULE 1: 16QAM Constellation
    # ------------------------------------------------------------------
    with st.expander("📊 1. 16QAM星座图与符号映射", expanded=False):
        section_header("📊 步骤1: 16QAM (Quadrature Amplitude Modulation)")

        st.markdown(
            """
        16QAM 每个符号携带 **4 bit**，I/Q 各使用 4-PAM (4电平脉冲幅度调制):
        - I 路: 4-PAM → 电平 {−3, −1, +1, +3} → 2 bit
        - Q 路: 4-PAM → 电平 {−3, −1, +1, +3} → 2 bit
        - 组合: 4 × 4 = **16 个星座点**

        归一化后电平: {−3/√10, −1/√10, +1/√10, +3/√10}，平均功率 = 1。
        """
        )

        st.markdown("#### 16QAM Gray编码星座图")
        levels = np.array([-3, -1, 1, 3]) / SCALE16Q
        bit_labels_4pam = ["00", "01", "11", "10"]

        fig_c, ax_c = plt.subplots(figsize=(7, 7))
        fig_c.patch.set_facecolor("white")
        for il, li in enumerate(levels):
            for ql, lq in enumerate(levels):
                ax_c.scatter(li, lq, c="#1f77b4", s=100, zorder=5,
                             edgecolors="black", linewidths=0.5)
                lbl = f"{bit_labels_4pam[il]},{bit_labels_4pam[ql]}"
                ax_c.annotate(lbl, (li, lq), textcoords="offset points",
                              xytext=(8, 8), fontsize=7, color="#1f77b4")
        ax_c.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_c.axvline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_c.set_xlim(-1.8, 1.8)
        ax_c.set_ylim(-1.8, 1.8)
        ax_c.set_xlabel("I (同相)", fontsize=10)
        ax_c.set_ylabel("Q (正交)", fontsize=10)
        ax_c.set_title("16QAM 星座图 (Gray编码)", fontsize=12, fontweight="bold")
        ax_c.set_aspect("equal")
        ax_c.grid(True, alpha=0.3)
        fig_c.tight_layout()
        st.pyplot(fig_c)
        plt.close(fig_c)

        st.info(
            "💡 16QAM 的 16 个星座点密度远高于 4QAM (仅4点)。"
            "星座点间距更小 → 相同噪声下更容易误判 → **高阶调制抗噪能力更弱**。"
            "但每个符号携带 4 bit，频谱效率是 4QAM 的 2 倍。"
        )

    # ------------------------------------------------------------------
    # MODULE 2: SC-16QAM multipath damage
    # ------------------------------------------------------------------
    with st.expander("🌊 2. 单载波16QAM — 多径损伤 (ISI)", expanded=False):
        section_header("🌊 步骤2: 单载波 16QAM 通过多径信道")

        st.markdown(
            f"""
        单载波 16QAM 信号通过两径信道:
        $$h(t) = \\delta(t) + {multipath_atten:.2f} \\cdot \\delta(t - {multipath_delay}/{FS:.0f})$$

        - 符号周期: **{total_duration/num_sym_16qam*1e6:.1f} μs** (数据速率高 → 符号短)
        - 多径延迟: **{multipath_delay/FS*1e6:.1f} μs**

        当多径延迟接近符号周期时 → **码间干扰 (ISI)** 严重 → 星座图混乱。
        """
        )

        st.markdown("#### 信道脉冲响应")
        fig_h, ax_h = plt.subplots(figsize=(10, 2))
        fig_h.patch.set_facecolor("white")
        taps = [1.0, multipath_atten]
        tap_t = [0, multipath_delay / FS * 1e6]
        markerline, stemlines, baseline = ax_h.stem(tap_t, taps, linefmt="C0-", markerfmt="C0o")
        ax_h.set_xlabel("延迟 (μs)", fontsize=10)
        ax_h.set_ylabel("幅度", fontsize=10)
        ax_h.set_title(f"两径信道 h(t) (延迟={multipath_delay/FS*1e6:.1f}μs, 衰减={multipath_atten:.2f})",
                       fontsize=11, fontweight="bold")
        ax_h.grid(True, alpha=0.3)
        fig_h.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

        st.markdown("#### 🎯 接收星座图 (SC-16QAM) — ISI 导致星座点散开")
        fig_sc, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(13, 5))
        fig_sc.patch.set_facecolor("white")

        ax_s1.scatter(I_sam_16qam, Q_sam_16qam, c="purple", s=8, alpha=0.5, edgecolors="none")
        ax_s1.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_s1.axvline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_s1.set_xlabel("I", fontsize=10)
        ax_s1.set_ylabel("Q", fontsize=10)
        ax_s1.set_title(f"SC-16QAM 接收星座 (BER={sc_16qam_ber:.4f})", fontsize=11, fontweight="bold")
        ax_s1.set_aspect("equal")
        ax_s1.grid(True, alpha=0.3)
        ax_s1.set_xlim(-2.0, 2.0)
        ax_s1.set_ylim(-2.0, 2.0)

        for lx in levels:
            for ly in levels:
                ax_s2.scatter(lx, ly, c="#1f77b4", s=40, zorder=5,
                              edgecolors="black", linewidths=0.3, alpha=0.3)
        ax_s2.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_s2.axvline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_s2.set_xlabel("I", fontsize=10)
        ax_s2.set_ylabel("Q", fontsize=10)
        ax_s2.set_title("理想 16QAM 星座 (参考)", fontsize=11, fontweight="bold")
        ax_s2.set_aspect("equal")
        ax_s2.grid(True, alpha=0.3)
        ax_s2.set_xlim(-2.0, 2.0)
        ax_s2.set_ylim(-2.0, 2.0)

        fig_sc.tight_layout()
        st.pyplot(fig_sc)
        plt.close(fig_sc)

        st.error(
            "❌ **单载波16QAM的多径灾难**: 没有均衡器的情况下，ISI 使星座点严重散开。"
            f"BER = {sc_16qam_ber:.4f}。"
            "符号速率越高 (即带宽利用率越高)，ISI 越严重。"
        )

    # ------------------------------------------------------------------
    # MODULE 3: OFDM multipath immunity
    # ------------------------------------------------------------------
    with st.expander("🛡️ 3. OFDM-16QAM — 循环前缀抗多径", expanded=False):
        section_header("🛡️ 步骤3: OFDM-16QAM 通过相同多径信道 (基带仿真)")

        st.markdown(
            f"""
        OFDM 把高速串行数据**分解到 {N_FFT} 个并行的低速子载波**上:

        - 每个子载波符号周期: **{T_useful*1e6:.1f} μs** (是SC的 {N_FFT} 倍!)
        - 循环前缀 (CP): **{T_cp*1e6:.1f} μs** (CP={CP_RATIO*100:.0f}%)
        - 多径延迟: **{multipath_delay/FS*1e6:.1f} μs** {"" if multipath_delay/FS <= T_cp else "⚠️ > CP!"}

        **关键**: 只要多径延迟 ≤ CP 长度，接收端去掉 CP 后 → DFT → **每个子载波只乘了一个复系数!**
        简单的一阶均衡 (除以信道系数) 即可完美恢复。
        (复基带仿真，无 RF 上下变频 / LPF 失真)
        """
        )

        st.markdown("#### 🎯 每个子载波的接收星座图 (ZF均衡后)")
        n_cols = N_FFT
        n_rows = (N_FFT + n_cols - 1) // n_cols
        fig_scs, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.8 * n_rows))
        fig_scs.patch.set_facecolor("white")
        axes = np.atleast_1d(axes).flatten()

        for k in range(N_FFT):
            ax = axes[k]
            rx_pts = rx_X_all[:100, k].flatten() if num_ofdm >= 100 else rx_X_all[:, k].flatten()
            ax.scatter(rx_pts.real, rx_pts.imag, c="purple", s=10, alpha=0.5, edgecolors="none")
            for lx in levels:
                for ly in levels:
                    ax.scatter(lx, ly, c="#1f77b4", s=30, zorder=5,
                               edgecolors="black", linewidths=0.3, alpha=0.3)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.8, 1.8)
            ax.set_aspect("equal")
            ax.set_title(f"子载波 #{k}  (BER={ofdm_per_sc_ber[k]:.4f})", fontsize=10)
            ax.grid(True, alpha=0.2)

        for k in range(N_FFT, len(axes)):
            axes[k].set_visible(False)

        fig_scs.tight_layout()
        st.pyplot(fig_scs)
        plt.close(fig_scs)

        if multipath_delay / FS <= T_cp:
            st.success(
                f"✅ **OFDM 完胜**: CP ({T_cp*1e6:.1f}μs) > 多径延迟 ({multipath_delay/FS*1e6:.1f}μs)。"
                f"所有 {N_FFT} 个子载波星座图清晰! OFDM BER = {ofdm_ber:.4f}。"
                f"每个子载波仅需简单一阶均衡 (除以 H_k)。"
            )
        else:
            st.warning(
                f"⚠️ CP ({T_cp*1e6:.1f}μs) < 多径延迟 ({multipath_delay/FS*1e6:.1f}μs)。"
                f"仍有残存 ISI。增大延迟滑块试试。"
            )

        st.info(
            "💡 OFDM 的核心思想: 用**并行化**换取**抗多径能力**。\n\n"
            "串行高速 → 拆成并行的低速子载波 → 每子载波经历的仅仅是'平衰落'(乘一个复数), "
            "而不是'频率选择性衰落' → 均衡极其简单。\n\n"
            "4G LTE / 5G / WiFi / DVB 都使用 OFDM。"
        )

    # ------------------------------------------------------------------
    # MODULE 4: Comparison
    # ------------------------------------------------------------------
    with st.expander("🔊 4. 对比 — 音频听感与BER", expanded=False):
        section_header("🔊 步骤4: 音频听感对比 — SC-16QAM vs OFDM-16QAM")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📤 原始音频")
            st.audio(pcm_to_wav_bytes(pcm_original), format="audio/wav")
            st.caption(f"无调制 · {audio_duration:.1f}s")
        with col2:
            st.markdown("### 📡 SC-16QAM")
            st.audio(sc16_wav, format="audio/wav")
            if sc_16qam_ber < 0.01:
                st.success(f"BER = {sc_16qam_ber:.4f}")
            elif sc_16qam_ber < 0.1:
                st.warning(f"BER = {sc_16qam_ber:.4f}")
            else:
                st.error(f"BER = {sc_16qam_ber:.4f}")
        with col3:
            st.markdown("### 📡 OFDM-16QAM")
            st.audio(ofdm_wav, format="audio/wav")
            if ofdm_ber < 0.01:
                st.success(f"BER = {ofdm_ber:.4f}")
            elif ofdm_ber < 0.1:
                st.warning(f"BER = {ofdm_ber:.4f}")
            else:
                st.error(f"BER = {ofdm_ber:.4f}")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("SC-16QAM BER", f"{sc_16qam_ber:.4f}")
        col_m2.metric("OFDM-16QAM BER", f"{ofdm_ber:.4f}")
        col_m3.metric("多径延迟", f"{multipath_delay/FS*1e6:.1f} μs")
        col_m4.metric("噪声 σ", f"{noise_std:.2f}")

        st.markdown("### 📊 波形对比")
        t_ap = np.arange(num_samples_pcm) / AUDIO_SR
        fig_wc, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
        fig_wc.patch.set_facecolor("white")

        axes[0].plot(t_ap, pcm_original.astype(int), color="blue", linewidth=0.5)
        axes[0].set_ylabel("原始 PCM", fontsize=9)
        axes[0].set_title("原始音频", fontsize=11, fontweight="bold")
        axes[0].set_ylim(0, 255)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_ap, pcm_sc16qam.astype(int), color="purple", linewidth=0.5)
        axes[1].set_ylabel("SC-16QAM", fontsize=9)
        axes[1].set_title(f"单载波16QAM (BER={sc_16qam_ber:.4f})", fontsize=11, fontweight="bold")
        axes[1].set_ylim(0, 255)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t_ap, pcm_ofdm.astype(int), color="#2ecc71", linewidth=0.5)
        axes[2].set_xlabel("时间 (s)", fontsize=9)
        axes[2].set_ylabel("OFDM", fontsize=9)
        axes[2].set_title(f"OFDM-16QAM (BER={ofdm_ber:.4f})", fontsize=11, fontweight="bold")
        axes[2].set_ylim(0, 255)
        axes[2].grid(True, alpha=0.3)

        fig_wc.tight_layout()
        st.pyplot(fig_wc)
        plt.close(fig_wc)

        st.markdown("---")
        st.markdown("### 🧪 总结")
        if multipath_delay / FS <= T_cp:
            st.success(
                f"**OFDM优势明显**: CP ({T_cp*1e6:.1f}μs) > 多径 ({multipath_delay/FS*1e6:.1f}μs)。"
                f"SC-16QAM BER={sc_16qam_ber:.4f} vs OFDM BER={ofdm_ber:.4f}。"
                "OFDM通过并行化+CP，用极低代价解决了多径ISI问题。"
            )
        else:
            st.warning(
                f"CP不足。尝试降低多径延迟或增大OFDM符号周期。"
            )


# ============================================================================
# ============================================================================
#  COMPARE PAGE: 4QAM vs DSB-SC vs FM
# ============================================================================
# ============================================================================
else:
    section_header("🔊 4QAM vs DSB-SC vs FM — 数字与模拟对比")

    st.markdown(
        f"""
    同一段音频，同步经过 **4QAM (QPSK) 数字调制**、**DSB-SC 模拟调制** 和 **FM 频率调制**。
    
    三者使用完全相同的载波 ($f_c = {FC}$ Hz) 和相同的高斯噪声 ($\\sigma = {noise_std:.2f}$, 输入 SNR = {input_snr_db:.1f} dB)。
    **公平对比**: 相同载波、相同输入 SNR。4QAM 带宽约 32 kHz，DSB-SC 带宽约 8 kHz，FM (β=3) 卡森带宽约 32 kHz——4QAM 和 FM 带宽相等。
    
    点击下方播放器试听四种音频，调节侧边栏的 **噪声标准差 σ** 来感受不同噪声水平下的效果差异。
    """
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 📤 原始音频")
        st.audio(pcm_to_wav_bytes(pcm_original), format="audio/wav")
        st.caption(f"无噪声 · {audio_duration:.1f}s")

    with col2:
        st.markdown("### 📡 4QAM 解调")
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

    with col4:
        st.markdown("### 📡 FM 解调")
        st.audio(fm_wav, format="audio/wav")
        if fm_snr_db > 30:
            st.success(f"SNR = {fm_snr_db:.1f} dB")
        elif fm_snr_db > 15:
            st.warning(f"SNR = {fm_snr_db:.1f} dB")
        else:
            st.error(f"SNR = {fm_snr_db:.1f} dB")

    st.markdown("---")

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("输入 SNR", f"{input_snr_db:.1f} dB")
    col_m2.metric("4QAM BER", f"{ber:.4f}")
    col_m3.metric("DSB-SC SNR", f"{dsbsc_snr_db:.1f} dB")
    col_m4.metric("FM SNR", f"{fm_snr_db:.1f} dB")
    col_m5.metric("噪声 σ", f"{noise_std:.2f}")

    st.markdown("---")
    st.markdown("### 📊 音频波形对比")

    fig_wcmp, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    fig_wcmp.patch.set_facecolor("white")
    t_audio_plt = np.arange(num_samples_pcm) / AUDIO_SR

    axes[0].plot(t_audio_plt, pcm_original.astype(int), color="blue", linewidth=0.5)
    axes[0].set_ylabel("原始 PCM", fontsize=9)
    axes[0].set_title("原始音频", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 255)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_audio_plt, pcm_recovered.astype(int), color="purple", linewidth=0.5)
    axes[1].set_ylabel("4QAM 恢复", fontsize=9)
    axes[1].set_title(f"4QAM 解调 (BER={ber:.4f})", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 255)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_audio_plt, pcm_dsbsc_recovered.astype(int),
                 color="#e67e22", linewidth=0.5)
    axes[2].set_ylabel("DSB-SC 恢复", fontsize=9)
    axes[2].set_title(
        f"DSB-SC 解调 (SNR={dsbsc_snr_db:.1f} dB)", fontsize=11, fontweight="bold",
    )
    axes[2].set_ylim(0, 255)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t_audio_plt, pcm_fm_recovered.astype(int),
                 color="#2ecc71", linewidth=0.5)
    axes[3].set_xlabel("时间 (s)", fontsize=9)
    axes[3].set_ylabel("FM 恢复", fontsize=9)
    axes[3].set_title(
        f"FM 解调 (SNR={fm_snr_db:.1f} dB)", fontsize=11, fontweight="bold",
    )
    axes[3].set_ylim(0, 255)
    axes[3].grid(True, alpha=0.3)

    fig_wcmp.tight_layout()
    st.pyplot(fig_wcmp)
    plt.close(fig_wcmp)

    st.markdown("---")
    st.markdown("### 🧪 抗干扰行为总结")

    if noise_std < 0.01:
        st.success("**σ≈0**: 三者均完美。DSB-SC、FM 和 4QAM 都能无损恢复。")
    elif noise_std < 0.4:
        st.info(
            f"**σ={noise_std:.2f} (低噪声)**: DSB-SC 已出现轻微底噪 (SNR={dsbsc_snr_db:.0f} dB)，"
            f"FM 仍保持较高质量 (SNR={fm_snr_db:.0f} dB)，"
            f"4QAM 通过判决边界保护保持完美 (BER={ber:.4f})。"
            "**数字调制展示了门限效应前的绝对优势**。"
        )
    elif noise_std < 1.0:
        st.warning(
            f"**σ={noise_std:.2f} (中等噪声)**: DSB-SC 嘶嘶声明显 (SNR={dsbsc_snr_db:.0f} dB)，"
            f"FM 仍可辨认旋律 (SNR={fm_snr_db:.0f} dB)，"
            f"4QAM 开始出现比特错误 (BER={ber:.4f})，产生咔嚓声。"
        )
    else:
        st.error(
            f"**σ={noise_std:.2f} (高噪声)**: DSB-SC 仍能听到模糊旋律痕迹 "
            f"(SNR={dsbsc_snr_db:.0f} dB)。"
            f"FM 出现严重嘶嘶声 (SNR={fm_snr_db:.0f} dB)。"
            f"4QAM BER={ber:.4f}，音频几乎崩溃。"
            "**模拟调制的渐进退化优势在高噪声下显现**。"
        )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(
    "💡 提示: 使用左侧边栏切换查看页面、调节噪声 σ / 多径参数观察效果变化。"
    " | 调制通信原理可视化: 4QAM | DSB-SC | FM | 16QAM vs OFDM 多径 | OFDM自适应抗干扰"
)
