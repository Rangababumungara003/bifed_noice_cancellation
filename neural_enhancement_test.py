
"""
BiFed-S: Neural Speech Enhancement Comparison
DeepFilterNet3 vs Noisy Input vs NoiseReduce 0.8

DeepFilterNet3 processes audio at 48 kHz.
Metrics are calculated at the clean reference sample rate.
"""

from pathlib import Path
from math import gcd
import csv

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torch
from scipy.signal import resample_poly

from df.enhance import enhance, init_df, load_audio, save_audio


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path("outputs")
OUT = ROOT / "neural_outputs"
OUT.mkdir(parents=True, exist_ok=True)

REFERENCE = ROOT / "clean.wav"

MODEL_SR = 48000

SCENARIOS = {
    "fan": {
        "noisy": ROOT / "fan_noisy.wav",
        "baseline": ROOT / "fan_reduced_0.8.wav",
    },
    "traffic_like": {
        "noisy": ROOT / "traffic_like_noisy.wav",
        "baseline": ROOT / "traffic_like_reduced_0.8.wav",
    },
    "mixed_echo": {
        "noisy": ROOT / "mixed_echo_noisy.wav",
        "baseline": ROOT / "mixed_echo_reduced_0.8.wav",
    },
}


# --------------------------------------------------
# AUDIO UTILITIES
# --------------------------------------------------

def read_audio(path):
    audio, sr = sf.read(str(path))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    return audio.astype(np.float64), sr


def resample_audio(audio, source_sr, target_sr):
    if source_sr == target_sr:
        return audio

    divisor = gcd(int(source_sr), int(target_sr))

    return resample_poly(
        audio,
        target_sr // divisor,
        source_sr // divisor,
    )


def fit_length(audio, length):
    if len(audio) > length:
        return audio[:length]

    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)))

    return audio


def calculate_metrics(reference, candidate):
    # Center the signals for SI-SDR and correlation.
    ref_centered = reference - np.mean(reference)
    candidate_centered = candidate - np.mean(candidate)

    if (
        np.std(ref_centered) > 1e-10
        and np.std(candidate_centered) > 1e-10
    ):
        correlation = float(
            np.corrcoef(ref_centered, candidate_centered)[0, 1]
        )
    else:
        correlation = float("nan")

    ref_energy = np.sum(ref_centered ** 2) + 1e-12

    scale = (
        np.dot(candidate_centered, ref_centered)
        / ref_energy
    )

    target = scale * ref_centered
    residual = candidate_centered - target

    sisdr = 10 * np.log10(
        (np.sum(target ** 2) + 1e-12)
        / (np.sum(residual ** 2) + 1e-12)
    )

    # Reference-relative SNR is sensitive to alignment and gain.
    error = reference - candidate

    relative_snr = 10 * np.log10(
        (np.sum(reference ** 2) + 1e-12)
        / (np.sum(error ** 2) + 1e-12)
    )

    rms = float(np.sqrt(np.mean(candidate ** 2)))

    return {
        "RMS": rms,
        "Correlation": correlation,
        "SI-SDR_dB": float(sisdr),
        "Reference_relative_SNR_dB": float(relative_snr),
    }


# --------------------------------------------------
# DEEPFILTERNET AUDIO LOADING
# --------------------------------------------------

def load_model_audio(path):
    """
    Load audio using DeepFilterNet's utility at 48 kHz.
    Handle versions returning either a tensor or a tuple.
    """
    loaded = load_audio(str(path), sr=MODEL_SR)

    audio = loaded[0] if isinstance(loaded, tuple) else loaded

    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio, dtype=torch.float32)

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    return audio.to(dtype=torch.float32)


# --------------------------------------------------
# MAIN PROCESS
# --------------------------------------------------

def main():
    required_files = [REFERENCE]

    for scenario_files in SCENARIOS.values():
        required_files.extend([
            scenario_files["noisy"],
            scenario_files["baseline"],
        ])

    missing = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        print("ERROR: Required audio files were not found:")
        for filename in missing:
            print(f"  - {filename}")
        print("\nCheck the filenames inside the outputs folder.")
        return

    print("Loading clean reference...", flush=True)

    reference, reference_sr = read_audio(REFERENCE)

    if reference_sr <= 0:
        raise ValueError("Invalid reference sample rate.")

    print(f"Reference sample rate: {reference_sr} Hz", flush=True)

    print("Loading DeepFilterNet3...", flush=True)

    model, df_state, _ = init_df()

    # Verify the installed model's actual sample rate.
    model_sr = df_state.sr()

    if model_sr != MODEL_SR:
        raise RuntimeError(
            f"Expected a {MODEL_SR} Hz model, "
            f"but DeepFilterNet reports {model_sr} Hz."
        )

    print(f"DeepFilterNet sample rate: {model_sr} Hz", flush=True)

    results = []

    for name, files in SCENARIOS.items():
        print(f"\nProcessing scenario: {name}", flush=True)

        # Load original signals.
        noisy_original, noisy_sr = read_audio(files["noisy"])
        baseline_original, baseline_sr = read_audio(files["baseline"])

        # Resample noisy input to the model's expected 48 kHz.
        noisy_tensor = load_model_audio(files["noisy"])

        # Enhance audio.
        with torch.no_grad():
            enhanced_tensor = enhance(
                model,
                df_state,
                noisy_tensor,
            )

        # Save neural output at 48 kHz.
        neural_path = OUT / f"{name}_deepfilternet.wav"

        save_audio(
            str(neural_path),
            enhanced_tensor,
            MODEL_SR,
        )

        print(f"Saved neural audio: {neural_path}", flush=True)

        # Read saved enhanced audio and its actual sample rate.
        enhanced_audio, enhanced_sr = read_audio(neural_path)

        # Resample every signal to the reference rate for fair comparison.
        noisy_metric = resample_audio(
            noisy_original, noisy_sr, reference_sr
        )

        baseline_metric = resample_audio(
            baseline_original, baseline_sr, reference_sr
        )

        enhanced_metric = resample_audio(
            enhanced_audio, enhanced_sr, reference_sr
        )

        # Crop or pad to the clean reference duration.
        length = len(reference)

        noisy_metric = fit_length(noisy_metric, length)
        baseline_metric = fit_length(baseline_metric, length)
        enhanced_metric = fit_length(enhanced_metric, length)

        comparisons = [
            ("Noisy", noisy_metric),
            ("NoiseReduce_0.8", baseline_metric),
            ("DeepFilterNet", enhanced_metric),
        ]

        for method, audio in comparisons:
            row = {
                "Scenario": name,
                "Method": method,
                "Reference_sample_rate_Hz": reference_sr,
                "Model_sample_rate_Hz": model_sr,
            }

            row.update(calculate_metrics(reference, audio))
            results.append(row)

    # --------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------

    csv_path = OUT / "neural_vs_baseline_metrics.csv"

    fields = [
        "Scenario",
        "Method",
        "Reference_sample_rate_Hz",
        "Model_sample_rate_Hz",
        "RMS",
        "Correlation",
        "SI-SDR_dB",
        "Reference_relative_SNR_dB",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nMetrics saved: {csv_path}", flush=True)

    # --------------------------------------------------
    # CREATE SI-SDR COMPARISON CHART
    # --------------------------------------------------

    scenarios = list(SCENARIOS.keys())
    methods = ["Noisy", "NoiseReduce_0.8", "DeepFilterNet"]

    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for index, method in enumerate(methods):
        values = []

        for scenario in scenarios:
            match = next(
                row["SI-SDR_dB"]
                for row in results
                if row["Scenario"] == scenario
                and row["Method"] == method
            )
            values.append(match)

        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=method,
        )

    ax.set_xlabel("Noise scenario")
    ax.set_ylabel("SI-SDR (dB)")
    ax.set_title("Speech Enhancement Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.axhline(0, linewidth=0.8)
    ax.legend()

    fig.tight_layout()

    chart_path = OUT / "neural_vs_baseline_sisdr.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print(f"Chart saved: {chart_path}", flush=True)

    print("\nAll scenarios processed successfully.", flush=True)
    print(f"Results folder: {OUT.resolve()}", flush=True)


if __name__ == "__main__":
    main()

