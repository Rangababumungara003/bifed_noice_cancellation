
import numpy as np
import soundfile as sf
import noisereduce as nr
from pathlib import Path

INPUT_FILE = "clean.wav"
OUTPUT_DIR = Path("outputs")
TARGET_SNR_DB = 5

OUTPUT_DIR.mkdir(exist_ok=True)

speech, sr = sf.read(INPUT_FILE)

if speech.ndim == 2:
    speech = speech.mean(axis=1)

speech = speech.astype(np.float32)
speech /= max(np.max(np.abs(speech)), 1e-8)

rng = np.random.default_rng(42)
t = np.arange(len(speech)) / sr


def add_noise_at_snr(clean, noise, snr_db):
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(
        clean_power / (noise_power * 10 ** (snr_db / 10))
    )
    return clean + scale * noise


def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    return audio


def make_fan_hum():
    return (
        0.7 * np.sin(2 * np.pi * 100 * t)
        + 0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 300 * t)
    ).astype(np.float32)


def make_traffic_like_noise():
    white = rng.normal(0, 1, len(speech))
    kernel = np.ones(120) / 120
    low = np.convolve(white, kernel, mode="same")
    low /= np.std(low) + 1e-8
    broadband = rng.normal(0, 0.25, len(speech))
    return (low + broadband).astype(np.float32)


def add_echo(audio, delay_seconds=0.18, decay=0.35):
    delay = int(delay_seconds * sr)
    echoed = audio.copy()
    if 0 < delay < len(audio):
        echoed[delay:] += decay * audio[:-delay]
    return echoed


scenarios = {
    "fan": add_noise_at_snr(
        speech, make_fan_hum(), TARGET_SNR_DB
    ),
    "traffic_like": add_noise_at_snr(
        speech, make_traffic_like_noise(), TARGET_SNR_DB
    ),
    "mixed_echo": add_echo(
        add_noise_at_snr(
            speech,
            make_fan_hum() + make_traffic_like_noise(),
            TARGET_SNR_DB
        )
    )
}

sf.write(OUTPUT_DIR / "clean.wav", speech, sr)


for name, noisy in scenarios.items():
    noisy = normalize_audio(noisy)

    sf.write(OUTPUT_DIR / f"{name}_noisy.wav", noisy, sr)

    for strength in [0.8, 0.95, 1.0]:
        enhanced = nr.reduce_noise(
            y=noisy,
            sr=sr,
            stationary=False,
            prop_decrease=strength
        )

        enhanced = normalize_audio(enhanced)

        output_name = f"{name}_reduced_{strength}.wav"
        sf.write(OUTPUT_DIR / output_name, enhanced, sr)

        print(f"Saved: {output_name}")

print(f"All files saved to: {OUTPUT_DIR.resolve()}")
