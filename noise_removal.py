"""
BiFed-S: DeepFilterNet3 Noise-Removal Tool
Run with: python noise_removal.py
Prompts for an audio path and saves a cleaned WAV under outputs/cleaned/.
The original input is never overwritten.
"""
from pathlib import Path
from math import gcd
import sys

import soundfile as sf
import torch
from scipy.signal import resample_poly
from df.enhance import enhance, init_df, load_audio, save_audio

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "cleaned"


def ask_for_input_path() -> Path:
    raw = input("Enter the path to your noisy audio file: ").strip().strip('"').strip("'")
    if not raw:
        raise ValueError("No input path was provided.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path


def load_audio_tensor(path: Path, sample_rate: int) -> torch.Tensor:
    loaded = load_audio(str(path), sr=sample_rate)
    audio = loaded[0] if isinstance(loaded, tuple) else loaded
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio, dtype=torch.float32)
    audio = audio.to(dtype=torch.float32)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim != 2:
        raise ValueError(f"Unexpected audio tensor shape: {tuple(audio.shape)}")
    if audio.numel() == 0:
        raise ValueError("The input audio file contains no samples.")
    return audio.contiguous()


def make_output_path(input_path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{input_path.stem}_cleaned.wav"
    counter = 1
    while output.exists():
        output = OUTPUT_DIR / f"{input_path.stem}_cleaned_{counter}.wav"
        counter += 1
    return output


def main() -> int:
    try:
        input_path = ask_for_input_path()
        info = sf.info(str(input_path))
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError("The input file has invalid audio metadata.")

        print(f"\nInput: {input_path}")
        print(f"Original sample rate: {info.samplerate} Hz")
        print(f"Channels: {info.channels}")
        print(f"Duration: {info.duration:.2f} seconds")
        print("\nLoading DeepFilterNet3...")

        model, df_state, _ = init_df()
        model_sr = int(df_state.sr())
        print(f"Model sample rate: {model_sr} Hz")

        # DeepFilterNet's loader resamples to the model sample rate.
        audio = load_audio_tensor(input_path, model_sr)
        print("Enhancing audio...")

        with torch.no_grad():
            enhanced = enhance(model, df_state, audio)

        output_path = make_output_path(input_path)
        save_audio(str(output_path), enhanced, model_sr)

        saved_info = sf.info(str(output_path))
        if saved_info.frames <= 0 or saved_info.samplerate != model_sr:
            raise RuntimeError(
                f"Output validation failed: sample rate is {saved_info.samplerate} Hz."
            )

        print("\nNoise removal completed.")
        print(f"Enhanced file: {output_path.resolve()}")
        print(f"Output sample rate: {saved_info.samplerate} Hz")
        print(f"Output duration: {saved_info.duration:.2f} seconds")
        print("Original file was not modified.")
        return 0

    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "Make sure .venv is active and DeepFilterNet, PyTorch, "
            "SoundFile, and SciPy are installed. WAV input is recommended.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
