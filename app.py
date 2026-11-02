import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from df.enhance import enhance, init_df, load_audio


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "outputs" / "uploads"
CLEANED_DIR = BASE_DIR / "outputs" / "cleaned"

ALLOWED_EXTENSIONS = {"wav", "flac", "ogg", "aif", "aiff"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# Initialize the enhancement model once when the app starts.
print("Loading DeepFilterNet model...")
model, df_state, _ = init_df()
MODEL_SR = int(df_state.sr())
print(f"DeepFilterNet loaded successfully. Model sample rate: {MODEL_SR} Hz")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def allowed_file(filename):
    """Check whether the uploaded file has a supported extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def to_numpy(audio):
    """Convert a PyTorch tensor or array into a NumPy array."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    return np.asarray(audio)


def trim_silence(
    audio,
    sample_rate,
    threshold_db=-45,
    frame_ms=20,
    padding_ms=80,
):
    """
    Trim silence only from the beginning and end of audio.

    Pauses inside the recording are preserved.

    DeepFilterNet commonly represents audio as (channels, samples).
    This function supports mono, channels-first, and samples-first arrays.
    """
    audio = to_numpy(audio).astype(np.float32, copy=False)

    if audio.size == 0:
        return audio

    if audio.ndim not in (1, 2):
        raise ValueError(
            f"Unsupported audio dimensions: {audio.shape}"
        )

    # Convert to a temporary samples-first representation for analysis.
    was_mono = audio.ndim == 1
    channels_first = False

    if was_mono:
        samples_first = audio.reshape(-1, 1)
    else:
        # Most audio tensors returned by DeepFilterNet are channels-first.
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            channels_first = True
            samples_first = audio.T
        else:
            samples_first = audio

    number_of_samples = samples_first.shape[0]

    if number_of_samples == 0:
        return audio

    frame_length = max(1, int(sample_rate * frame_ms / 1000))
    padding_samples = int(sample_rate * padding_ms / 1000)

    # Calculate RMS for each frame, using all channels.
    frame_rms = []

    for start in range(0, number_of_samples, frame_length):
        frame = samples_first[start:start + frame_length]

        if frame.size == 0:
            continue

        rms = np.sqrt(np.mean(np.square(frame, dtype=np.float64)))
        frame_rms.append(rms)

    if not frame_rms:
        return audio

    frame_rms = np.asarray(frame_rms)

    # Convert dB threshold to a linear amplitude threshold.
    threshold = 10 ** (threshold_db / 20)

    active_frames = np.flatnonzero(frame_rms >= threshold)

    # If the entire recording is below the threshold, leave it unchanged.
    if active_frames.size == 0:
        return audio

    first_active_sample = int(active_frames[0] * frame_length)
    last_active_sample = min(
        number_of_samples,
        int((active_frames[-1] + 1) * frame_length),
    )

    # Preserve a small amount of audio around the detected speech.
    start_sample = max(0, first_active_sample - padding_samples)
    end_sample = min(
        number_of_samples,
        last_active_sample + padding_samples,
    )

    trimmed = samples_first[start_sample:end_sample]

    # Restore the original channel layout.
    if was_mono:
        return trimmed[:, 0]

    if channels_first:
        return trimmed.T

    return trimmed


def save_audio_wav(audio, output_path, sample_rate):
    """
    Save audio as WAV.

    soundfile expects a 2D multichannel array in (samples, channels) order.
    """
    audio = to_numpy(audio).astype(np.float32, copy=False)

    if audio.ndim == 2:
        # Convert likely channels-first audio to samples-first.
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
    elif audio.ndim != 1:
        raise ValueError(f"Cannot save audio with shape {audio.shape}")

    # Avoid writing invalid values.
    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    sf.write(str(output_path), audio, sample_rate, subtype="PCM_16")


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Display the upload page."""
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_audio():
    """Enhance an uploaded audio file and return the result page."""

    if "audio" not in request.files:
        return render_template(
            "index.html",
            error="No audio file was submitted. Please choose a file.",
        ), 400

    uploaded_file = request.files["audio"]

    if not uploaded_file or uploaded_file.filename == "":
        return render_template(
            "index.html",
            error="Please select an audio file first.",
        ), 400

    if not allowed_file(uploaded_file.filename):
        return render_template(
            "index.html",
            error="Unsupported format. Upload WAV, FLAC, OGG, AIFF, or AIF.",
        ), 400

    # Generate a unique filename to avoid overwriting existing files.
    original_name = secure_filename(uploaded_file.filename)
    unique_id = uuid.uuid4().hex[:10]

    input_path = UPLOAD_DIR / f"{unique_id}_{original_name}"
    output_filename = f"cleaned_{unique_id}.wav"
    output_path = CLEANED_DIR / output_filename

    try:
        uploaded_file.save(str(input_path))

        print(f"\nProcessing file: {original_name}")
        print("Loading audio...")

        # IMPORTANT:
        # load_audio returns both the audio and its sample rate.
        audio, loaded_sr = load_audio(
            str(input_path),
            sr=MODEL_SR,
        )

        print("Audio loaded successfully.")
        print("Loaded sample rate:", loaded_sr)
        print("Audio shape:", tuple(audio.shape))

        print("Reducing background noise...")

        # Run the enhancement model.
        with torch.inference_mode():
            enhanced = enhance(model, df_state, audio)

        print("Noise reduction completed.")

        enhanced_np = to_numpy(enhanced)

        print("Trimming leading and trailing silence...")

        trimmed_audio = trim_silence(
            enhanced_np,
            sample_rate=MODEL_SR,
            threshold_db=-30,
            frame_ms=20,
            padding_ms=50,
        )

        print("Saving enhanced WAV file...")

        save_audio_wav(
            trimmed_audio,
            output_path,
            MODEL_SR,
        )

        print("Saved:", output_path)

        return render_template(
            "index.html",
            success="Audio processed successfully!",
            original_filename=original_name,
            output_filename=output_filename,
            sample_rate=MODEL_SR,
        )

    except Exception:
        app.logger.exception("Audio processing failed.")

        return render_template(
            "index.html",
            error=(
                "Audio processing failed. Check the terminal for the "
                "full error message and traceback."
            ),
        ), 500

    finally:
        # Remove the temporary upload; retain the processed WAV.
        try:
            if input_path.exists():
                input_path.unlink()
        except OSError:
            app.logger.warning("Could not remove temporary upload.")


@app.route("/play/<filename>", methods=["GET"])
def play_audio(filename):
    """Serve the enhanced WAV file for browser playback."""
    safe_name = secure_filename(filename)
    file_path = CLEANED_DIR / safe_name

    if not file_path.is_file():
        return "Audio file not found.", 404

    return send_from_directory(
        str(CLEANED_DIR),
        safe_name,
        mimetype="audio/wav",
        as_attachment=False,
    )


@app.route("/download/<filename>", methods=["GET"])
def download_audio(filename):
    """Download the enhanced WAV file."""
    safe_name = secure_filename(filename)
    file_path = CLEANED_DIR / safe_name

    if not file_path.is_file():
        return "Audio file not found.", 404

    return send_from_directory(
        str(CLEANED_DIR),
        safe_name,
        mimetype="audio/wav",
        as_attachment=True,
        download_name=safe_name,
    )


@app.errorhandler(413)
def file_too_large(_error):
    """Display an error when the upload exceeds the size limit."""
    return render_template(
        "index.html",
        error="The file is too large. Maximum upload size is 50 MB.",
    ), 413


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )