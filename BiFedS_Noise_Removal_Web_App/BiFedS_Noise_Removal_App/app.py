from pathlib import Path
from uuid import uuid4
from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
import soundfile as sf
import torch
from df.enhance import enhance, init_df, load_audio, save_audio

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "cleaned"
UPLOADS = ROOT / "uploads"
ALLOWED = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
MAX_MB = 50
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
OUT.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)

print("Loading DeepFilterNet3...")
model, df_state, _ = init_df()
MODEL_SR = int(df_state.sr())
print(f"Ready at {MODEL_SR} Hz.")

@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error=f"File is too large. Maximum size: {MAX_MB} MB."), 413

@app.route("/", methods=["GET", "POST"])
def index():
    error = result = None
    if request.method == "POST":
        file = request.files.get("audio")
        if not file or not file.filename:
            error = "Please choose an audio file."
        elif Path(file.filename).suffix.lower() not in ALLOWED:
            error = "Unsupported format. Upload WAV, FLAC, OGG, AIFF, or AIF. Convert MP3/MPEG to WAV first."
        else:
            original = secure_filename(file.filename) or "audio"
            source = UPLOADS / f"{uuid4().hex}_{original}"
            name = f"{Path(original).stem}_cleaned_{uuid4().hex[:8]}.wav"
            destination = OUT / name
            try:
                file.save(source)
                info = sf.info(str(source))
                if info.frames <= 0 or info.samplerate <= 0:
                    raise ValueError("The file has invalid audio metadata.")
                loaded = load_audio(str(source), sr=MODEL_SR)
                audio = loaded[0] if isinstance(loaded, tuple) else loaded
                if not isinstance(audio, torch.Tensor):
                    audio = torch.as_tensor(audio, dtype=torch.float32)
                audio = audio.to(dtype=torch.float32)
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)
                if audio.ndim != 2 or audio.numel() == 0:
                    raise ValueError("Unexpected or empty audio data.")
                with torch.no_grad():
                    cleaned = enhance(model, df_state, audio.contiguous())
                save_audio(str(destination), cleaned, MODEL_SR)
                check = sf.info(str(destination))
                if check.frames <= 0 or check.samplerate != MODEL_SR:
                    raise RuntimeError("Output validation failed.")
                result = {
                    "original": original, "name": name,
                    "duration": f"{check.duration:.2f}",
                    "rate": check.samplerate,
                    "play": url_for("play_file", filename=name),
                    "download": url_for("download_file", filename=name)
                }
            except Exception as exc:
                error = f"Could not process this file. Try converting it to WAV. Details: {exc}"
                if destination.exists():
                    destination.unlink()
            finally:
                if source.exists():
                    source.unlink()
    return render_template("index.html", error=error, result=result)

@app.route("/download/<path:filename>")
def download_file(filename):
    safe = secure_filename(filename)
    if safe != filename or not safe or not (OUT / safe).is_file():
        return "File not found", 404
    return send_from_directory(OUT, safe, as_attachment=True, download_name=safe)

@app.route("/play/<path:filename>")
def play_file(filename):
    safe = secure_filename(filename)
    if safe != filename or not safe or not (OUT / safe).is_file():
        return "File not found", 404
    return send_from_directory(OUT, safe, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
