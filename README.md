# BiFed-S Noise Removal Web App

## Run in your existing project environment
1. Copy `app.py` and the `templates` folder into your existing `BiFedS-Voice-Prototype` project folder.
2. Open PowerShell in that folder and activate your existing environment:
   `.\.venv\Scripts\Activate.ps1`
3. Install Flask:
   `python -m pip install Flask`
4. Start the app:
   `python app.py`
5. Open `http://127.0.0.1:5000` in your browser.

DeepFilterNet3 loads once at startup. It uses the model's sample rate (expected to be 48 kHz), and enhanced WAV files are saved in `outputs\cleaned\`. Temporary upload copies are removed after processing; original user files are not overwritten.

Supported uploads in this first version: WAV, FLAC, OGG, AIFF, AIF, provided the installed audio decoder can read them. MP3/MPEG upload is intentionally not enabled because decoding support varies; convert those files to WAV first.

This is a local prototype, not a production deployment. Do not expose Flask's development server directly to the internet.
