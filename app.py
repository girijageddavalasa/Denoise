import onnxruntime as ort
import numpy as np
import librosa
import soundfile as sf
import os
from flask import Flask, request, send_file, render_template, jsonify
from io import BytesIO
from pydub import AudioSegment # <--- NEW IMPORT

# --- Flask App Setup ---
# This setup tells Flask that the 'static_folder' is the
# current directory ('.') where app.py is.
app = Flask(__name__, static_folder='.', static_url_path='')

# --- Load ONNX Model (once, at startup) ---
# IMPORTANT: Model file must be in the same directory!
MODEL_PATH = 'denoiser_tcn.onnx' 
SESSION = None
INPUT_NAME = None
if os.path.exists(MODEL_PATH):
    try:
        SESSION = ort.InferenceSession(MODEL_PATH)
        # This will get 'noisy_wave' (as seen in your log)
        INPUT_NAME = SESSION.get_inputs()[0].name 
        print(f"Successfully loaded model '{MODEL_PATH}'")
        print(f"Model Input Name: {INPUT_NAME}")
    except Exception as e:
        print(f"!!! ERROR loading ONNX model: {e}")
        print("!!! Server will run, but /denoise endpoint will fail.")
else:
    print(f"!!! WARNING: Model file not found at '{MODEL_PATH}'")
    print("!!! Server will run, but /denoise endpoint will fail.")


# --- Frontend Route ---
@app.route('/')
def index():
    """Serves the main index.html file."""
    
    # *** THIS IS THE FIX for 'TemplateNotFound' ***
    # Instead of 'render_template', we use 'send_static_file'
    # to send 'index.html' from our 'static_folder' (which is '.')
    return app.send_static_file('index.html')


# --- Denoise API Route ---
@app.route('/denoise', methods=['POST'])
def denoise_audio():
    """
    Handles the audio file upload, processes it, and returns the cleaned audio.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not SESSION:
        return jsonify({"error": "Model is not loaded on the server"}), 500

    try:
        # --- 1. Load Audio & Convert ---
        # Read the uploaded file (e.g., 'webm' from browser)
        input_io = BytesIO(file.read())
        
        # *** NEW CONVERSION STEP ***
        # This uses pydub (which uses FFmpeg) to read the browser's
        # audio format (like 'webm') and convert it.
        try:
            audio_segment = AudioSegment.from_file(input_io)
        except Exception as pydub_err:
            print(f"PYDUB ERROR: {pydub_err}")
            print("This likely means FFmpeg is not installed or not in your system's PATH.")
            return jsonify({"error": "Server failed to convert audio. Is FFmpeg installed?"}), 500

        # Now, export the audio as a 'wav' file into a new buffer
        wav_io = BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0) # Rewind the buffer
        
        # Librosa loads the audio *from the new wav buffer*.
        audio, sr = librosa.load(wav_io, sr=None) 
        # *** END NEW CONVERSION STEP ***

        # --- 2. Prepare Model Input (based on your script) ---
        
        # *** FIX for ONNXRuntimeError ***
        # The error "Invalid rank... Got: 3 Expected: 2" means the model
        # wants a 2-dimensional input, not 3-dimensional.
        # The expected shape is likely (batch_size, length) or (1, length).
        
        # Old code (created 3D tensor):
        # audio_input = np.expand_dims(audio, axis=0)      # (1, length)
        # audio_input = np.expand_dims(audio_input, axis=0) # (1, 1, length)

        # New code (creates 2D tensor):
        audio_input = np.expand_dims(audio, axis=0)      # (1, length)
        audio_input = audio_input.astype(np.float32)

        # --- 3. Run Inference ---
        # Use the INPUT_NAME we found at startup ('noisy_wave')
        output = SESSION.run(None, {INPUT_NAME: audio_input})

        # --- 4. Process Output ---
        cleaned_audio = np.squeeze(output[0]) # shape (length,)

        # --- 5. Save Cleaned Audio to a new buffer ---
        output_io = BytesIO()
        sf.write(output_io, cleaned_audio, sr, format='WAV')
        output_io.seek(0) # Rewind buffer to the beginning

        # --- 6. Send File Back ---
        return send_file(
            output_io, 
            as_attachment=True, 
            download_name='cleaned.wav', 
            mimetype='audio/wav'
        )

    except Exception as e:
        # This is where the "Format not recognised" error was coming from
        print(f"Error processing audio: {e}") 
        return jsonify({"error": str(e)}), 500


# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server... go to http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

