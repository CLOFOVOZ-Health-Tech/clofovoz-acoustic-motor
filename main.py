from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import librosa
import numpy as np
import tempfile
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from functools import lru_cache

app = FastAPI(title="CLOFOVOZ Acoustic Engine")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    audio_url: str
    user_id: str

class AnalysisResponse(BaseModel):
    jitter: float
    shimmer: float
    f0_mean: float
    f0_std: float
    quality_score: float
    feedback: str
    spectrogram_url: str

@lru_cache(maxsize=1)
def get_frequency_bounds():
    """Cache frequency bounds to avoid recalculation"""
    return librosa.note_to_hz('C2'), librosa.note_to_hz('C7')

def calculate_jitter(f0_series):
    """Vectorized jitter calculation with early exit"""
    n = len(f0_series)
    if n < 2:
        return 0.0
    mean_f0 = np.mean(f0_series)
    if mean_f0 == 0:
        return 0.0
    # Vectorized: calculate mean of absolute differences directly
    mean_diff = np.mean(np.abs(np.diff(f0_series)))
    return (mean_diff / mean_f0) * 100

def calculate_shimmer(amplitude_series):
    """Vectorized shimmer calculation with early exit"""
    n = len(amplitude_series)
    if n < 2:
        return 0.0
    mean_amp = np.mean(amplitude_series)
    if mean_amp == 0:
        return 0.0
    # Vectorized: calculate mean of absolute differences directly
    mean_diff = np.mean(np.abs(np.diff(amplitude_series)))
    return (mean_diff / mean_amp) * 100

@app.get("/")
def root():
    return {"message": "CLOFOVOZ Acoustic Engine API", "status": "running"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_voice(request: AnalysisRequest):
    temp_path = None
    try:
        # Descargar audio con timeout y streaming
        response = requests.get(request.audio_url, timeout=30, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="No se pudo descargar el audio")
        
        # Guardar temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
            temp_path = f.name
        
        # Analizar con parámetros optimizados
        y, sr = librosa.load(temp_path, sr=None, mono=True)
        
        # F0 con límites cacheados
        fmin, fmax = get_frequency_bounds()
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr
        )
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            f0 = np.array([220.0])
        
        f0_mean = float(np.mean(f0))
        f0_std = float(np.std(f0))
        
        # Jitter y Shimmer
        amplitude = librosa.feature.rms(y=y)[0]
        jitter = calculate_jitter(f0)
        shimmer = calculate_shimmer(amplitude)
        
        # Calidad
        quality_score = max(0, min(10, 10 - (jitter * 2) - (shimmer * 2)))
        
        # Generar Espectrograma optimizado
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
        ax.set_title('Espectrogramo')
        fig.colorbar(img, format='%+2.0f dB', ax=ax)
        
        # Guardar en memoria como base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        buf.close()
        
        spectrogram_url = f"data:image/png;base64,{image_base64}"
        
        # Feedback
        feedback = []
        if jitter < 1.0:
            feedback.append("✅ Excelente estabilidad de frecuencia")
        elif jitter < 2.0:
            feedback.append("⚠️ Ligera inestabilidad en la frecuencia")
        else:
            feedback.append("❌ Alta variación en la frecuencia")
        
        if shimmer < 3.0:
            feedback.append("✅ Buen control de volumen")
        elif shimmer < 5.0:
            feedback.append("️ Variaciones en la intensidad")
        else:
            feedback.append("❌ Inconsistencia en el volumen")
        
        feedback_text = " | ".join(feedback)
        
        return AnalysisResponse(
            jitter=round(jitter, 4),
            shimmer=round(shimmer, 4),
            f0_mean=round(f0_mean, 2),
            f0_std=round(f0_std, 2),
            quality_score=round(quality_score, 2),
            feedback=feedback_text,
            spectrogram_url=spectrogram_url
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpieza garantizada
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
