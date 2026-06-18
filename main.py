from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import librosa
import numpy as np
from typing import Optional

app = FastAPI(title="CLOFOVOZ Acoustic Engine")

# CORS para permitir conexiones desde localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, cambia esto por tu dominio
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

def calculate_jitter(f0_series):
    """Calcula Jitter (variación de frecuencia)"""
    if len(f0_series) < 2:
        return 0.0
    differences = np.abs(np.diff(f0_series))
    mean_diff = np.mean(differences)
    mean_f0 = np.mean(f0_series)
    if mean_f0 == 0:
        return 0.0
    return (mean_diff / mean_f0) * 100

def calculate_shimmer(amplitude_series):
    """Calcula Shimmer (variación de amplitud)"""
    if len(amplitude_series) < 2:
        return 0.0
    differences = np.abs(np.diff(amplitude_series))
    mean_diff = np.mean(differences)
    mean_amp = np.mean(amplitude_series)
    if mean_amp == 0:
        return 0.0
    return (mean_diff / mean_amp) * 100

def extract_vocal_features(audio_path):
    """Extrae características vocales del audio"""
    try:
        # Cargar audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extraer F0 (frecuencia fundamental) usando YIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Filtrar valores NaN
        f0 = f0[~np.isnan(f0)]
        
        if len(f0) == 0:
            f0 = np.array([220.0])  # Valor por defecto
        
        # Calcular métricas
        f0_mean = float(np.mean(f0))
        f0_std = float(np.std(f0))
        
        # Calcular Jitter y Shimmer
        jitter = calculate_jitter(f0)
        
        # Amplitud RMS para shimmer
        amplitude = librosa.feature.rms(y=y)[0]
        shimmer = calculate_shimmer(amplitude)
        
        # Calcular calidad vocal (0-10)
        # Menos jitter/shimmer = mejor calidad
        quality_score = max(0, min(10, 10 - (jitter * 2) - (shimmer * 2)))
        
        return {
            'jitter': round(jitter, 4),
            'shimmer': round(shimmer, 4),
            'f0_mean': round(f0_mean, 2),
            'f0_std': round(f0_std, 2),
            'quality_score': round(quality_score, 2)
        }
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return {
            'jitter': 0.0,
            'shimmer': 0.0,
            'f0_mean': 220.0,
            'f0_std': 0.0,
            'quality_score': 5.0
        }

def generate_feedback(metrics):
    """Genera feedback basado en las métricas"""
    feedback = []
    
    # Feedback basado en Jitter
    if metrics['jitter'] < 1.0:
        feedback.append("✅ Excelente estabilidad de frecuencia")
    elif metrics['jitter'] < 2.0:
        feedback.append("⚠️ Ligera inestabilidad en la frecuencia. Practica sostener notas largas")
    else:
        feedback.append("❌ Alta variación en la frecuencia. Trabaja en tu afinación")
    
    # Feedback basado en Shimmer
    if metrics['shimmer'] < 3.0:
        feedback.append("✅ Buen control de volumen")
    elif metrics['shimmer'] < 5.0:
        feedback.append("⚠️ Variaciones en la intensidad. Practica respiración diafragmática")
    else:
        feedback.append("❌ Inconsistencia en el volumen. Trabaja tu soporte respiratorio")
    
    # Feedback basado en F0
    if metrics['f0_mean'] < 200:
        feedback.append(f"🎵 Rango vocal grave (F0: {metrics['f0_mean']} Hz)")
    elif metrics['f0_mean'] < 400:
        feedback.append(f"🎵 Rango vocal medio (F0: {metrics['f0_mean']} Hz)")
    else:
        feedback.append(f"🎵 Rango vocal agudo (F0: {metrics['f0_mean']} Hz)")
    
    return " | ".join(feedback)

@app.get("/")
def root():
    return {"message": "CLOFOVOZ Acoustic Engine API", "status": "running"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_voice(request: AnalysisRequest):
    """
    Analiza un archivo de audio y devuelve métricas vocales
    """
    try:
        # Descargar audio desde la URL
        audio_url = request.audio_url
        response = requests.get(audio_url)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="No se pudo descargar el audio")
        
        # Guardar temporalmente
        temp_path = "temp_audio.wav"
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        # Extraer características
        metrics = extract_vocal_features(temp_path)
        
        # Generar feedback
        feedback = generate_feedback(metrics)
        
        # Limpiar archivo temporal
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return AnalysisResponse(
            jitter=metrics['jitter'],
            shimmer=metrics['shimmer'],
            f0_mean=metrics['f0_mean'],
            f0_std=metrics['f0_std'],
            quality_score=metrics['quality_score'],
            feedback=feedback
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)