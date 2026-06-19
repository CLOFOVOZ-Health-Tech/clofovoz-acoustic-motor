import matplotlib
matplotlib.use('Agg') # Importante para servidores sin interfaz gráfica
import matplotlib.pyplot as plt
import io
import base64

# ... (dentro de la función analyze_voice) ...

    # 1. Generar Espectrograma
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma')
    
    # Guardar en memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close() # Limpiar memoria

    # ... (en el return) ...
    return AnalysisResponse(
        # ... otros campos ...
        spectrogram_url=f"data:image/png;base64,{image_base64}", # <--- Nuevo campo
    )
