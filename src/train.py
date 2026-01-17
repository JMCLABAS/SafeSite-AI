from ultralytics import YOLO
import os

#.\venv\Scripts\activate

def main():
    # Rutas dinámicas 
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(base_dir, 'data', 'data.yaml')
    models_dir = os.path.join(base_dir, 'models')
    
    # Verificación
    if not os.path.exists(yaml_path):
        print(f"❌ Error crítico: No se encuentra {yaml_path}")
        return

    print("🚀 SafeSite AI: Iniciando entrenamiento de arquitectura YOLOv11...")
    
    # Cargar modelo base (Medium para balance peso/potencia)
    model = YOLO('yolo11m.pt') 

    # Entrenamiento
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,           # Ajustado para mi GPU 3050
        device=0,
        patience=15,       # Early stopping si no mejora
        optimizer='AdamW',
        lr0=0.001,
        project=models_dir,      # Guardar ordenado en /models
        name='SafeSite-AI_v2', # Nombre 
        exist_ok=True,
        verbose=True
    )
    
    print(f"✅ Entrenamiento completado. Modelo guardado en: {models_dir}")

if __name__ == '__main__':
    main()