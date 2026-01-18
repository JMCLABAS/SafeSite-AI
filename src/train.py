"""
Módulo de Entrenamiento de Modelos (Pipeline de Aprendizaje).

Gestiona el ciclo de vida del entrenamiento mediante Transfer Learning sobre YOLOv11.
Configura hiperparámetros, gestión de artefactos y estrategias de optimización
(Early Stopping, optimizadores adaptativos) para la generación del modelo de producción.
"""

from ultralytics import YOLO
import os

def main():
    # Configuración agnóstica del entorno (CI/CD friendly)
    # Garantiza que las rutas sean relativas a la ejecución del script para portabilidad entre máquinas.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(base_dir, 'data', 'data.yaml')
    models_dir = os.path.join(base_dir, 'models')
    
    # Validación de integridad de datos antes de iniciar el consumo de recursos GPU
    if not os.path.exists(yaml_path):
        print(f"Error crítico: No se encuentra {yaml_path}")
        return

    print("SafeSite AI: Iniciando entrenamiento de arquitectura YOLOv11...")
    
    # Inicialización de pesos pre-entrenados (Transfer Learning)
    # Se selecciona la variante 'Medium' para optimizar el trade-off entre latencia de inferencia y precisión (mAP).
    model = YOLO('yolo11m.pt') 

    # Ejecución del bucle de entrenamiento supervisado
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,           # Tamaño de lote ajustado para evitar OOM (Out Of Memory) en hardware con VRAM limitada
        device=0,
        patience=15,       # Regularización por Early Stopping: Previene overfitting si la validación se estanca
        optimizer='AdamW', # Optimizador con desacople de decaimiento de peso para convergencia robusta
        lr0=0.001,
        project=models_dir,      # Estructura de directorios para versionado de experimentos
        name='SafeSite-AI_v2', # Tag de la versión del modelo
        exist_ok=True,
        verbose=True
    )
    
    print(f"Entrenamiento completado. Modelo guardado en: {models_dir}")

if __name__ == '__main__':
    main()