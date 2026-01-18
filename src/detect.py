"""
Módulo de Detección de EPP en Tiempo Real.

Ejecuta la inferencia utilizando el modelo YOLOv11 personalizado sobre un flujo de video.
Implementa la lógica de negocio para validar el cumplimiento de normas de seguridad
(Cascos/Chalecos) y renderiza alertas visuales en caso de infracción.
"""

import cv2
import os
from ultralytics import YOLO

def main():
    # Resolución de ruta absoluta al modelo para garantizar robustez en el despliegue
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'SafeSite-AI_v2', 'weights', 'best.pt')

    if not os.path.exists(model_path):
        print(f"SISTEMA NO INICIADO: No se encuentra el modelo en: {model_path}")
        print("Ejecuta primero 'python src/train.py'")
        return

    print(f"SafeSite AI: Cargando motor de inferencia ({model_path})...")
    model = YOLO(model_path)

    # Inicialización de captura de video (Webcam 0)
    # Se fuerza resolución 720p para balancear carga de inferencia y FPS
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    print("Monitor activo. Pulsa 'ESC' para cerrar el sistema.")

    while True:
        success, img = cap.read()
        if not success:
            break

        # Pipeline de inferencia
        # Se activa NMS agnóstico (agnostic_nms) para evitar solapamiento de clases
        # conflictivas (ej: detectar 'Cabeza' y 'Casco' simultáneamente).
        results = model(img, stream=True, verbose=False, conf=0.5, agnostic_nms=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls_name = model.names[int(box.cls[0])]

                # --- Lógica de Cumplimiento Normativo (EPP) ---
                draw = False
                color = (200, 200, 200) # Placeholder por defecto
                
                # Sanitización de etiquetas para compatibilidad con fuentes Hershey (OpenCV)
                label = f"{cls_name}" 

                # 1. Evaluación de Protección Craneal
                if cls_name == 'Hardhat':
                    color = (0, 255, 0)   # Cumplimiento
                    label = f"SAFE: CASCO" # Añadir {conf} para ver el nivel de confianza del modelo en su decisión
                    draw = True
                elif cls_name in ['NO-Hardhat', 'No-Helmet', 'Head']:
                    color = (0, 0, 255)   # Infracción
                    label = f"PELIGRO: SIN CASCO" # Añadir {conf} para ver el nivel de confianza del modelo en su decisión
                    draw = True
                
                # 2. Evaluación de Visibilidad (Torso)
                elif cls_name == 'Safety Vest':
                    color = (0, 255, 0)   # Cumplimiento
                    label = f"SAFE: CHALECO" # Añadir {conf} para ver el nivel de confianza del modelo en su decisión
                    draw = True
                elif cls_name == 'NO-Safety Vest':
                    color = (0, 0, 255)   # Infracción
                    label = f"PELIGRO: SIN CHALECO" # Añadir {conf} para ver el nivel de confianza del modelo en su decisión
                    draw = True

                # Renderizado de UI condicional (Filtrado de ruido de fondo)
                if draw:
                    # Bounding Box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    # Fondo de etiqueta para contraste y legibilidad
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
                    # Texto de la etiqueta
                    cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("SafeSite AI - Monitor EPP", img)

        # Rutina de salida (Tecla ESC)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()