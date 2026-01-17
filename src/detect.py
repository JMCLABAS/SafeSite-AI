import cv2
import os
from ultralytics import YOLO

def main():
    # Ruta dinámica al modelo entrenado
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'SafeSite-AI_v2', 'weights', 'best.pt')

    if not os.path.exists(model_path):
        print(f"⚠️ SISTEMA NO INICIADO: No se encuentra el modelo en: {model_path}")
        print("   👉 Ejecuta primero 'python src/train.py'")
        return

    print(f"🛡️ SafeSite AI: Cargando motor de inferencia ({model_path})...")
    model = YOLO(model_path)

    # Configuración de Cámara (16:9)
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    print("🎥 Monitor activo. Pulsa 'ESC' para cerrar el sistema.")

    while True:
        success, img = cap.read()
        if not success:
            break

        # INFERENCIA AVANZADA
        results = model(img, stream=True, verbose=False, conf=0.5, agnostic_nms=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls_name = model.names[int(box.cls[0])]

                # --- REGLAS DE NEGOCIO (EPP) ---
                draw = False
                color = (200, 200, 200) # Gris por defecto
                
                # Usamos texto simple sin emojis para evitar los "???"
                label = f"{cls_name}" 

                # 1. CASCOS (Cabeza)
                if cls_name == 'Hardhat':
                    color = (0, 255, 0)   # VERDE
                    label = f"SAFE: CASCO {conf}"
                    draw = True
                elif cls_name in ['NO-Hardhat', 'No-Helmet', 'Head']:
                    color = (0, 0, 255)   # ROJO
                    label = f"PELIGRO: SIN CASCO {conf}"
                    draw = True
                
                # 2. CHALECOS (Tronco)
                elif cls_name == 'Safety Vest':
                    color = (0, 255, 0)   # VERDE
                    label = f"SAFE: CHALECO {conf}"
                    draw = True
                elif cls_name == 'NO-Safety Vest':
                    color = (0, 0, 255)   # ROJO
                    label = f"PELIGRO: SIN CHALECO {conf}"
                    draw = True

                # Renderizado condicional
                if draw:
                    # Caja
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    # Etiqueta con fondo sólido
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
                    # Texto en blanco
                    cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("SafeSite AI - Monitor EPP", img)

        # Salir con ESC (Código ASCII 27)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()