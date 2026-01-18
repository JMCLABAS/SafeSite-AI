"""
Script de Ingesta y Normalización de Datos (ETL).

Este módulo integra datasets externos al pipeline de entrenamiento principal.
Realiza la reasignación de IDs de clases (Schema Mapping) para alinear
etiquetas heterogéneas con la ontología del proyecto (SafeSite Schema).
"""

import os
import shutil

def main():
    # --- CONFIGURACIÓN DE INGESTA ---
    # Rutas del área de preparación (Staging Area) para datos crudos externos
    source_labels_dir = r"C:\Users\pclav\Desktop\Pepe2\Proyectos propios\SafeSite-AI\temp_caps\train\labels" # Importante: Cambia esto por tu ruta real
    source_images_dir = r"C:\Users\pclav\Desktop\Pepe2\Proyectos propios\SafeSite-AI\temp_caps\train\images" # Importante: Cambia esto por tu ruta real

    # Rutas del Dataset de Producción
    dest_labels_dir = r"data/train/labels" # Importante: Cambia esto por tu ruta real
    dest_images_dir = r"data/train/images" # Importante: Cambia esto por tu ruta real

    # --- NORMALIZACIÓN DE ONTOLOGÍA (Schema Mapping) ---
    # Mapeo de IDs del dataset externo al esquema interno del proyecto.
    # Estrategia: Unificar clases 'cap' (0) y 'nocap' (1) bajo la clase 'NO-Hardhat' (5)
    # para reforzar el aprendizaje de "Hard Negatives".
    class_mapping = {
        '0': '5',  # External: Cap   -> Internal: NO-Hardhat
        '1': '5'   # External: Head  -> Internal: NO-Hardhat
    }
    # ---------------------

    # Inicialización de estructura de directorios si no existe
    os.makedirs(dest_labels_dir, exist_ok=True)
    os.makedirs(dest_images_dir, exist_ok=True)

    # Procesamiento por lotes (Batch processing)
    files = [f for f in os.listdir(source_labels_dir) if f.endswith('.txt')]
    print(f"🔄 Procesando {len(files)} etiquetas e imágenes...")

    count = 0
    for filename in files:
        src_txt_path = os.path.join(source_labels_dir, filename)
        dst_txt_path = os.path.join(dest_labels_dir, filename)
        
        # Resolución dinámica de la extensión de imagen asociada
        img_name_jpg = filename.replace('.txt', '.jpg')
        img_name_png = filename.replace('.txt', '.png')
        
        src_img_path = None
        dst_img_path = None

        # Verificación de integridad referencial (Asset existence check)
        if os.path.exists(os.path.join(source_images_dir, img_name_jpg)):
            src_img_path = os.path.join(source_images_dir, img_name_jpg)
            dst_img_path = os.path.join(dest_images_dir, img_name_jpg)
        elif os.path.exists(os.path.join(source_images_dir, img_name_png)):
            src_img_path = os.path.join(source_images_dir, img_name_png)
            dst_img_path = os.path.join(dest_images_dir, img_name_png)
        
        if src_img_path:
            # 1. TRANSFORMACIÓN (Transform)
            new_lines = []
            with open(src_txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class = parts[0]
                        # Filtrado y reasignación de clase según mapa definido
                        if old_class in class_mapping:
                            parts[0] = class_mapping[old_class]
                            new_lines.append(" ".join(parts) + "\n")
            
            # 2. CARGA (Load)
            # Solo persistimos el registro si contiene clases relevantes tras el filtrado
            if new_lines:
                with open(dst_txt_path, 'w') as f_out:
                    f_out.writelines(new_lines)
                
                # Migración del asset de imagen correspondiente
                shutil.copy2(src_img_path, dst_img_path)
                count += 1
        else:
            print(f"Imagen no encontrada para {filename}")

    print(f"¡Éxito! Se han importado {count} imágenes de gorras convertidas a la clase 5 (NO-Hardhat).")
    print("Ahora ejecuta 'python src/train.py' para reentrenar.")

if __name__ == "__main__":
    main()