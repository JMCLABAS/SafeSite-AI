import os
import shutil

def main():
    # --- CONFIGURACIÓN ---
    # 1. Ruta donde descomprimiste los LABELS del dataset nuevo (carpeta temporal)
    # IMPORTANTE: Cambia esto por tu ruta real. Ej: "C:/Users/pclav/Desktop/temp_caps/train/labels"
    source_labels_dir = r"C:\Users\pclav\Desktop\Pepe2\Proyectos propios\SafeSite-AI\temp_caps\train\labels"
    
    # 2. Ruta donde están las IMÁGENES nuevas (carpeta temporal)
    source_images_dir = r"C:\Users\pclav\Desktop\Pepe2\Proyectos propios\SafeSite-AI\temp_caps\train\images"

    # 3. Rutas de destino (Tu proyecto real)
    dest_labels_dir = r"data/train/labels"
    dest_images_dir = r"data/train/images"

    # 4. MAPEO DE CLASES (La Traducción)
    # Dataset Gorras (0=cap, 1=nocap) ---> Tu Proyecto (5=NO-Hardhat)
    # Convertimos tanto gorra como "sin gorra" (cabeza) a NO-Hardhat
    class_mapping = {
        '0': '5',  # cap   -> NO-Hardhat
        '1': '5'   # nocap -> NO-Hardhat
    }
    # ---------------------

    # Asegurarnos de que existen las carpetas de destino
    os.makedirs(dest_labels_dir, exist_ok=True)
    os.makedirs(dest_images_dir, exist_ok=True)

    # Listar archivos .txt
    files = [f for f in os.listdir(source_labels_dir) if f.endswith('.txt')]
    print(f"🔄 Procesando {len(files)} etiquetas e imágenes...")

    count = 0
    for filename in files:
        # Rutas completas
        src_txt_path = os.path.join(source_labels_dir, filename)
        dst_txt_path = os.path.join(dest_labels_dir, filename) # Guardamos ya en data/train/labels
        
        # Nombre de la imagen correspondiente (asumimos .jpg, si hay .png el script avisa)
        img_name_jpg = filename.replace('.txt', '.jpg')
        img_name_png = filename.replace('.txt', '.png')
        
        src_img_path = None
        dst_img_path = None

        # Buscar la imagen (puede ser jpg o png)
        if os.path.exists(os.path.join(source_images_dir, img_name_jpg)):
            src_img_path = os.path.join(source_images_dir, img_name_jpg)
            dst_img_path = os.path.join(dest_images_dir, img_name_jpg)
        elif os.path.exists(os.path.join(source_images_dir, img_name_png)):
            src_img_path = os.path.join(source_images_dir, img_name_png)
            dst_img_path = os.path.join(dest_images_dir, img_name_png)
        
        if src_img_path:
            # 1. TRADUCIR Y COPIAR ETIQUETA
            new_lines = []
            with open(src_txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class = parts[0]
                        # Si la clase está en nuestro mapa (0 o 1), la cambiamos a 5
                        if old_class in class_mapping:
                            parts[0] = class_mapping[old_class]
                            new_lines.append(" ".join(parts) + "\n")
            
            # Solo guardamos si hay líneas válidas
            if new_lines:
                with open(dst_txt_path, 'w') as f_out:
                    f_out.writelines(new_lines)
                
                # 2. COPIAR IMAGEN
                shutil.copy2(src_img_path, dst_img_path)
                count += 1
        else:
            print(f"⚠️ Imagen no encontrada para {filename}")

    print(f"✅ ¡Éxito! Se han importado {count} imágenes de gorras convertidas a la clase 5 (NO-Hardhat).")
    print("🚀 Ahora ejecuta 'python src/train.py' para reentrenar.")

if __name__ == "__main__":
    main()