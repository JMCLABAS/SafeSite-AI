# 🛡️ SafeSite AI

**SafeSite AI** es un sistema de visión artificial en tiempo real desarrollado en Python que automatiza la supervisión de seguridad en entornos industriales. Utiliza modelos de Deep Learning de última generación (YOLOv11) para verificar el cumplimiento de Equipos de Protección Personal (EPP), detectando instantáneamente si un operario lleva casco y chaleco reflectante o si está cometiendo una infracción.

Este proyecto demuestra la implementación de un flujo de trabajo profesional en Machine Learning: desde la curación de datasets y estrategias de *Active Learning* para corregir falsos positivos, hasta una arquitectura de software limpia y modular lista para producción.

🚧 **Estado del Proyecto:** En desarrollo activo (WIP). La versión 1.0 (detectora de cascos de seguridad y chalecos reflectantes) está completa y es funcional, en futuras actualizaciones se añadirá la detección de otros elementos de seguridad como gafas de protección, guantes o botas de seguridad.

---

## 📱 Características Principales

* **👁️ Detección en Tiempo Real:** Monitorización continua mediante webcam con baja latencia, utilizando la arquitectura YOLOv11 Medium para un balance óptimo entre velocidad y precisión.
* **🧠 Lógica de Negocio EPP:** Sistema de reglas condicionales que evalúa el equipamiento:
    * 🟢 **SAFE:** Cuadros verdes si se detecta Casco y Chaleco.
    * 🔴 **PELIGRO:** Alertas visuales rojas inmediatas ante la ausencia de protección (Clases `NO-Hardhat`, `NO-Safety Vest`).
* **🎯 Corrección de "Hard Negatives":** Entrenamiento robusto capaz de diferenciar objetos confusos (como gorras de béisbol o gorros de lana) de los cascos de seguridad reglamentarios.
* **🧹 Arquitectura Limpia:** Estructura de código modular (`src/`, `models/`, `data/`) con gestión de rutas dinámicas, alejándose de scripts monolíticos y facilitando la escalabilidad.
* **⚡ Filtrado Inteligente:** Implementación de *Agnostic NMS* para evitar la superposición de detecciones contradictorias sobre el mismo objeto.

---

## 🛠️ Stack Tecnológico

### Core & IA
* **Lenguaje:** Python 3.11.
* **Modelo:** Ultralytics YOLOv11m (Pre-entrenado y Fine-tuned).
* **Visión por Computador:** `OpenCV` para captura de video, pre-procesamiento de frames y renderizado de la interfaz gráfica (UI).

### Datos & Entrenamiento
* **Dataset:** Fusión de "Construction Site Safety v30" (Roboflow) + Hat detection (Roboflow)" para reducción de falsos positivos.
* **Etiquetado:** re-etiquetado de clases conflictivas.
* **Entorno:** Gestión de dependencias mediante `venv` y aceleración por GPU (CUDA) con PyTorch.

---

## 🏗️ Retos Técnicos Superados

### 1. El Problema de la Gorra (Hard Negatives)
El modelo inicial confundía gorras de béisbol con cascos de seguridad debido a la similitud de forma geométrica.
* **Solución:** Implementación de una estrategia de **Active Learning**. Se integró un dataset específico de gorras mapeando sus etiquetas a la clase `NO-Hardhat`, enseñando explícitamente a la red neuronal la diferencia de texturas entre tela y plástico rígido.

### 2. Detecciones Fantasma y Superpuestas
En ocasiones, el modelo detectaba "Cabeza" y "Casco" simultáneamente en el mismo lugar, o parpadeaba entre ambas clases.
* **Solución:** Activación de **Agnostic NMS** (Non-Maximum Suppression agnóstico a la clase) en el pipeline de inferencia. Esto fuerza al modelo a elegir matemáticamente la predicción con mayor confianza, eliminando el ruido y las cajas duplicadas.

### 3. Rutas y Despliegue
Transformación de un entorno de scripts de prueba ("código espagueti") a una estructura de ingeniería de software profesional.
* **Solución:** Desarrollo de scripts universales (`train.py`, `detect.py`) que calculan rutas relativas al sistema operativo (`os.path`), permitiendo que el proyecto funcione en cualquier máquina sin modificar ni una línea de código.

---

## 📸 Galería


---

## 🚀 Cómo ejecutar el proyecto

**1º) Clonar el repositorio:**
```bash
git clone [https://github.com/JMCLABAS/SafeSite-AI.git](https://github.com/JMCLABAS/SafeSite-AI.git)
cd SafeSite-AI
```

**2º) Configuración del Entorno:**

Crear y activar el entorno virtual para aislar las dependencias.

```bash
python -m venv venv
.\venv\Scripts\activate  # En Windows
```
**3º) Instalar dependencias:**

```bash
pip install ultralytics opencv-python labelImg
```

**4º) Ejecutar Inferencia (Webcam):**

El sistema buscará automáticamente el modelo entrenado best.pt en la carpeta models.
```bash
python src/detect.py
```

(Para re-entrenar el modelo con nuevos datos, ejecutar `python src/train.py`)

---

## 📲 Prueba el Sistema

El código está listo para ser desplegado en cualquier PC con webcam. Pulsa `ESC` para cerrar el monitor de seguridad.

---

## 👨‍💻 Autor y Contacto

Desarrollado por **Jose María Clavijo Basáñez.**

Si tienes interés en el código, la arquitectura o quieres colaborar, contáctame en:

* **📧 Email: pclavijobasanez@gmail.com**
* **💼 LinkedIn: www.linkedin.com/in/jose-maría-clavijo-basáñez**

