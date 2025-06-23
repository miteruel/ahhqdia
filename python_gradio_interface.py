import gradio as gr
import requests
import cv2
import numpy as np
from PIL import Image
import io
import time
import threading
from datetime import datetime
import os

class ArduinoCameraInterface:
    def __init__(self, arduino_ip="192.168.1.100"):
        """
        Inicializa la interfaz de la cámara Arduino
        
        Args:
            arduino_ip: IP del dispositivo Arduino (ESP32-CAM)
        """
        self.arduino_ip = arduino_ip
        self.capture_url = f"http://{arduino_ip}/capture"
        self.stream_url = f"http://{arduino_ip}/stream"
        self.last_image = None
        self.is_streaming = False
        
        # Crear directorio para guardar imágenes
        if not os.path.exists("captured_images"):
            os.makedirs("captured_images")
    
    def test_connection(self):
        """
        Prueba la conexión con el dispositivo Arduino
        """
        try:
            response = requests.get(self.capture_url, timeout=5)
            return response.status_code == 200, "Conexión exitosa"
        except requests.exceptions.RequestException as e:
            return False, f"Error de conexión: {str(e)}"
    
    def capture_image(self):
        """
        Captura una imagen desde el Arduino
        """
        try:
            response = requests.get(self.capture_url, timeout=10)
            if response.status_code == 200:
                # Convertir bytes a imagen PIL
                image = Image.open(io.BytesIO(response.content))
                self.last_image = image
                
                # Guardar imagen con timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_images/capture_{timestamp}.jpg"
                image.save(filename)
                
                return image, f"Imagen capturada exitosamente. Guardada como: {filename}"
            else:
                return None, f"Error al capturar imagen. Código de estado: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return None, f"Error de red: {str(e)}"
    
    def apply_filter(self, image, filter_type):
        """
        Aplica filtros a la imagen capturada
        """
        if image is None:
            return None, "No hay imagen para procesar"
        
        # Convertir PIL a OpenCV
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if filter_type == "Escala de grises":
            filtered = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
        elif filter_type == "Desenfoque":
            filtered = cv2.GaussianBlur(opencv_image, (15, 15), 0)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        elif filter_type == "Detección de bordes":
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            filtered = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        elif filter_type == "Sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
            filtered = cv2.transform(opencv_image, kernel)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        else:  # Original
            filtered = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        # Convertir de vuelta a PIL
        result_image = Image.fromarray(filtered)
        
        # Guardar imagen filtrada
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_images/filtered_{filter_type.lower().replace(' ', '_')}_{timestamp}.jpg"
        result_image.save(filename)
        
        return result_image, f"Filtro '{filter_type}' aplicado. Guardado como: {filename}"
    
    def get_camera_info(self):
        """
        Obtiene información del estado de la cámara
        """
        connection_status, message = self.test_connection()
        
        info = f"""
        🔗 **Estado de Conexión:** {'✅ Conectado' if connection_status else '❌ Desconectado'}
        📡 **IP del Dispositivo:** {self.arduino_ip}
        📷 **URL de Captura:** {self.capture_url}
        🎥 **URL de Stream:** {self.stream_url}
        📁 **Directorio de Imágenes:** ./captured_images/
        ⏰ **Última Actualización:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        **Mensaje:** {message}
        """
        
        return info
    
    def change_ip(self, new_ip):
        """
        Cambia la IP del dispositivo Arduino
        """
        self.arduino_ip = new_ip
        self.capture_url = f"http://{new_ip}/capture"
        self.stream_url = f"http://{new_ip}/stream"
        
        connection_status, message = self.test_connection()
        
        return f"IP actualizada a: {new_ip}\n{message}"

def create_gradio_interface():
    """
    Crea la interfaz Gradio para el control de la cámara Arduino
    """
    # Inicializar la interfaz de cámara
    camera_interface = ArduinoCameraInterface()
    
    with gr.Blocks(
        title="Control de Cámara Arduino",
        theme=gr.themes.Soft(),
        css="""
        .main-container { max-width: 1200px; margin: 0 auto; }
        .status-box { background: #f0f0f0; padding: 15px; border-radius: 10px; }
        """
    ) as interface:
        
        gr.Markdown(
            """
            # 📷 Control de Cámara Arduino ESP32-CAM
            
            Interfaz para capturar y procesar imágenes desde un dispositivo Arduino ESP32-CAM.
            Asegúrate de que tu dispositivo esté conectado y configurado correctamente.
            """,
            elem_classes=["main-container"]
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Sección de configuración
                gr.Markdown("## ⚙️ Configuración")
                
                ip_input = gr.Textbox(
                    label="IP del Dispositivo Arduino",
                    value="192.168.1.100",
                    placeholder="Ej: 192.168.1.100",
                    info="Ingresa la IP de tu ESP32-CAM"
                )
                
                with gr.Row():
                    update_ip_btn = gr.Button("🔄 Actualizar IP", variant="secondary")
                    test_connection_btn = gr.Button("🔍 Probar Conexión", variant="secondary")
                
                connection_status = gr.Markdown(
                    camera_interface.get_camera_info(),
                    elem_classes=["status-box"]
                )
            
            with gr.Column(scale=3):
                # Sección de captura
                gr.Markdown("## 📸 Captura de Imágenes")
                
                capture_btn = gr.Button("📷 Capturar Imagen", variant="primary", size="lg")
                capture_status = gr.Textbox(
                    label="Estado de Captura",
                    interactive=False,
                    show_label=True
                )
        
        with gr.Row():
            with gr.Column():
                # Imagen original
                gr.Markdown("### 🖼️ Imagen Original")
                original_image = gr.Image(
                    label="Imagen Capturada",
                    type="pil",
                    height=400
                )
            
            with gr.Column():
                # Imagen procesada
                gr.Markdown("### 🎨 Imagen Procesada")
                
                filter_selector = gr.Dropdown(
                    choices=["Original", "Escala de grises", "Desenfoque", "Detección de bordes", "Sepia"],
                    value="Original",
                    label="Seleccionar Filtro"
                )
                
                apply_filter_btn = gr.Button("✨ Aplicar Filtro", variant="secondary")
                
                processed_image = gr.Image(
                    label="Imagen Procesada",
                    type="pil",
                    height=400
                )
                
                filter_status = gr.Textbox(
                    label="Estado del Filtro",
                    interactive=False,
                    show_label=True
                )
        
        # Sección de información
        with gr.Accordion("📋 Información y Ayuda", open=False):
            gr.Markdown(
                """
                ### 🔧 Configuración del Hardware
                
                1. **ESP32-CAM**: Asegúrate de que tu ESP32-CAM esté programado con el código Arduino proporcionado
                2. **WiFi**: Configura las credenciales WiFi en el código Arduino
                3. **IP**: Anota la IP que aparece en el monitor serial del Arduino
                
                ### 📱 Funcionalidades
                
                - **Captura de Imágenes**: Toma fotos instantáneas desde la cámara
                - **Filtros**: Aplica diferentes efectos a las imágenes capturadas
                - **Guardado Automático**: Las imágenes se guardan automáticamente con timestamp
                - **Prueba de Conexión**: Verifica el estado de la conexión con el dispositivo
                
                ### 🔍 Solución de Problemas
                
                - Verifica que el ESP32-CAM esté encendido y conectado a WiFi
                - Asegúrate de que la IP sea correcta
                - Revisa que no haya firewall bloqueando la conexión
                - El dispositivo debe estar en la misma red que tu computadora
                """
            )
        
        # Eventos de la interfaz
        def capture_and_update():
            image, status = camera_interface.capture_image()
            return image, image, status, ""
        
        def update_ip_and_test(new_ip):
            result = camera_interface.change_ip(new_ip)
            info = camera_interface.get_camera_info()
            return result, info
        
        def test_connection_only():
            return camera_interface.get_camera_info()
        
        def apply_filter_and_update(image, filter_type):
            if image is None:
                return None, "No hay imagen para procesar"
            processed, status = camera_interface.apply_filter(image, filter_type)
            return processed, status
        
        # Conectar eventos
        capture_btn.click(
            fn=capture_and_update,
            outputs=[original_image, processed_image, capture_status, filter_status]
        )
        
        update_ip_btn.click(
            fn=update_ip_and_test,
            inputs=[ip_input],
            outputs=[capture_status, connection_status]
        )
        
        test_connection_btn.click(
            fn=test_connection_only,
            outputs=[connection_status]
        )
        
        apply_filter_btn.click(
            fn=apply_filter_and_update,
            inputs=[original_image, filter_selector],
            outputs=[processed_image, filter_status]
        )
    
    return interface

if __name__ == "__main__":
    # Crear y lanzar la interfaz
    interface = create_gradio_interface()
    
    # Configuración de lanzamiento
    interface.launch(
        server_name="0.0.0.0",  # Permite acceso desde otras IPs en la red
        server_port=7860,       # Puerto por defecto de Gradio
        share=False,            # Cambiar a True para crear un enlace público
        debug=True,             # Habilitar modo debug
        show_error=True         # Mostrar errores en la interfaz
    )
