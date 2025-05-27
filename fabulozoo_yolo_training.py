#!/usr/bin/env python3
"""
FabuloZoo - YOLO Training Environment
Sistema de entrenamiento para detección de especies en zoológico
Especies objetivo: León, Mono, Elefante, Jirafa, Humanos
"""

import os
import yaml
import requests
import zipfile
from pathlib import Path
import shutil
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fabulozoo_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FabuloZooYOLOTrainer:
    """
    Clase principal para configurar y entrenar modelo YOLO 
    para detección de especies en FabuloZoo
    """
    
    def __init__(self, project_name: str = "FabuloZoo"):
        self.project_name = project_name
        self.base_dir = Path(project_name)
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        
        # Especies objetivo con IDs
        self.species_config = {
            0: "leon",
            1: "mono", 
            2: "elefante",
            3: "jirafa",
            4: "humano"
        }
        
        # URLs de datasets públicos (ejemplos)
        self.dataset_sources = {
            "coco_animals": "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
            "open_images": "https://storage.googleapis.com/openimages/web/index.html"
        }
        
    def setup_project_structure(self):
        """Crea la estructura de directorios del proyecto"""
        logger.info("🏗️  Creando estructura de proyecto...")
        
        directories = [
            self.base_dir,
            self.data_dir,
            self.data_dir / "images" / "train",
            self.data_dir / "images" / "val", 
            self.data_dir / "images" / "test",
            self.data_dir / "labels" / "train",
            self.data_dir / "labels" / "val",
            self.data_dir / "labels" / "test",
            self.data_dir / "raw",
            self.models_dir,
            self.results_dir,
            self.base_dir / "scripts",
            self.base_dir / "configs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Creado: {directory}")
    
    def create_dataset_yaml(self):
        """Crea el archivo YAML de configuración del dataset"""
        logger.info("📋 Creando configuración de dataset...")
        
        dataset_config = {
            'path': str(self.data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': len(self.species_config),  # número de clases
            'names': list(self.species_config.values())
        }
        
        yaml_path = self.base_dir / "configs" / "fabulozoo_dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"✅ Dataset config guardado: {yaml_path}")
        return yaml_path
    
    def create_training_script(self):
        """Genera script de entrenamiento personalizado"""
        logger.info("🎯 Creando script de entrenamiento...")
        
        training_script = '''#!/usr/bin/env python3
"""
Script de entrenamiento FabuloZoo YOLO
"""
from ultralytics import YOLO
import torch
import os
from pathlib import Path

def train_fabulozoo_model():
    """Entrena el modelo YOLO para FabuloZoo"""
    
    print("🚀 Iniciando entrenamiento FabuloZoo YOLO...")
    
    # Verificar GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Usando dispositivo: {device}")
    
    # Cargar modelo pre-entrenado
    model = YOLO('yolov8n.pt')  # nano version para prototipo rápido
    # Para mejor precisión usar: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    
    # Configuración de entrenamiento
    results = model.train(
        data='configs/fabulozoo_dataset.yaml',
        epochs=100,           # Ajustar según necesidades
        imgsz=640,           # Tamaño de imagen
        batch=16,            # Ajustar según GPU disponible
        device=device,
        project='results',
        name='fabulozoo_v1',
        save=True,
        plots=True,
        verbose=True,
        patience=10,         # Early stopping
        save_period=10,      # Guardar cada 10 epochs
        workers=4,           # Número de workers para DataLoader
        cos_lr=True,         # Cosine learning rate scheduler
        mixup=0.1,           # Mixup augmentation
        copy_paste=0.1,      # Copy-paste augmentation
        mosaic=1.0,          # Mosaic augmentation
        degrees=10,          # Rotación aleatoria
        translate=0.1,       # Translación aleatoria
        scale=0.9,           # Escalado aleatorio
        fliplr=0.5,          # Flip horizontal
        flipud=0.1,          # Flip vertical
        hsv_h=0.015,         # Augmentación HSV hue
        hsv_s=0.7,           # Augmentación HSV saturación
        hsv_v=0.4            # Augmentación HSV valor
    )
    
    print("✅ Entrenamiento completado!")
    print(f"📊 Resultados guardados en: {results.save_dir}")
    
    # Validar modelo
    print("🔍 Validando modelo...")
    validation_results = model.val()
    
    # Exportar modelo para producción
    print("📦 Exportando modelo...")
    model.export(format='onnx')  # Para optimización
    
    return results

if __name__ == "__main__":
    train_fabulozoo_model()
'''
        
        script_path = self.base_dir / "scripts" / "train_model.py"
        with open(script_path, 'w') as f:
            f.write(training_script)
        
        # Hacer ejecutable
        os.chmod(script_path, 0o755)
        logger.info(f"✅ Script de entrenamiento: {script_path}")
        
    def create_data_collection_guide(self):
        """Crea guía para recolección de datos"""
        logger.info("📖 Creando guía de recolección de datos...")
        
        guide = '''# Guía de Recolección de Datos - FabuloZoo

## 📸 Especificaciones de Imágenes

### Requisitos Técnicos:
- **Formato**: JPG, PNG
- **Resolución mínima**: 640x640 píxeles
- **Tamaño máximo**: 5MB por imagen
- **Calidad**: Alta resolución, bien iluminadas

### Distribución por Especie:
- **León**: 500-1000 imágenes
- **Mono**: 500-1000 imágenes  
- **Elefante**: 500-1000 imágenes
- **Jirafa**: 500-1000 imágenes
- **Humanos**: 1000-2000 imágenes

### División del Dataset:
- **Entrenamiento (train)**: 70%
- **Validación (val)**: 20% 
- **Prueba (test)**: 10%

## 🎯 Criterios de Calidad

### Para cada especie:
1. **Variedad de ángulos**: frontal, lateral, posterior
2. **Diferentes posturas**: parado, acostado, en movimiento
3. **Variedad de iluminación**: día, sombra, atardecer
4. **Diferentes distancias**: primer plano, plano medio, plano general
5. **Contextos variados**: diferentes fondos del zoológico

### Evitar:
- Imágenes borrosas o desenfocadas
- Sobreexposición o subexposición extrema
- Animales apenas visibles
- Imágenes duplicadas o muy similares

## 🏷️ Etiquetado (Annotations)

### Formato YOLO:
```
class_id center_x center_y width height
```

### Herramientas Recomendadas:
1. **LabelImg**: https://github.com/tzutalin/labelImg
2. **CVAT**: https://cvat.org/
3. **Roboflow**: https://roboflow.com/

### Ejemplo de etiqueta:
```
0 0.5 0.3 0.4 0.6  # León en el centro-superior de la imagen
```

## 📁 Estructura de Archivos:
```
data/
├── images/
│   ├── train/
│   │   ├── leon_001.jpg
│   │   ├── mono_001.jpg
│   │   └── ...
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    │   ├── leon_001.txt
    │   ├── mono_001.txt
    │   └── ...
    ├── val/
    └── test/
```

## 🌐 Fuentes de Datos Recomendadas:

### Datasets Públicos:
1. **Open Images Dataset**: https://storage.googleapis.com/openimages/web/index.html
2. **COCO Dataset**: https://cocodataset.org/
3. **iNaturalist**: https://www.inaturalist.org/
4. **Flickr Animal Dataset**: https://www.flicr.com/

### Datos Propios:
- Fotografías del zoológico objetivo
- Videos convertidos a frames
- Cámaras de seguridad existentes

## ⚡ Automatización

### Script de descarga automática:
```bash
python scripts/download_data.py --species leon --count 1000
```

### Validación automática:
```bash
python scripts/validate_dataset.py
```
'''
        
        guide_path = self.base_dir / "DATA_COLLECTION_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide)
        logger.info(f"✅ Guía de datos: {guide_path}")
    
    def create_validation_script(self):
        """Crea script para validar el dataset"""
        logger.info("🔍 Creando script de validación...")
        
        validation_script = '''#!/usr/bin/env python3
"""
Script de validación del dataset FabuloZoo
"""
import os
from pathlib import Path
import yaml
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def validate_fabulozoo_dataset():
    """Valida la estructura y calidad del dataset"""
    
    print("🔍 Validando dataset FabuloZoo...")
    
    # Cargar configuración
    with open('configs/fabulozoo_dataset.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = Path(config['path'])
    species_names = config['names']
    
    # Validar estructura de directorios
    required_dirs = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    
    print("📁 Validando estructura de directorios...")
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            print(f"❌ Falta directorio: {dir_path}")
        else:
            print(f"✅ Directorio existe: {dir_path}")
    
    # Contar imágenes y etiquetas
    splits = ['train', 'val', 'test']
    stats = {}
    
    for split in splits:
        img_dir = data_path / 'images' / split
        label_dir = data_path / 'labels' / split
        
        # Contar archivos
        images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        labels = list(label_dir.glob('*.txt'))
        
        stats[split] = {
            'images': len(images),
            'labels': len(labels),
            'match': len(images) == len(labels)
        }
        
        print(f"📊 {split.upper()}:")
        print(f"   Imágenes: {len(images)}")
        print(f"   Etiquetas: {len(labels)}")
        print(f"   Coinciden: {'✅' if stats[split]['match'] else '❌'}")
    
    # Análisis de distribución de clases
    print("\\n📈 Analizando distribución de clases...")
    class_counts = Counter()
    
    for split in splits:
        label_dir = data_path / 'labels' / split
        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[species_names[class_id]] += 1
    
    # Mostrar estadísticas
    print("\\n🎯 Distribución por especie:")
    for species, count in class_counts.items():
        print(f"   {species}: {count} instancias")
    
    # Crear gráfico de distribución
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title('Distribución de Especies en Dataset FabuloZoo')
    plt.xlabel('Especies')
    plt.ylabel('Número de Instancias')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/dataset_distribution.png')
    print("\\n📊 Gráfico guardado: results/dataset_distribution.png")
    
    # Recomendaciones
    print("\\n💡 Recomendaciones:")
    total_samples = sum(class_counts.values())
    if total_samples < 1000:
        print("⚠️  Dataset pequeño. Recomendado: >1000 muestras por clase")
    
    min_samples = min(class_counts.values()) if class_counts else 0
    max_samples = max(class_counts.values()) if class_counts else 0
    
    if max_samples > min_samples * 3:
        print("⚠️  Dataset desbalanceado. Considerar balanceo de clases")
    
    print("\\n✅ Validación completada")
    return stats

if __name__ == "__main__":
    validate_fabulozoo_dataset()
'''
        
        script_path = self.base_dir / "scripts" / "validate_dataset.py"
        with open(script_path, 'w') as f:
            f.write(validation_script)
        
        os.chmod(script_path, 0o755)
        logger.info(f"✅ Script de validación: {script_path}")
    
    def create_requirements_file(self):
        """Crea archivo de dependencias"""
        logger.info("📦 Creando archivo de dependencias...")
        
        requirements = '''# FabuloZoo YOLO Training Requirements

# Core ML/CV libraries
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pillow>=10.0.0

# Data handling
numpy>=1.24.0
pandas>=2.0.0
PyYAML>=6.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.0.0

# Utilities
tqdm>=4.65.0
requests>=2.31.0
pathlib

# Optional: For advanced augmentations
albumentations>=1.3.0

# Optional: For dataset management
roboflow>=1.0.0

# Development tools
jupyter>=1.0.0
ipykernel>=6.0.0

# For production deployment
onnx>=1.14.0
onnxruntime>=1.15.0
'''
        
        req_path = self.base_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write(requirements)
        logger.info(f"✅ Requirements: {req_path}")
    
    def create_installation_script(self):
        """Crea script de instalación automática"""
        logger.info("⚙️ Creando script de instalación...")
        
        install_script = '''#!/bin/bash
# FabuloZoo Installation Script

echo "🦁 Instalando entorno FabuloZoo YOLO..."

# Crear entorno virtual
echo "🐍 Creando entorno virtual..."
python -m venv fabulozoo_env

# Activar entorno
echo "✅ Activando entorno..."
source fabulozoo_env/bin/activate  # Linux/Mac
# fabulozoo_env\\Scripts\\activate  # Windows

# Actualizar pip
echo "📦 Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

# Descargar modelo pre-entrenado
echo "🎯 Descargando modelo YOLO pre-entrenado..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "🎉 Instalación completada!"
echo "🚀 Para entrenar: python scripts/train_model.py"
echo "🔍 Para validar: python scripts/validate_dataset.py"
'''
        
        script_path = self.base_dir / "install.sh"
        with open(script_path, 'w') as f:
            f.write(install_script)
        
        os.chmod(script_path, 0o755)
        logger.info(f"✅ Script instalación: {script_path}")
    
    def create_readme(self):
        """Crea README del proyecto"""
        logger.info("📖 Creando README...")
        
        readme = f'''# 🦁 FabuloZoo - YOLO Training Environment

Sistema de entrenamiento para detección de especies en zoológico usando Ultralytics YOLO.

## 🎯 Especies Objetivo
- 🦁 **León** (Panthera leo)
- 🐵 **Mono** (Primates)  
- 🐘 **Elefante** (Elephantidae)
- 🦒 **Jirafa** (Giraffa)
- 👤 **Humanos** (Homo sapiens)

## 🚀 Inicio Rápido

### 1. Instalación
```bash
# Clonar y configurar
git clone <repository>
cd {self.project_name}
./install.sh
```

### 2. Preparar Datos
1. Seguir la guía: `DATA_COLLECTION_GUIDE.md`
2. Colocar imágenes en: `data/images/train|val|test/`
3. Colocar etiquetas en: `data/labels/train|val|test/`

### 3. Validar Dataset
```bash
python scripts/validate_dataset.py
```

### 4. Entrenar Modelo
```bash
python scripts/train_model.py
```

## 📁 Estructura del Proyecto
```
{self.project_name}/
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
├── results/
├── scripts/
│   ├── train_model.py
│   └── validate_dataset.py
├── configs/
│   └── fabulozoo_dataset.yaml
├── requirements.txt
└── README.md
```

## ⚙️ Configuración

### Dataset YAML
```yaml
path: ./data
train: images/train
val: images/val
test: images/test
nc: 5
names: [leon, mono, elefante, jirafa, humano]
```

### Parámetros de Entrenamiento
- **Modelo base**: YOLOv8n (nano) para prototipo
- **Épocas**: 100 (ajustable)
- **Batch size**: 16 (ajustar según GPU)
- **Imagen**: 640x640 píxeles
- **Augmentaciones**: Activadas

## 📊 Métricas Objetivo
- **Precisión**: >85%
- **Recall**: >80%
- **mAP@0.5**: >0.8
- **Velocidad**: <50ms por inferencia

## 🛠️ Herramientas Incluidas

### Scripts Principales:
- `train_model.py`: Entrenamiento del modelo
- `validate_dataset.py`: Validación del dataset

### Utilerías:
- Validación automática de estructura
- Análisis de distribución de clases
- Gráficos de métricas
- Exportación a ONNX

## 📈 Monitoreo del Entrenamiento

### TensorBoard (opcional):
```bash
tensorboard --logdir results/
```

### Métricas principales:
- Loss (train/val)
- Precision/Recall
- mAP (mean Average Precision)
- F1-Score

## 🔧 Troubleshooting

### Problemas Comunes:
1. **GPU no detectada**: Verificar instalación CUDA
2. **Memoria insuficiente**: Reducir batch_size
3. **Dataset pequeño**: Aumentar augmentaciones
4. **Sobreentrenamiento**: Usar early stopping

### Logs:
- Entrenamiento: `fabulozoo_training.log`
- Resultados: `results/fabulozoo_v1/`

## 📝 Notas

- Modelo inicial optimizado para velocidad (YOLOv8n)
- Para mayor precisión usar YOLOv8s/m/l/x
- Dataset mínimo: 500 imágenes por clase
- Recomendado: 1000+ imágenes por clase

## 🚀 Próximos Pasos

1. **Recolección de datos** específicos del zoológico
2. **Fine-tuning** con datos reales
3. **Optimización** para hardware específico
4. **Integración** con sistema de alertas
5. **Deploy** en producción

---
Creado para el proyecto FabuloZoo - Sistema Inteligente de Monitoreo de Zoológico
'''
        
        readme_path = self.base_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        logger.info(f"✅ README: {readme_path}")
    
    def setup_complete_environment(self):
        """Configura el entorno completo de entrenamiento"""
        logger.info("🚀 Configurando entorno completo FabuloZoo YOLO...")
        
        try:
            # Crear estructura
            self.setup_project_structure()
            
            # Crear archivos de configuración
            self.create_dataset_yaml()
            self.create_training_script()
            self.create_validation_script()
            self.create_requirements_file()
            self.create_installation_script()
            self.create_data_collection_guide()
            self.create_readme()
            
            logger.info("🎉 ¡Entorno FabuloZoo YOLO configurado exitosamente!")
            logger.info(f"📁 Directorio del proyecto: {self.base_dir.absolute()}")
            
            print(f"""
🦁 FabuloZoo YOLO Training Environment - LISTO!

📁 Proyecto creado en: {self.base_dir.absolute()}

🚀 Próximos pasos:
1. cd {self.project_name}
2. ./install.sh  (instalar dependencias)
3. Seguir DATA_COLLECTION_GUIDE.md para obtener datos
4. python scripts/validate_dataset.py  (validar datos)
5. python scripts/train_model.py  (entrenar modelo)

📊 El proyecto incluye:
✅ Estructura completa de directorios
✅ Scripts de entrenamiento y validación
✅ Configuración YAML del dataset
✅ Guía de recolección de datos
✅ Script de instalación automática
✅ Documentación completa

🎯 Especies configuradas: León, Mono, Elefante, Jirafa, Humanos
            """)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error configurando entorno: {e}")
            return False


def main():
    """Función principal"""
    print("🦁 FabuloZoo - Configuración de Entorno YOLO")
    print("=" * 50)
    
    # Crear instancia del trainer
    trainer = FabuloZooYOLOTrainer("FabuloZoo")
    
    # Configurar entorno completo
    success = trainer.setup_complete_environment()
    
    if success:
        print("\n🎉 ¡Configuración exitosa!")
        print("📖 Lee el README.md para instrucciones detalladas")
    else:
        print("\n❌ Error en la configuración")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())