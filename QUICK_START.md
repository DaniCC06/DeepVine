# 🚀 DeepVine RTX 4060Ti - Quick Start

## ✅ Auditoría Completada

Se encontraron y **corrigieron 8 problemas críticos**:
- ✓ Batch size optimizado (16 → 8)
- ✓ Pipeline YOLO integrado
- ✓ Memoria optimizada (workers: 4 → 2)
- ✓ Documentación completa agregada

---

## 📋 Cómo Hacer el Entrenamiento (RTX 4060Ti)

### Paso 1: Preparar Entorno
```bash
# Crear ambiente virtual
python -m venv deepvine_env

# Activar (Windows)
deepvine_env\Scripts\activate

# Activar (Linux/Mac)
source deepvine_env/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 2: Validar Configuración
```bash
# Verificar GPU, dataset, dependencias
python verify_setup.py
```

Deberías ver:
```
✓ PASS: GPU & CUDA
✓ PASS: PyTorch Dependencies
✓ PASS: Classification Dataset
✓ PASS: Detection Dataset (Optional)
✓ PASS: Checkpoint Directories

✅ All checks passed! Ready to train.
```

### Paso 3: Ejecutar Entrenamiento

#### Opción A: Solo Clasificador (10 minutos, prueba rápida)
```bash
python run_train.py
```

#### Opción B: YOLO + Clasificador RECOMENDADO ⭐
```bash
python train_integrated.py --data-yaml annotations/data.yaml
```

Esto entrena:
1. **YOLO Detector** (~90 min) - Detecta hojas en imágenes
2. **Clasificador** (~90 min) - Clasifica variedad de uva

**Total:** ~3 horas en RTX 4060Ti

#### Opción C: Solo YOLO (si necesitas detector)
```bash
python train_yolo.py --data annotations/data.yaml --batch 16
```

### Paso 4: Monitorear GPU (en otra terminal)
```bash
nvidia-smi -l 1
```

Deberías ver:
```
GPU Memory: 4-6 GB en uso
Temperature: 50-75°C
Utilization: 80-95%
```

---

## 📂 Estructura de Dataset Requerida

### Para Clasificación (OBLIGATORIO)
```
dataset/
├── albillo_mayor/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── ...
├── garnacha/
│   ├── foto1.jpg
│   └── ...
├── mencia/
│   └── ...
└── [más variedades...]
```

**Mínimo:** 20-30 imágenes por variedad

### Para Detección (OPCIONAL)
Si tienes anotaciones en formato YOLO/COCO:
```
annotations/
├── data.yaml          # Configuración
├── images/
│   ├── train/*.jpg
│   └── val/*.jpg
└── labels/
    ├── train/*.txt    # Anotaciones COCO
    └── val/*.txt
```

---

## 🎯 Comandos Útiles

| Comando | Qué hace |
|---------|----------|
| `python run_train.py` | Entrena solo clasificador (rápido) |
| `python train_integrated.py --data-yaml data.yaml` | Entrena YOLO + Clasificador |
| `python verify_setup.py` | Verifica GPU y dataset |
| `nvidia-smi -l 1` | Monitorea GPU en tiempo real |

---

## ⚙️ Parámetros RTX 4060Ti (OPTIMIZADOS)

```python
# Automático en train.py y train_integrated.py
batch_size = 8          # Optimizado para 8GB VRAM
num_workers = 2         # Reduce overhead de memoria
epochs = 30             # Suficiente para convergencia
mixed_precision = True  # Activado para CUDA
```

### Si tienes problemas:

**Error: OOM (Out of Memory)**
```bash
python train_integrated.py --batch-size 4  # Reduce batch
```

**Entrenamiento lento**
```bash
python train_integrated.py --batch-size 10 --workers 1
```

**GPU no se usa**
```bash
python verify_setup.py  # Verifica CUDA
```

---

## 📊 Resultados Esperados

### Tiempo
- Clasificador: 2 horas
- YOLO: 1.5 horas  
- Total: 3-3.5 horas

### Métricas
- **Clasificador:** Macro F1 = 0.75-0.85
- **YOLO:** mAP50 = 0.70-0.85

### Memoria
- **Pico:** 5-6 GB de 8GB disponibles
- **Seguro:** No hay riesgo de crash

---

## 📁 Archivos Generados

Después del entrenamiento:
```
checkpoints/
├── best_model.pt              # Mejor clasificador
└── yolo_leaf_detector.pt      # Mejor detector
```

**Usar en producción:**
```python
from app import InferenceService, InferenceConfig

config = InferenceConfig(
    checkpoint_path="./checkpoints/best_model.pt",
    detector_path="./checkpoints/yolo_leaf_detector.pt",
)
service = InferenceService(config)
```

---

## 📖 Documentación Completa

- **`TRAINING_GUIDE.md`** - Guía detallada (todas las opciones)
- **`AUDIT_REPORT.md`** - Reporte técnico completo
- **`verify_setup.py`** - Verificación de configuración
- **`train_integrated.py`** - Pipeline principal

---

## ✅ CHECKLIST Final

Antes de entrenar:
- [ ] GPU detectada (`nvidia-smi`)
- [ ] Dataset en `./dataset` con al menos 20 imgs/variedad
- [ ] `python verify_setup.py` devuelve ✓ (o con detection opcional)
- [ ] 3-4 horas disponibles
- [ ] Otra terminal con `nvidia-smi -l 1` para monitoreo

---

## 🚀 COMANDO FINAL

```bash
# Entrenamiento completo (YOLO + Clasificador) - RECOMENDADO
python train_integrated.py --data-yaml annotations/data.yaml

# O solo clasificador (más rápido para pruebas)
python run_train.py
```

**¡Eso es! El sistema se encargará del resto.**

Modelos entrenados se guardan en `checkpoints/` automáticamente.

---

**Preguntas?** Ver `TRAINING_GUIDE.md` para detalles completos.
