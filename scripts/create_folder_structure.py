"""
01_create_folder_structure.py
=============================
Crea la estructura de carpetas compatible con Ultralytics Platform
para el proyecto Wildfire_Mega_Dataset.

Estructura generada:
    /datasets/Wildfire_Mega_Dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── raw_sources/          ← descargas sin procesar por dataset
    │   ├── 01_AI_ForMankind/
    │   ├── 02_FASDD_UAV/
    │   ├── 03_FASDD_RS/
    │   ├── 04_FLAME2/
    │   ├── 05_Canungra/
    │   ├── 06_HPWREN_FIgLib/
    │   ├── 07_gengyanlei/
    │   └── 08_BoWFire/
    ├── logs/                 ← logs de descarga y conversión
    └── data.yaml             ← config Ultralytics

Autor: Oscar David Hospinal | Hospinal Systems
Fecha: 2026-04-03
Proyecto: Ultralytics Platform — Wildfire Early Detection
"""

import os
import yaml
from pathlib import Path


# ─── Configuración ────────────────────────────────────────────────────────────

# Cambia BASE_DIR según el entorno:
#   Google Colab + Drive:  "/content/drive/MyDrive/ultralytics_wildfire"
#   Local Linux/WSL:       "/mnt/c/Users/David/Documents/HospinalSystems/03-26-Ultralytics/datasets"
#   Ultralytics Platform:  "/datasets"

BASE_DIR = os.environ.get("WILDFIRE_BASE_DIR", "/content/drive/MyDrive/ultralytics_wildfire")
DATASET_NAME = "Wildfire_Mega_Dataset"
DATASET_ROOT = Path(BASE_DIR) / "datasets" / DATASET_NAME

# Clases unificadas (YOLO index → nombre)
CLASSES = {
    0: "smoke",  # humo en cualquier densidad (transparente, denso, pluma)
    1: "fire",   # llamas visibles
}

# Split ratio
SPLIT_RATIO = {"train": 0.80, "val": 0.10, "test": 0.10}

# Sub-datasets que se descargarán
RAW_SOURCES = [
    "01_AI_ForMankind",
    "02_FASDD_UAV",
    "03_FASDD_RS",
    "04_FLAME2",
    "05_Canungra",
    "06_HPWREN_FIgLib",
    "07_gengyanlei",
    "08_BoWFire",
]


# ─── Creación de estructura ────────────────────────────────────────────────────

def create_ultralytics_structure(root: Path) -> dict:
    """
    Crea la estructura de carpetas estándar de Ultralytics y devuelve
    un dict con todas las rutas creadas.
    """
    dirs = {}

    # Directorios principales de imágenes y etiquetas
    for split in ("train", "val", "test"):
        for modality in ("images", "labels"):
            p = root / modality / split
            p.mkdir(parents=True, exist_ok=True)
            dirs[f"{modality}_{split}"] = p

    # Directorio de fuentes sin procesar
    raw_root = root / "raw_sources"
    for src in RAW_SOURCES:
        p = raw_root / src
        p.mkdir(parents=True, exist_ok=True)
        dirs[f"raw_{src}"] = p

    # Directorio de logs
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    dirs["logs"] = logs_dir

    return dirs


def create_data_yaml(root: Path) -> Path:
    """
    Genera el archivo data.yaml compatible con Ultralytics Platform.
    """
    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": list(CLASSES.values()),
        # Metadata adicional
        "project": "Wildfire Early Detection — Ultralytics x Hospinal Systems",
        "description": (
            "Mega-dataset unificado de detección de humo y fuego en incendios forestales. "
            "Combina FASDD-UAV/RS, AI ForMankind, FLAME 2, Canungra UAS, HPWREN FIgLib, "
            "gengyanlei y BoWFire. Optimizado para YOLO11 → YOLO26 benchmark."
        ),
        "version": "1.0.0",
        "split_ratio": SPLIT_RATIO,
        "class_mapping": {
            "smoke": "Clase 0 — humo de cualquier densidad (transparente, pluma, difuso)",
            "fire": "Clase 1 — llamas visibles detectadas",
        },
        "sources": [
            {
                "id": "01_AI_ForMankind",
                "name": "AI for Mankind Wildfire Smoke Dataset",
                "url": "https://github.com/aiformankind/wildfire-smoke-dataset",
                "license": "MIT",
                "images": 737,
                "classes": ["smoke"],
            },
            {
                "id": "02_FASDD_UAV",
                "name": "FASDD UAV Sub-dataset",
                "url": "https://doi.org/10.57760/sciencedb.j00104.00103",
                "license": "CC BY 4.0",
                "images": "~40000",
                "classes": ["fire", "smoke"],
                "note": "Etiquetas YOLO nativas incluidas",
            },
            {
                "id": "03_FASDD_RS",
                "name": "FASDD Remote Sensing Sub-dataset",
                "url": "https://doi.org/10.57760/sciencedb.j00104.00103",
                "license": "CC BY 4.0",
                "images": "~40000",
                "classes": ["fire", "smoke"],
                "note": "Imágenes satelitales Sentinel-2 ESA",
            },
            {
                "id": "04_FLAME2",
                "name": "FLAME 2 — Aerial RGB+IR Dataset",
                "url": "https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset",
                "license": "IEEE DataPort Open Access",
                "images": 53451,
                "classes": ["fire", "smoke"],
                "note": "Pares RGB + Infrarrojo Térmico — quema controlada Arizona",
            },
            {
                "id": "05_Canungra",
                "name": "Canungra UAS-data on Control Fire (QUT)",
                "url": "https://doi.org/10.25912/RDF_1764134706710",
                "license": "QUT Research Data",
                "images": "TBD",
                "classes": ["fire", "smoke", "plume"],
                "note": "Plume detection desde base — quema controlada Queensland",
            },
            {
                "id": "06_HPWREN_FIgLib",
                "name": "HPWREN Fire Ignition Images Library",
                "url": "https://www.hpwren.ucsd.edu/FIgLib/",
                "license": "UCSD Research",
                "images": "~5000",
                "classes": ["smoke"],
                "note": "Cámaras fijas PTZ — perspectiva terrestre — chaparral",
            },
            {
                "id": "07_gengyanlei",
                "name": "Fire & Smoke Detect YOLOv4 Dataset",
                "url": "https://github.com/gengyanlei/fire-smoke-detect-yolov4",
                "license": "Open Source",
                "images": "~4000",
                "classes": ["fire", "smoke"],
            },
            {
                "id": "08_BoWFire",
                "name": "BoWFire Dataset (GBDI/UNICAMP)",
                "url": "https://bitbucket.org/gbdi/bowfire-dataset/downloads/",
                "license": "Academic Research",
                "images": 240,
                "classes": ["fire", "smoke"],
                "note": "Hard negatives: atardeceres, luz artificial, brasas",
            },
        ],
    }

    yaml_path = root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    return yaml_path


def create_label_mapping_guide(root: Path) -> Path:
    """
    Crea un archivo de referencia para conversión de etiquetas por dataset.
    """
    content = """# Label Mapping Guide — Wildfire_Mega_Dataset
# ===============================================
# Clase 0: smoke — humo de cualquier densidad
# Clase 1: fire  — llamas visibles
#
# FORMATO YOLO (por línea en .txt):
# <class_id> <x_center> <y_center> <width> <height>
# Todos los valores normalizados [0.0, 1.0]

# ─── Dataset 01: AI for Mankind ───────────────
# Etiquetas originales: smoke
# Conversión: smoke → 0
# Acción: Renombrar class_id si es diferente de 0

# ─── Dataset 02-03: FASDD UAV / RS ────────────
# Etiquetas YOLO incluidas — verificar orden de clases en su data.yaml:
#   Si: names: [fire, smoke] → fire=0, smoke=1 ← REQUIERE SWAP
#   Si: names: [smoke, fire] → smoke=0, fire=1 ← OK directo
# Acción: ejecutar script 03_label_converter.py

# ─── Dataset 04: FLAME 2 ──────────────────────
# Etiquetas: formato propio o PASCAL VOC XML
# Conversión: usar script de conversión VOC→YOLO incluido
# fire → 1 | smoke → 0

# ─── Dataset 05: Canungra UAS ─────────────────
# Etiquetas: verificar formato en el repositorio QUT
# plume → tratar como smoke → 0
# fire → 1

# ─── Dataset 06: HPWREN FIgLib ───────────────
# Etiquetas: archivos de texto con coordenadas propias
# Conversión manual requerida → smoke → 0
# Ver: script 03_label_converter.py función hpwren_to_yolo()

# ─── Dataset 07: gengyanlei ──────────────────
# Etiquetas YOLO incluidas — verificar nomenclatura de clases
# fire → 1 | smoke → 0

# ─── Dataset 08: BoWFire ─────────────────────
# Etiquetas: clasificación por carpetas (fire/ notfire/ smoke/)
# Convertir a bbox YOLO usando imagen completa como bbox:
#   x_center=0.5, y_center=0.5, width=1.0, height=1.0
# fire → 1 | smoke → 0 | notfire → IGNORAR (hard negatives opcionales)
"""
    guide_path = root / "logs" / "label_mapping_guide.txt"
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(content)
    return guide_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Ultralytics Wildfire Mega Dataset — Setup")
    print(f"{'='*60}")
    print(f"\n  Root: {DATASET_ROOT}\n")

    # 1. Crear estructura de carpetas
    print("[1/3] Creando estructura de carpetas...")
    dirs = create_ultralytics_structure(DATASET_ROOT)
    for name, path in sorted(dirs.items()):
        status = "✓" if path.exists() else "✗"
        print(f"      {status}  {path.relative_to(DATASET_ROOT)}")

    # 2. Crear data.yaml
    print("\n[2/3] Generando data.yaml...")
    yaml_path = create_data_yaml(DATASET_ROOT)
    print(f"      ✓  {yaml_path.name}")

    # 3. Crear guía de mapeo
    print("\n[3/3] Generando label_mapping_guide.txt...")
    guide_path = create_label_mapping_guide(DATASET_ROOT)
    print(f"      ✓  {guide_path.relative_to(DATASET_ROOT)}")

    print(f"\n{'='*60}")
    print(f"  ✅ Estructura creada exitosamente")
    print(f"  Próximo paso: ejecutar notebooks/wildfire_dataset_pipeline.ipynb")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
