"""
03_label_converter.py
=====================
Conversor universal de etiquetas para el Wildfire_Mega_Dataset.

Soporta los formatos de todos los datasets incluidos:
  - YOLO .txt (con remapeo de class_ids)
  - PASCAL VOC XML
  - BoWFire (carpetas por clase → bbox completa)
  - HPWREN FIgLib (formato de texto propio)
  - Clasificación binaria → bbox imagen completa

Clases unificadas de salida:
  0: smoke  — humo de cualquier densidad (transparente, denso, plume)
  1: fire   — llamas visibles

Autor: Oscar David Hospinal | Hospinal Systems
Fecha: 2026-04-03
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
import shutil
import json
from tqdm import tqdm


# ─── Configuración global ─────────────────────────────────────────────────────

UNIFIED_CLASSES = {
    "smoke": 0, "Smoke": 0, "SMOKE": 0,
    "fire":  1, "Fire":  1, "FIRE":  1,
    "plume": 0, "Plume": 0, "PLUME": 0,  # plume = columna de humo temprana
    "flame": 1, "Flame": 1, "FLAME": 1,
}

# Mapeos numéricos por dataset (cuando las etiquetas son index, no nombre)
# Formato: {dataset_name: {original_id: new_id}}
NUMERIC_MAPS = {
    "FASDD_UAV": {
        # Verificar en el data.yaml del dataset cuál es el orden
        # Si names: [fire, smoke] → fire=0→1, smoke=1→0
        # Si names: [smoke, fire] → smoke=0→0, fire=1→1  ← OK directo
        0: 0,  # smoke → smoke  (ajustar si el orden es diferente)
        1: 1,  # fire  → fire
    },
    "FASDD_RS": {
        0: 0,
        1: 1,
    },
    "gengyanlei": {
        # fire=0, smoke=1 en el repo original → swap necesario
        0: 1,  # fire  → fire (id 1 en nuestro esquema)
        1: 0,  # smoke → smoke (id 0)
    },
    "AI_ForMankind": {
        0: 0,  # smoke → smoke (único label)
    },
    "FLAME2": {
        0: 0,  # smoke
        1: 1,  # fire
    },
    "Canungra": {
        0: 0,  # smoke/plume
        1: 1,  # fire
    },
    "HPWREN": {
        0: 0,  # smoke (único label)
    },
}


# ─── Conversor YOLO → YOLO (remapeo de IDs) ──────────────────────────────────

def convert_yolo_to_yolo(
    src_label: Path,
    dst_label: Path,
    id_map: dict,
    skip_unmapped: bool = True
) -> int:
    """
    Lee un archivo YOLO .txt y escribe uno nuevo con class_ids remapeados.

    Args:
        src_label:     archivo .txt de entrada
        dst_label:     archivo .txt de salida
        id_map:        {old_class_id: new_class_id}
        skip_unmapped: si True, omite líneas cuyo class_id no está en id_map

    Returns:
        número de anotaciones escritas
    """
    if not src_label.exists():
        return 0

    lines_out = []
    with open(src_label, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                old_id = int(parts[0])
                new_id = id_map.get(old_id)
                if new_id is None and skip_unmapped:
                    continue
                if new_id is None:
                    new_id = old_id
                lines_out.append(f"{new_id} {' '.join(parts[1:])}")
            except ValueError:
                continue

    if lines_out:
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_label, "w") as f:
            f.write("\n".join(lines_out) + "\n")

    return len(lines_out)


# ─── Conversor VOC XML → YOLO ─────────────────────────────────────────────────

def convert_voc_to_yolo(
    xml_path: Path,
    img_w: int,
    img_h: int,
    class_map: Optional[dict] = None,
) -> list[str]:
    """
    Convierte un archivo de anotación PASCAL VOC XML al formato YOLO.

    Args:
        xml_path:   ruta al archivo .xml
        img_w:      ancho de la imagen en píxeles
        img_h:      alto de la imagen en píxeles
        class_map:  {nombre_clase: id_entero} — usa UNIFIED_CLASSES si None

    Returns:
        lista de strings en formato YOLO
    """
    if class_map is None:
        class_map = UNIFIED_CLASSES

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  ⚠️  XML malformado: {xml_path.name} — {e}")
        return []

    lines = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None:
            continue
        name = name_el.text.strip() if name_el.text else ""
        cls_id = class_map.get(name)
        if cls_id is None:
            # Intentar búsqueda case-insensitive
            cls_id = class_map.get(name.lower())
        if cls_id is None:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        try:
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            continue

        # Clamp a límites de imagen
        xmin = max(0.0, min(xmin, img_w))
        ymin = max(0.0, min(ymin, img_h))
        xmax = max(0.0, min(xmax, img_w))
        ymax = max(0.0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            continue

        x_c = ((xmin + xmax) / 2) / img_w
        y_c = ((ymin + ymax) / 2) / img_h
        w   = (xmax - xmin) / img_w
        h   = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    return lines


def batch_convert_voc_to_yolo(
    src_dir: Path,
    dst_dir: Path,
    class_map: Optional[dict] = None,
) -> dict:
    """
    Convierte todos los XML en src_dir a archivos .txt YOLO en dst_dir.
    Requiere que las imágenes estén en el mismo directorio que los XML
    o que las dimensiones sean recuperables de las etiquetas VOC (<size>).
    """
    from PIL import Image

    if class_map is None:
        class_map = UNIFIED_CLASSES

    dst_dir.mkdir(parents=True, exist_ok=True)
    stats = {"converted": 0, "skipped": 0, "errors": 0}

    for xml_path in tqdm(sorted(src_dir.rglob("*.xml")), desc="VOC→YOLO"):
        # Obtener dimensiones desde el XML primero
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size_el = root.find("size")
            if size_el is not None:
                img_w = int(size_el.find("width").text)
                img_h = int(size_el.find("height").text)
            else:
                # Intentar abrir imagen
                for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                    img_path = xml_path.with_suffix(ext)
                    if img_path.exists():
                        with Image.open(img_path) as im:
                            img_w, img_h = im.size
                        break
                else:
                    stats["skipped"] += 1
                    continue
        except Exception:
            stats["errors"] += 1
            continue

        lines = convert_voc_to_yolo(xml_path, img_w, img_h, class_map)
        if lines:
            out_path = dst_dir / (xml_path.stem + ".txt")
            with open(out_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            stats["converted"] += 1
        else:
            stats["skipped"] += 1

    return stats


# ─── Conversor BoWFire (folders → YOLO) ──────────────────────────────────────

def convert_bowfire_to_yolo(
    bowfire_root: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    include_notfire: bool = False,
) -> dict:
    """
    BoWFire organiza las imágenes en carpetas:
      fire/   → class 1
      smoke/  → class 0
      notfire → ignorar (a menos que include_notfire=True para hard negatives)

    La bbox es la imagen completa (x=0.5, y=0.5, w=1.0, h=1.0) ya que
    es un dataset de clasificación convertido a detección.
    """
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    folder_map = {"fire": 1, "smoke": 0}
    if include_notfire:
        pass  # no añadimos clase para notfire — son imágenes sin anotación

    stats = {"fire": 0, "smoke": 0, "notfire_skipped": 0}

    for cls_name, cls_id in folder_map.items():
        cls_dirs = [d for d in bowfire_root.rglob(cls_name) if d.is_dir()]
        for cls_dir in cls_dirs:
            for img_path in cls_dir.glob("*.jpg"):
                stem = f"bow_{cls_name}_{img_path.stem}"
                # Copiar imagen
                shutil.copy2(img_path, dst_images_dir / f"{stem}.jpg")
                # Crear etiqueta con bbox = imagen completa
                with open(dst_labels_dir / f"{stem}.txt", "w") as f:
                    f.write(f"{cls_id} 0.500000 0.500000 1.000000 1.000000\n")
                stats[cls_name] += 1

    # Contar notfire ignorados
    for d in bowfire_root.rglob("notfire"):
        if d.is_dir():
            stats["notfire_skipped"] += len(list(d.glob("*.jpg")))

    return stats


# ─── Conversor HPWREN FIgLib ─────────────────────────────────────────────────

def parse_hpwren_annotation(txt_path: Path, img_w: int, img_h: int) -> list[str]:
    """
    HPWREN FIgLib usa un formato de texto: x1 y1 x2 y2 per line (píxeles absolutos).
    Todas son clase 0 (smoke).
    """
    if not txt_path.exists():
        return []

    lines = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                x_c = ((x1 + x2) / 2) / img_w
                y_c = ((y1 + y2) / 2) / img_h
                w   = abs(x2 - x1) / img_w
                h   = abs(y2 - y1) / img_h
                lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            except ValueError:
                continue
    return lines


# ─── Función de verificación ─────────────────────────────────────────────────

def verify_yolo_label(txt_path: Path) -> tuple[bool, list[str]]:
    """
    Verifica que un archivo .txt sea YOLO válido.
    Retorna (es_válido, lista_de_errores).
    """
    errors = []
    if not txt_path.exists():
        return False, ["archivo no existe"]

    with open(txt_path) as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                errors.append(f"línea {i}: menos de 5 campos ({len(parts)})")
                continue
            try:
                cls_id = int(parts[0])
                if cls_id not in (0, 1):
                    errors.append(f"línea {i}: class_id {cls_id} fuera del rango [0,1]")
                for j, val in enumerate(parts[1:5], 1):
                    v = float(val)
                    if not (0.0 <= v <= 1.0):
                        errors.append(f"línea {i}: campo {j} fuera de [0,1]: {v}")
            except ValueError as e:
                errors.append(f"línea {i}: valor no numérico — {e}")

    return len(errors) == 0, errors


def batch_verify(labels_dir: Path, sample_size: int = 0) -> dict:
    """
    Verifica todos (o una muestra) de los archivos .txt en labels_dir.
    sample_size=0 → verificar todos.
    """
    import random
    all_labels = list(labels_dir.rglob("*.txt"))
    if sample_size > 0:
        all_labels = random.sample(all_labels, min(sample_size, len(all_labels)))

    results = {"valid": 0, "invalid": 0, "errors": []}
    for lbl in tqdm(all_labels, desc="Verificando etiquetas"):
        ok, errs = verify_yolo_label(lbl)
        if ok:
            results["valid"] += 1
        else:
            results["invalid"] += 1
            results["errors"].extend([f"{lbl.name}: {e}" for e in errs[:2]])

    return results


# ─── Main (uso de línea de comandos) ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Conversor de etiquetas para Wildfire_Mega_Dataset"
    )
    parser.add_argument(
        "--action",
        choices=["verify", "convert_voc", "convert_bowfire", "remap_yolo"],
        required=True,
        help="Acción a ejecutar"
    )
    parser.add_argument("--src",  type=str, help="Directorio fuente")
    parser.add_argument("--dst",  type=str, help="Directorio destino")
    parser.add_argument("--dataset", type=str, default="gengyanlei",
                        help="Nombre del dataset para remapeo numérico")
    args = parser.parse_args()

    src = Path(args.src) if args.src else None
    dst = Path(args.dst) if args.dst else None

    if args.action == "verify" and src:
        print(f"🔍 Verificando etiquetas en {src}...")
        results = batch_verify(src)
        print(f"  ✅ Válidas:   {results['valid']}")
        print(f"  ❌ Inválidas: {results['invalid']}")
        if results["errors"]:
            print("  Primeros errores:")
            for e in results["errors"][:10]:
                print(f"    {e}")

    elif args.action == "convert_voc" and src and dst:
        print(f"🔄 Convirtiendo VOC XML → YOLO en {dst}...")
        stats = batch_convert_voc_to_yolo(src, dst)
        print(f"  Convertidos: {stats['converted']}")
        print(f"  Omitidos:    {stats['skipped']}")
        print(f"  Errores:     {stats['errors']}")

    elif args.action == "convert_bowfire" and src and dst:
        dst_img = Path(dst) / "images"
        dst_lbl = Path(dst) / "labels"
        print(f"🔄 Convirtiendo BoWFire → YOLO en {dst}...")
        stats = convert_bowfire_to_yolo(src, dst_img, dst_lbl)
        print(f"  fire:  {stats['fire']}")
        print(f"  smoke: {stats['smoke']}")
        print(f"  notfire ignorados: {stats['notfire_skipped']}")

    elif args.action == "remap_yolo" and src and dst:
        id_map = NUMERIC_MAPS.get(args.dataset, {0: 0, 1: 1})
        print(f"🔄 Remapeando YOLO IDs para dataset '{args.dataset}'...")
        print(f"   Mapa: {id_map}")
        src_p = Path(src)
        dst_p = Path(dst)
        count = 0
        for lbl in tqdm(sorted(src_p.rglob("*.txt"))):
            out = dst_p / lbl.relative_to(src_p)
            n = convert_yolo_to_yolo(lbl, out, id_map)
            count += n
        print(f"  ✅ Total anotaciones remapeadas: {count}")

    else:
        parser.print_help()
