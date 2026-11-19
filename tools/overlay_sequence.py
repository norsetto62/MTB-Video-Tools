from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from autocut.dataset.avinox_proto import parse

SCRIPT_DIR = Path(__file__).resolve().parent
FONT_PATH = SCRIPT_DIR / "Fredoka-Bold.ttf"

# Dimensioni compatte per il singolo widget senza box
WIDTH = 260
HEIGHT = 110

if not FONT_PATH.exists():
    raise FileNotFoundError(
        f"❌ ERRORE CRITICO: Non trovo il file del font in: {FONT_PATH}\n"
        "Assicurati che 'Fredoka-Bold.ttf' sia nella stessa cartella dello script."
    )

try:
    # Caricamento una tantum all'avvio ottimizzato per l'altezza del widget singolo
    FONT_LARGE = ImageFont.truetype(str(FONT_PATH), int(HEIGHT * 0.52))
    FONT_SMALL = ImageFont.truetype(str(FONT_PATH), int(HEIGHT * 0.20))
    print(f"✔ Font caricato con successo da: {FONT_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Errore durante il parsing del font TrueType: {e}")

def create_single_widget(
    label: str,
    value_str: str,
    unit: str,
    width: int = WIDTH,
    height: int = HEIGHT,
    value_color: tuple = (255, 255, 255, 255)
) -> Image.Image:
    # Canvas RGBA trasparente, senza rettangolo di sfondo
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    stroke_w = max(2, int(height * 0.04))
    
    # Posizionamento relativo all'interno del singolo widget
    label_x = int(width * 0.05)
    label_y = int(height * 0.05)
    
    number_x = int(width * 0.05)
    number_y = int(height * 0.28)
    
    unit_x = int(width * 0.62)  # Offset orizzontale per l'unità di misura
    unit_y = int(height * 0.50)  # Offset verticale per l'allineamento tipografico

    # 1. Etichetta in alto (es. VELOCITY, SLOPE, ALTITUDE)
    draw.text((label_x, label_y), label, fill=(160, 165, 175, 255), font=FONT_SMALL)
    
    # 2. Numero principale ciccione con bordo nero
    draw.text(
        (number_x, number_y), value_str, 
        fill=value_color, font=FONT_LARGE,
        stroke_width=stroke_w, stroke_fill=(0, 0, 0, 255)
    )
    
    # 3. Unità di misura affiancata e ribassata
    draw.text((unit_x, unit_y), unit, fill=(180, 185, 195, 255), font=FONT_SMALL)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Genera widget Avinox singoli e trasparenti per Kdenlive")
    parser.add_argument("recording", type=Path, help="File .proto Avinox")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate del video di destinazione")
    parser.add_argument("--start", type=float, default=0.0, help="Momento di partenza in secondi")
    parser.add_argument("--duration", type=float, default=60.0, help="Durata da esportare in secondi")
    parser.add_argument("--output-dir", type=Path, default=Path("output_widgets"), help="Cartella PNG di output")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Parsing Avinox
    decoded = parse(args.recording)
    samples = sorted(
        (s for s in decoded["samples"] if s.get("timestamp") is not None),
        key=lambda s: s["timestamp"],
    )
    avinox_start = float(decoded["metadata"].get("start") or samples[0]["timestamp"])
    
    # 2. Generazione dei frame
    total_frames = int(args.fps * args.duration)
    print(f"Generazione di {total_frames} frame per ciascun widget ({args.fps} FPS) in {args.output_dir}...")
    
    for f in range(total_frames):
        video_time = args.start + (f / args.fps)
        target_time = avinox_start + video_time
        
        # Ricerca del campione telemetrico tramite vicinanza temporale
        sample = samples[min(range(len(samples)), key=lambda i: abs(float(samples[i]["timestamp"]) - target_time))]
        
        speed = float(sample.get("speed", 0.0))
        gradient = float(sample.get("gradient", 0.0))
        altitude = float(sample.get("altitude", 0.0))
        
        # Colore dinamico per la pendenza (verde in salita/piano, rosso/arancio in discesa)
        slope_color = (70, 220, 100, 255) if gradient >= 0 else (240, 100, 90, 255)
        
        # Creazione dei 3 widget separati
        img_speed = create_single_widget("VELOCITY", f"{speed:.1f}", "km/h", value_color=(255, 255, 255, 255))
        img_slope = create_single_widget("SLOPE", f"{gradient:+.1f}", "%", value_color=slope_color)
        img_alt = create_single_widget("ALTITUDE", f"{int(altitude)}", "m", value_color=(255, 255, 255, 255))
        
        # Salvataggio sequenze separate
        img_speed.save(args.output_dir / f"speed_{f:06d}.png", "PNG")
        img_slope.save(args.output_dir / f"slope_{f:06d}.png", "PNG")
        img_alt.save(args.output_dir / f"alt_{f:06d}.png", "PNG")
        
        if f % int(args.fps * 5) == 0 and f > 0:
            print(f"Generati frame fino a {video_time:.1f}s...")

    print("Esportazione completata! Ora puoi importare le tre sequenze indipendentemente in Kdenlive.")

if __name__ == "__main__":
    main()