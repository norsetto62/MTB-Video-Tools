from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from autocut.dataset.avinox_proto import parse

SCRIPT_DIR = Path(__file__).resolve().parent
FONT_PATH = SCRIPT_DIR / "Fredoka-Bold.ttf"

AVAILABLE_WIDGETS = {
    "speed": "Velocità corrente (km/h)",
    "slope": "Pendenza della strada (%)",
    "alt": "Altitudine numerica (m)",
    "track": "Traccia GPS con cursore in movimento",
    "elevation": "Profilo altimetrico grafico con cursore e area sfumata",
    "power": "Power totale numerica, con livello assistenza e tachimetro grafico"
}

ASSIST_LVL = ["off", "eco", "trail", "turbo", "auto", "boost", "walk"]
ASSIST_CLR = [(0, 0, 0, 255), (46, 204, 113, 255), (241, 196, 15, 255), (231, 76, 60, 255), (128, 0, 128, 255), (255, 165, 0, 255)]
# Parametri del tachimetro
MAX_WATT = 1000  # Valore massimo della scala (es. 1000W)

if not FONT_PATH.exists():
    raise FileNotFoundError(f"❌ ERRORE CRITICO: Non trovo il file del font in: {FONT_PATH}")


def load_fonts_for_size(height: int):
    try:
        font_large = ImageFont.truetype(str(FONT_PATH), int(height * 0.52))
        font_small = ImageFont.truetype(str(FONT_PATH), int(height * 0.20))
        return font_large, font_small
    except Exception as e:
        raise RuntimeError(f"❌ Errore durante il parsing del font TrueType: {e}")


def get_color_for_watt(watt):
  """Restituisce il colore in base alle zone di potenza"""
  if watt < 300:
    return (46, 204, 113, 255)  # Verde (Agile / Z1-Z2)
  elif watt < 600:
    return (241, 196, 15, 255)  # Giallo (Soglia / Z3-Z4)
  else:
    return (231, 76, 60, 255)  # Rosso (Fuorigiri / Z5)
  
def create_single_widget(label, value_str, unit, height, value_color=(255, 255, 255, 255)):
    width = int(height * 2.36)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    font_large, font_small = load_fonts_for_size(height)
    stroke_w = max(2, int(height * 0.04))
    
    draw.text((int(width * 0.05), int(height * 0.08)), label, fill=(160, 165, 175, 255), font=font_small)
    draw.text((int(width * 0.05), int(height * 0.30)), value_str, fill=value_color, font=font_large, stroke_width=stroke_w, stroke_fill=(0, 0, 0, 255))
    draw.text((int(width * 0.65), int(height * 0.52)), unit, fill=(180, 185, 195, 255), font=font_small)
    return img


def create_track_widget(lats, lons, current_idx, height):
    width = int(height * 1.55)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    if not lats or not lons or len(lats) < 2:
        return img

    pad = int(width * 0.12)
    w_draw, h_draw = width - (2 * pad), height - (2 * pad)
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    lat_range = max_lat - min_lat if max_lat != min_lat else 1e-6
    lon_range = max_lon - min_lon if max_lon != min_lon else 1e-6

    points = [
        (pad + ((lon - min_lon) / lon_range) * w_draw, height - pad - ((lat - min_lat) / lat_range) * h_draw)
        for lat, lon in zip(lats, lons)
    ]
    
    draw.line(points, fill=(255, 255, 255, 180), width=max(2, int(height * 0.025)))
    if 0 <= current_idx < len(points):
        cx, cy = points[current_idx]
        draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(255, 255, 255, 120))
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)

    return img


def create_elevation_widget(altitudes, current_idx, height):
    width = int(height * 2.2)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    if not altitudes or len(altitudes) < 2:
        return img

    pad_x = int(width * 0.08)
    pad_y = int(height * 0.15)
    w_draw = width - (2 * pad_x)
    h_draw = height - (2 * pad_y)

    min_alt = min(altitudes)
    max_alt = max(altitudes)
    alt_range = max_alt - min_alt if max_alt != min_alt else 1.0

    points = []
    for i, alt in enumerate(altitudes):
        x = pad_x + (i / (len(altitudes) - 1)) * w_draw
        y = height - pad_y - ((alt - min_alt) / alt_range) * h_draw
        points.append((x, y))

    polygon_points = [(points[0][0], height - pad_y)] + points + [(points[-1][0], height - pad_y)]
    draw.polygon(polygon_points, fill=(255, 255, 255, 40))

    draw.line(points, fill=(255, 255, 255, 220), width=max(2, int(height * 0.03)))

    if 0 <= current_idx < len(points):
        cx, cy = points[current_idx]
        current_alt = altitudes[current_idx]
        
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)
        _, font_small = load_fonts_for_size(height)
        draw.text((pad_x, int(height * 0.05)), f"ALT: {int(current_alt)} m", fill=(255, 255, 255, 255), font=font_small, stroke_width=2, stroke_fill=(0, 0, 0, 255))

    return img


def create_power_widget(power, assist, height):
    width = int(height * 1.5)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Fattore di scala basato su un'altezza di riferimento di 200px.
    scale = height / 200.0
    radius = int(100 * scale)
    arc_width = max(2, int(14 * scale))
    outline_width = max(2, int(5 * scale))
    center_x = width // 2
    center_y_arc = int(120 * scale)

    arc_box = [
        center_x - radius,
        center_y_arc - radius,
        center_x + radius,
        center_y_arc + radius,
    ]

    # Bordo scuro sotto l'arco: mantiene il tachimetro leggibile anche
    # quando il video sottostante è molto chiaro.
    draw.arc(
        arc_box,
        start=180,
        end=360,
        fill=(0, 0, 0, 235),
        width=arc_width + outline_width * 2,
    )
    draw.arc(
        arc_box,
        start=180,
        end=360,
        fill=(255, 255, 255, 175),
        width=arc_width,
    )

    if power is not None:
        current_watt = max(0, min(power, MAX_WATT))
        current_angle = 180 + (current_watt / MAX_WATT) * 180
        gauge_color = get_color_for_watt(current_watt)

        if current_angle > 180:
            draw.arc(
                arc_box,
                start=180,
                end=current_angle,
                fill=(0, 0, 0, 235),
                width=arc_width + outline_width * 2,
            )
            draw.arc(
                arc_box,
                start=180,
                end=current_angle,
                fill=gauge_color,
                width=arc_width,
            )

    _, font_small = load_fonts_for_size(height)
    text_stroke = max(1, int(2 * scale))
    text_outline = (0, 0, 0, 255)
    power_text = "----" if power is None else f"{power:.0f}W"

    draw.text(
        (center_x, center_y_arc - int(10 * scale)),
        power_text,
        fill=(255, 255, 255, 255),
        anchor="mm",
        font=font_small,
        stroke_width=text_stroke,
        stroke_fill=text_outline,
    )

    # Saetta disegnata vettorialmente: non dipende dai glifi disponibili nel font.
    bolt_cx = center_x
    bolt_cy = center_y_arc + int(26 * scale)
    bolt_w = int(12 * scale)
    bolt_h = int(20 * scale)
    bolt_points = [
        (bolt_cx + int(2 * scale), bolt_cy - bolt_h),
        (bolt_cx - bolt_w, bolt_cy + int(1 * scale)),
        (bolt_cx - int(2 * scale), bolt_cy + int(1 * scale)),
        (bolt_cx - int(1 * scale), bolt_cy + bolt_h),
        (bolt_cx + bolt_w, bolt_cy - int(3 * scale)),
        (bolt_cx + int(3 * scale), bolt_cy - int(3 * scale)),
    ]
    draw.polygon(
        bolt_points,
        fill=(255, 210, 45, 255),
        outline=(0, 0, 0, 255),
    )
    # Rinforza il contorno della saetta alle dimensioni video più grandi.
    draw.line(
        bolt_points + [bolt_points[0]],
        fill=(0, 0, 0, 255),
        width=max(1, int(1.5 * scale)),
        joint="curve",
    )

    if assist is not None and 0 <= assist < len(ASSIST_LVL):
        assist_color = ASSIST_CLR[assist] if assist < len(ASSIST_CLR) else (180, 180, 180, 255)
        draw.text(
            (center_x, center_y_arc - int(45 * scale)),
            ASSIST_LVL[assist].upper(),
            fill=assist_color,
            anchor="mm",
            font=font_small,
            stroke_width=text_stroke,
            stroke_fill=text_outline,
        )

    return img

def main():
    parser = argparse.ArgumentParser(description="Genera widget PNG dai campioni del file .proto Avinox")
    parser.add_argument("recording", nargs="?", type=Path, help="File .proto Avinox")
    parser.add_argument("--output-dir", type=Path, default=Path("output_widgets"), help="Cartella output")
    parser.add_argument("--list-widgets", action="store_true", help="Mostra la lista dei widget disponibili ed esce")
    parser.add_argument(
        "--widgets", 
        type=str, 
        default="speed,slope,alt,track,elevation,power", 
        help="Widget da generare separati da virgola. Default: tutti."
    )
    parser.add_argument(
        "--size", 
        type=int, 
        default=110, 
        help="Altezza di riferimento in pixel per i widget. Default: 110."
    )
    parser.add_argument("--start", type=float, default=0.0, help="Secondo di inizio della registrazione")
    parser.add_argument("--duration", type=float, default=None, help="Durata in secondi da esportare")
    parser.add_argument(
        "--track-mode", 
        type=str, 
        choices=["full", "window"], 
        default="full", 
        help="Modalità traccia/altimetria: 'full' (intero percorso) o 'window' (solo intervallo start-duration). Default: full."
    )
    
    args = parser.parse_args()

    if args.list_widgets:
        print("\n📋 Widget disponibili:")
        for name, desc in AVAILABLE_WIDGETS.items():
            print(f"  - {name: <12} : {desc}")
        print("\nEsempio d'uso: python script.py file.proto --track-mode window --start 30 --duration 120\n")
        return

    if not args.recording:
        parser.error("❌ Errore: Devi specificare il file .proto di registrazione (oppure usa --list-widgets).")

    selected_widgets = [w.strip().lower() for w in args.widgets.split(",")]
    for w in selected_widgets:
        if w not in AVAILABLE_WIDGETS:
            parser.error(f"❌ Errore: Widget sconosciuto '{w}'. Usa --list-widgets per vedere quelli disponibili.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    decoded = parse(args.recording)
    all_samples = sorted(
        (s for s in decoded["samples"] if s.get("timestamp") is not None),
        key=lambda s: s["timestamp"],
    )
    
    avinox_start = float(decoded["metadata"].get("start") or all_samples[0]["timestamp"])
    
    # Campi corretti: latitude e longitude
    all_lats = [float(s["latitude"]) for s in all_samples if "latitude" in s and s["latitude"] is not None]
    all_lons = [float(s["longitude"]) for s in all_samples if "longitude" in s and s["longitude"] is not None]
    all_alts = [float(s.get("altitude", 0.0)) for s in all_samples]

    # Filtraggio dei campioni per intervallo temporale
    filtered_samples = []
    for s in all_samples:
        sample_time = float(s["timestamp"]) - avinox_start
        if sample_time < args.start:
            continue
        if args.duration is not None and sample_time > (args.start + args.duration):
            break
        filtered_samples.append(s)

    if not filtered_samples:
        print("❌ Nessun campione trovato nell'intervallo temporale specificato.")
        return

    print(f"Generazione di {len(filtered_samples)} campioni (track-mode: {args.track_mode}) in {args.output_dir}...")
    
    for idx_out, sample in enumerate(filtered_samples):
        global_idx = all_samples.index(sample)

        # Selezioniamo i dati in base alla modalità scelta (full o window)
        if args.track_mode == "full":
            track_lats, track_lons = all_lats, all_lons
            track_alts = all_alts
            track_idx = global_idx
        else:
            # Modalità window: estrae solo la porzione dell'intervallo
            window_lats_lons = [(float(s["latitude"]), float(s["longitude"]))
                                for s in filtered_samples 
                                if s.get("latitude") is not None and s.get("longitude") is not None
                                ]
            track_lats = [lat for lat, lon in window_lats_lons]
            track_lons = [lon for lat, lon in window_lats_lons]
            track_alts = [float(s.get("altitude", 0.0)) for s in filtered_samples]
            track_idx = idx_out

        if "speed" in selected_widgets:
            raw_speed = sample.get("speed", 0.0)
            speed = float(raw_speed) if raw_speed is not None else None
            speed_text = "----" if speed is None else f"{speed:.1f}"
            img_speed = create_single_widget("VELOCITY", speed_text, "km/h", height=args.size)
            img_speed.save(args.output_dir / f"speed_{idx_out:06d}.png", "PNG")

        if "power" in selected_widgets:
            raw_power = sample.get("totalPower", 0.0)
            raw_assist = sample.get("assist", 0.0)
            power = float(raw_power) if raw_power is not None else None
            assist = int(raw_assist) if raw_power is not None else None
            img_pow = create_power_widget(power, assist, args.size)
            img_pow.save(args.output_dir / f"power_{idx_out:06d}.png", "PNG")

        if "slope" in selected_widgets:
            raw_gradient = sample.get("gradient", 0.0)
            gradient = float(raw_gradient) if raw_gradient is not None else None
            if gradient is None:
                slope_color = (70, 220, 100, 255)
                slope_text = "----"
            else:
                slope_color = (70, 220, 100, 255) if gradient >= 0 else (240, 100, 90, 255)
                slope_text = f"{gradient:+.1f}"
            img_slope = create_single_widget("SLOPE", slope_text, "%", height=args.size, value_color=slope_color)
            img_slope.save(args.output_dir / f"slope_{idx_out:06d}.png", "PNG")

        if "alt" in selected_widgets:
            raw_altitude = sample.get("altitude", 0.0)
            altitude = float(raw_altitude) if raw_altitude is not None else None
            alt_text = "----" if altitude is None else f"{int(altitude)}"
            img_alt = create_single_widget("ALTITUDE", alt_text, "m", height=args.size)
            img_alt.save(args.output_dir / f"alt_{idx_out:06d}.png", "PNG")

        if "track" in selected_widgets:
            img_track = create_track_widget(track_lats, track_lons, current_idx=track_idx, height=args.size)
            img_track.save(args.output_dir / f"output_widgets/track_{idx_out:06d}.png" if False else args.output_dir / f"track_{idx_out:06d}.png", "PNG")

        if "elevation" in selected_widgets:
            img_elev = create_elevation_widget(track_alts, current_idx=track_idx, height=args.size)
            img_elev.save(args.output_dir / f"elevation_{idx_out:06d}.png", "PNG")

    print(f"Esportazione completata con successo!")

if __name__ == "__main__":
    main()