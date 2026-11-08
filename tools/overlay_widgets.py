from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import statistics
import subprocess
import math
from scipy.spatial.transform import Rotation as R
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont
from autocut.dataset.avinox_proto import parse

AVAILABLE_WIDGETS = {
    "speed": "Velocità corrente (km/h)",
    "slope": "Pendenza della strada (%)",
    "alt": "Altitudine numerica (m)",
    "track": "Traccia GPS con cursore in movimento",
    "elevation": "Profilo altimetrico grafico con cursore e area sfumata",
    "power": "Power totale numerica, con livello assistenza e tachimetro grafico",
    "pitch": "Angolo di beccheggio",
    "gear": "Marcia corrente (1-12)",
    "datetime": "Data e ora"
}

ASSIST_LVL = ["off", "eco", "trail", "turbo", "auto", "boost", "walk"]
ASSIST_CLR = [(0, 0, 0, 255), (46, 204, 113, 255), (241, 196, 15, 255), (231, 76, 60, 255), (128, 0, 128, 255), (255, 165, 0, 255)]
# Parametri del tachimetro
MAX_WATT = 1000  # Valore massimo della scala (es. 1000W)

SCRIPT_DIR = Path(__file__).resolve().parent
FONT_PATH = SCRIPT_DIR / "Fredoka-Bold.ttf"
if not FONT_PATH.exists():
    raise FileNotFoundError(f"❌ ERRORE CRITICO: Non trovo il file del font in: {FONT_PATH}")


def load_fonts_for_size(height: int):
    try:
        font_large = ImageFont.truetype(str(FONT_PATH), int(height * 0.52))
        font_small = ImageFont.truetype(str(FONT_PATH), int(height * 0.20))
        return font_large, font_small
    except Exception as e:
        raise RuntimeError(f"❌ Errore durante il parsing del font TrueType: {e}")


def lerp_color(color1, color2, t):
    """
    Linearly interpolates between two RGB colors.
    :param color1: Tuple/list of (R, G, B) from 0-255 or 0.0-1.0
    :param color2: Tuple/list of (R, G, B) from 0-255 or 0.0-1.0
    :param t: Scalar from 0.0 to 1.0
    :return: Interpolated (R, G, B) color
    """
    return tuple(int(a + (b - a) * t) for a, b in zip(color1, color2))


def get_color_for_watt(watt):
  """Restituisce il colore in base alle zone di potenza"""
  if watt < 300:
    return lerp_color((0, 0, 0, 255), (46, 204, 113, 255), watt/300)  # Verde (Agile / Z1-Z2)
  elif watt < 600:
    return lerp_color((46, 204, 113, 255), (195, 196, 15, 255), (watt-300)/300)  # Giallo (Soglia / Z3-Z4)
  else:
    return lerp_color((195, 196, 15, 255), (231, 76, 60, 255), (watt-600)/(MAX_WATT-600))  # Rosso (Fuorigiri / Z5)


def create_single_widget(label, value_str, unit, height, width, value_color=(255, 255, 255, 255)):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    font_large, font_small = load_fonts_for_size(height)
    stroke_w = max(2, int(height * 0.04))
    
    draw.text((int(width * 0.05), int(height * 0.08)), label, fill=(160, 165, 175, 255), font=font_small)
    draw.text((int(width * 0.05), int(height * 0.30)), value_str, fill=value_color, font=font_large, stroke_width=stroke_w, stroke_fill=(0, 0, 0, 255))
    draw.text((int(width * 0.65), int(height * 0.52)), unit, fill=(180, 185, 195, 255), font=font_small)
    return img


def create_gear_widget(gear_val, height):
    """Disegna una silhouette di un pignone stilizzato con dentro il numero della marcia."""
    width = height  # Widget quadrato
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    scale = height / 100.0
    cx, cy = width // 2, height // 2
    r_outer = int(36 * scale)
    r_inner = int(26 * scale)

    # 1. Disegno dei "denti" del pignone geometrici attorno al cerchio
    num_teeth = 10
    for i in range(num_teeth):
        angle = i * (2 * math.pi / num_teeth)
        # Coordinate per ogni dentino del pignone
        x1 = cx + int((r_outer + 4 * scale) * math.cos(angle - 0.15))
        y1 = cy + int((r_outer + 6 * scale) * math.sin(angle - 0.15))
        x2 = cx + int((r_outer + 4 * scale) * math.cos(angle + 0.15))
        y2 = cy + int((r_outer + 6 * scale) * math.sin(angle + 0.15))
        x3 = cx + int(r_outer * math.cos(angle + 0.25))
        y3 = cy + int(r_outer * math.sin(angle + 0.25))
        x4 = cx + int(r_outer * math.cos(angle - 0.25))
        y4 = cy + int(r_outer * math.sin(angle - 0.25))
        draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=(40, 40, 45, 230), outline=(0, 0, 0, 255))

    # 2. Corpo centrale del pignone (cerchio pieno con bordo scuro)
    draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], fill=(50, 55, 60, 240), outline=(0, 0, 0, 255), width=max(2, int(3 * scale)))
    draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=(20, 22, 25, 0), outline=(80, 85, 90, 255), width=max(1, int(2 * scale)))

    # 3. Testo della marcia al centro
    gear_text = "--" if gear_val is None else f"{int(gear_val)}"
    font_large, _ = load_fonts_for_size(int(height * 1.1))  # Font proporzionato per il cerchio
    
    draw.text(
        (cx, cy - int(2 * scale)),
        gear_text,
        fill=(255, 255, 255, 255),
        anchor="mm",
        font=font_large,
        stroke_width=max(2, int(3 * scale)),
        stroke_fill=(0, 0, 0, 255),
    )

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

class WebMWriter:
    """Stream RGBA frames directly to FFmpeg and encode a transparent VP9 WebM."""

    def __init__(self, output: Path, width: int, height: int, fps: float, crf: int):
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg non trovato. Installa FFmpeg e assicurati che sia nel PATH."
            )

        self.output = output
        self.width = width
        self.height = height
        # VP9 4:2:0 requires even dimensions. Pad with transparent pixels
        # rather than cropping or rescaling the widget.
        self.encoded_width = width + (width % 2)
        self.encoded_height = height + (height % 2)

        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgba",
            "-video_size", f"{self.encoded_width}x{self.encoded_height}",
            "-framerate", f"{fps:.8f}",
            "-i", "pipe:0",
            "-an",
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-crf", str(crf),
            "-b:v", "0",
            "-auto-alt-ref", "0",
            "-metadata:s:v:0", "alpha_mode=1",
            "-deadline", "good",
            "-cpu-used", "2",
            str(output),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if self.process.stdin is None:
            raise RuntimeError("Impossibile aprire lo stream di input di FFmpeg.")

    def write(self, frame: Image.Image) -> None:
        if frame.mode != "RGBA":
            frame = frame.convert("RGBA")
        if frame.size != (self.width, self.height):
            raise ValueError(
                f"Dimensione frame inattesa: {frame.size}, "
                f"attesa {(self.width, self.height)}"
            )

        if (self.encoded_width, self.encoded_height) != frame.size:
            padded = Image.new(
                "RGBA",
                (self.encoded_width, self.encoded_height),
                (0, 0, 0, 0),
            )
            padded.paste(frame, (0, 0))
            frame = padded

        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            error = self.process.stderr.read().decode(errors="replace")
            raise RuntimeError(f"FFmpeg ha interrotto la codifica di {self.output}: {error}") from exc

    def close(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        error = self.process.stderr.read().decode(errors="replace")
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg non è riuscito a generare {self.output}: {error}"
            )


def estimate_fps(samples: list[dict]) -> float:
    """Estimate the source cadence from positive timestamp deltas."""
    timestamps = [float(sample["timestamp"]) for sample in samples]
    deltas = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:])
        if later > earlier
    ]
    if not deltas:
        return 1.0
    return 1.0 / statistics.median(deltas)


def upsample_samples(samples: list[dict], target_fps: float) -> list[dict]:
    """Interpola linearmente i campioni del proto per raggiungere il target_fps desiderato."""
    if len(samples) < 2:
        return samples

    upsampled = []
    # Assumiamo che i campioni originali distino 1 secondo l'uno dall'altro
    # Calcoliamo quanti sotto-frame generare per ogni intervallo di 1 secondo
    steps_per_sec = int(round(target_fps))
    
    for i in range(len(samples) - 1):
        s_curr = samples[i]
        s_next = samples[i + 1]
        
        for step in range(steps_per_sec):
            alpha = step / float(steps_per_sec)
            
            # Creiamo un campione interpolato clonando il corrente
            interpolated = s_curr.copy()
            
            # Interpolazione lineare dei campi numerici principali
            for key in ["speed", "gradient", "altitude", "totalPower", "latitude", "longitude"]:
                val_curr = s_curr.get(key)
                val_next = s_next.get(key)
                if val_curr is not None and val_next is not None:
                    interpolated[key] = val_curr + (val_next - val_curr) * alpha
                    
            upsampled.append(interpolated)
            
    # Aggiungiamo l'ultimo campione
    upsampled.append(samples[-1])
    return upsampled

def main():
    parser = argparse.ArgumentParser(
        description="Genera overlay WebM trasparenti dai campioni Avinox"
    )
    parser.add_argument("recording", nargs="?", type=Path, help="File .proto Avinox")
    parser.add_argument("--output-dir", type=Path, default=Path("output_widgets"),
                        help="Cartella dei WebM generati")
    parser.add_argument("--list-widgets", action="store_true",
                        help="Mostra la lista dei widget disponibili ed esce")
    parser.add_argument(
        "--widgets", type=str,
        default="speed,slope,pitch,alt,track,elevation,power,datetime",
        help="Widget da generare separati da virgola. Default: tutti."
    )
    parser.add_argument("--size", type=int, default=110,
                        help="Altezza di riferimento in pixel per i widget. Default: 110.")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Secondo di inizio della registrazione")
    parser.add_argument("--duration", type=float, default=None,
                        help="Durata in secondi da esportare")
    parser.add_argument(
        "--track-mode", type=str, choices=["full", "window"], default="full",
        help="Modalità traccia/altimetria: intero percorso o solo intervallo esportato."
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="FPS del WebM. Default: cadenza mediana dei campioni sorgente."
    )
    parser.add_argument(
        "--crf", type=int, default=30,
        help="Qualità VP9: valori più bassi producono file più grandi e qualità maggiore (default: 30)."
    )

    args = parser.parse_args()

    if args.list_widgets:
        print("\n📋 Widget disponibili:")
        for name, desc in AVAILABLE_WIDGETS.items():
            print(f"  - {name: <12} : {desc}")
        print("\nEsempio d'uso: python tools/overlay_widgets.py file.proto --widgets power,speed\n")
        return

    if not args.recording:
        parser.error("Devi specificare il file .proto di registrazione (oppure usa --list-widgets).")
    if not args.recording.is_file():
        parser.error(f"File Avinox non trovato: {args.recording}")
    if args.size <= 0:
        parser.error("--size deve essere maggiore di zero.")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps deve essere maggiore di zero.")
    if not 0 <= args.crf <= 63:
        parser.error("--crf deve essere compreso tra 0 e 63.")
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg non trovato. Installa FFmpeg e assicurati che sia nel PATH.")

    selected_widgets = list(dict.fromkeys(
        widget.strip().lower() for widget in args.widgets.split(",") if widget.strip()
    ))
    for widget in selected_widgets:
        if widget not in AVAILABLE_WIDGETS:
            parser.error(
                f"Widget sconosciuto '{widget}'. Usa --list-widgets per vedere quelli disponibili."
            )
    if not selected_widgets:
        parser.error("Seleziona almeno un widget con --widgets.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    decoded = parse(args.recording)
    all_samples = sorted(
        (sample for sample in decoded["samples"] if sample.get("timestamp") is not None),
        key=lambda sample: sample["timestamp"],
    )
    if not all_samples:
        parser.error("Il file Avinox non contiene campioni con timestamp.")

    avinox_start = float(decoded["metadata"].get("start") or all_samples[0]["timestamp"])

    # Keep only complete GPS fixes, so latitude and longitude remain paired.
    gps_samples = [
        sample for sample in all_samples
        if sample.get("latitude") is not None and sample.get("longitude") is not None
    ]
    all_lats = [float(sample["latitude"]) for sample in gps_samples]
    all_lons = [float(sample["longitude"]) for sample in gps_samples]
    all_alts = [
        float(sample["altitude"]) if sample.get("altitude") is not None else 0.0
        for sample in all_samples
    ]

    filtered_samples = []
    for sample in all_samples:
        sample_time = float(sample["timestamp"]) - avinox_start
        if sample_time < args.start:
            continue
        if args.duration is not None and sample_time > args.start + args.duration:
            break
        filtered_samples.append(sample)

    if not filtered_samples:
        print("Nessun campione trovato nell'intervallo temporale specificato.")
        return

    fps = args.fps or estimate_fps(filtered_samples)

    print(
        f"Upsamplig di {len(filtered_samples)} frame per widget @ {fps:.3f} FPS "
    )
    if fps > 1.0:
        filtered_samples = upsample_samples(filtered_samples, fps)

    # Preparazione liste globali per traccia e altimetria (sull'intero file o sulla finestra)
    gps_samples = [s for s in all_samples if s.get("latitude") is not None and s.get("longitude") is not None]
    all_lats = [float(s["latitude"]) for s in gps_samples]
    all_lons = [float(s["longitude"]) for s in gps_samples]
    all_alts = [float(s.get("altitude", 0.0)) if s.get("altitude") is not None else 0.0 for s in all_samples]

    print(f"Generazione di {len(filtered_samples)} frame (track-mode: {args.track_mode}) in {args.output_dir}...")

    writers: dict[str, WebMWriter] = {}
    try:
        # Start one encoder per widget: each resulting WebM remains independently
        # positionable in the downstream video editor.
        for widget in selected_widgets:
            if widget == "speed" or widget == "slope" or widget == "alt" or widget == "pitch":
                width = int(args.size * 2.36)
            elif widget == "track":
                width = int(args.size * 1.55)
            elif widget == "elevation":
                width = int(args.size * 2.2)
            elif widget == "gear":
                width = args.size
            elif widget == "datetime":
                width = int(args.size * 4.72)
            else:
                width = int(args.size * 1.5)
            writers[widget] = WebMWriter(
                args.output_dir / f"{widget}.webm",
                width, args.size, fps, args.crf,
            )

        for idx_out, sample in enumerate(filtered_samples):
            # Calcolo posizione cursore GPS / Altimetria in base al timestamp corrente
            curr_t = float(sample["timestamp"])

            if args.track_mode == "full":
                track_lats, track_lons = all_lats, all_lons
                track_alts = all_alts
                # Trova il fix GPS valido più vicino temporalmente nel dataset completo
                valid_gps = [
                    (i, abs(float(s["timestamp"]) - curr_t)) 
                    for i, s in enumerate(all_samples) 
                    if s.get("latitude") is not None and s.get("longitude") is not None
                ]
                track_idx = min(valid_gps, key=lambda x: x[1])[0] if valid_gps else -1
                
                # Indice altimetrico globale più vicino
                elev_idx = min(range(len(all_samples)), key=lambda i: abs(float(all_samples[i]["timestamp"]) - curr_t))
            else:
                window_fixes = [
                    item for item in filtered_samples
                    if item.get("latitude") is not None and item.get("longitude") is not None
                ]
                track_lats = [float(item["latitude"]) for item in window_fixes]
                track_lons = [float(item["longitude"]) for item in window_fixes]
                track_alts = [
                    float(item.get("altitude", 0.0)) if item.get("altitude") is not None else 0.0
                    for item in filtered_samples
                ]
                window_gps_indices = [
                    i for i, item in enumerate(filtered_samples) 
                    if item.get("latitude") is not None and item.get("longitude") is not None
                ]
                if window_gps_indices:
                    closest_w_idx = min(window_gps_indices, key=lambda i: abs(float(filtered_samples[i]["timestamp"]) - curr_t))
                    track_idx = window_gps_indices.index(closest_w_idx)
                else:
                    track_idx = -1
                elev_idx = idx_out

            if "speed" in writers:
                raw_speed = sample.get("speed", 0.0)
                speed = float(raw_speed) if raw_speed is not None else 0.0
                speed_text = "----" if speed is None else f"{speed:.1f}"
                writers["speed"].write(
                    create_single_widget("VELOCITY", speed_text, "km/h", height=args.size, width=int(args.size * 2.36))
                )

            if "power" in writers:
                raw_power = sample.get("totalPower", 0.0)
                raw_assist = sample.get("assist")
                power = float(raw_power) if raw_power is not None else None
                assist = int(raw_assist) if raw_assist is not None else None
                writers["power"].write(create_power_widget(power, assist, args.size))

            if "slope" in writers:
                raw_gradient = sample.get("gradient", 0.0)
                gradient = float(raw_gradient) if raw_gradient is not None else None
                if gradient is None:
                    slope_color = (70, 220, 100, 255)
                    slope_text = "----"
                else:
                    slope_color = (70, 220, 100, 255) if gradient >= 0 else (240, 100, 90, 255)
                    slope_text = f"{gradient:+.1f}"
                writers["slope"].write(
                    create_single_widget(
                        "SLOPE", slope_text, "%", height=args.size, width=int(args.size * 2.36), value_color=slope_color
                    )
                )


            if "datetime" in writers:
                timestamp = sample.get("timestamp")

                if timestamp is not None:
                    dt = datetime.fromtimestamp(timestamp).astimezone()

                    date_str = dt.strftime("%d/%m/%Y")
                    time_str = dt.strftime("%H:%M")

                    writers["datetime"].write(
                        create_single_widget(
                            "",
                            f"{date_str} {time_str}",
                            "",
                            height=args.size,
                            width=int(args.size * 4.72),
                            value_color=(255, 255, 255, 255)
                        )
                    )


            if "pitch" in writers:
                raw_imu_x = sample.get("imuX", 0.0)
                raw_imu_y = sample.get("imuY", 0.0)
                raw_imu_z = sample.get("imuZ", 0.0)
                imu_x = float(raw_imu_x) if raw_imu_x is not None else None
                imu_y = float(raw_imu_y) if raw_imu_y is not None else None
                imu_z = float(raw_imu_z) if raw_imu_z is not None else None

                if imu_x is None or imu_y is None or imu_z is None:
                    pitch_color = (70, 220, 100, 255)
                    pitch_text = "---"
                else:
                    w = math.sqrt(1.0 - (imu_x * imu_x + imu_y * imu_y + imu_z * imu_z))
                    r = R.from_quat((imu_x, imu_y, imu_z, w))
                    _, pitch,_ = r.as_euler('XZY', degrees=True)
                    pitch += 50.7 #TBD bias to be computed automatically
                    pitch_text = f"{pitch:.1f}"
                    pitch_color = (70, 220, 100, 255) if pitch >= 0 else (240, 100, 90, 255)
                writers["pitch"].write(
                    create_single_widget(
                        "PITCH", pitch_text, "°", height=args.size, width=int(args.size * 2.36), value_color=pitch_color
                    )
                )

            if "alt" in writers:
                raw_altitude = sample.get("altitude", 0.0)
                altitude = float(raw_altitude) if raw_altitude is not None else None
                alt_text = "----" if altitude is None else f"{int(altitude)}"
                writers["alt"].write(
                    create_single_widget("ALTITUDE", alt_text, "m", height=args.size, width=int(args.size * 2.36))
                )

            if "gear" in writers:
                raw_gear = sample.get("gear")
                gear_val = int(raw_gear) if raw_gear is not None else None
                writers["gear"].write(create_gear_widget(gear_val, args.size))

            if "track" in writers:
                writers["track"].write(
                    create_track_widget(track_lats, track_lons, current_idx=track_idx, height=args.size)
                )

            if "elevation" in writers:
                writers["elevation"].write(
                    create_elevation_widget(track_alts, current_idx=elev_idx, height=args.size)
                )

            if (idx_out + 1) % max(1, int(fps * 10)) == 0:
                print(f"Generati {idx_out + 1}/{len(filtered_samples)} frame...", flush=True)

        for writer in writers.values():
            writer.close()
        writers.clear()

    finally:
        # Close or terminate every outstanding FFmpeg process on failure.
        for writer in writers.values():
            if writer.process.poll() is None:
                writer.process.terminate()
                writer.process.wait()

    print("Esportazione WebM completata:")
    for widget in selected_widgets:
        print(f"  - {args.output_dir / (widget + '.webm')}")
    print(f"Frame per widget: {len(filtered_samples)} | FPS: {fps:.3f}")
    
if __name__ == "__main__":
    main()
