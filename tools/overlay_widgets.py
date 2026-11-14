from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import statistics
import subprocess
import math
import bisect
import re
import html
from scipy.spatial.transform import Rotation as R
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont
from autocut.dataset.avinox_proto import parse
from dji_action4_telemetry import extract_clip, process


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



DJI_WIDGETS = {
    "rel_alt": "Altitudine relativa al punto di decollo (m)",
    "abs_alt": "Altitudine assoluta stimata (m)",
    "camera": "Impostazioni camera: ISO, shutter, apertura, EV, temperatura colore",
    "track": "Traccia GPS DJI con cursore in movimento",
    "datetime": "Data e ora registrate dalla camera",
    "pitch": "Angolo di beccheggio ricavato dal video",
}
_SRT_TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})$")


def _srt_seconds(value: str) -> float:
    match = _SRT_TIME_RE.match(value.strip())
    if not match:
        raise ValueError(f"Timestamp SRT non valido: {value!r}")
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000.0


def _number(value: str | None) -> float | None:
    try:
        number = float(value.strip()) if value is not None else None
        return number if number is not None and math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def parse_dji_srt(path: Path) -> list[dict]:
    """Parse DJI SRT records, retaining original start and end timestamps."""
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    samples = []
    for block in re.split(r"\r?\n\s*\r?\n", raw.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        timing = re.match(r"^(.+?)\s*-->\s*(.+?)$", lines[1])
        if not timing:
            continue
        try:
            start, end = _srt_seconds(timing.group(1)), _srt_seconds(timing.group(2))
        except ValueError:
            continue
        if end <= start:
            continue

        body = html.unescape(re.sub(r"</?font[^>]*>", "", " ".join(lines[2:])))
        frame_match = re.search(r"FrameCnt:\s*(\d+)", body)
        camera_time_match = re.search(
            r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)", body
        )
        recording_datetime = None
        if camera_time_match:
            try:
                recording_datetime = datetime.strptime(
                    camera_time_match.group(1), "%Y-%m-%d %H:%M:%S.%f"
                )
            except ValueError:
                pass

        def field(name: str) -> str | None:
            match = re.search(
                rf"\[{re.escape(name)}:\s*([^\]]*)\]", body, re.IGNORECASE
            )
            return match.group(1).strip() if match else None

        altitude_match = re.search(
            r"\[rel_alt:\s*([-+\d.]+)\s+abs_alt:\s*([-+\d.]+)", body, re.IGNORECASE
        )
        samples.append({
            "timestamp": start,
            "end_timestamp": end,
            "frame": int(frame_match.group(1)) if frame_match else None,
            "recording_datetime": recording_datetime,
            "iso": _number(field("iso")),
            "shutter": field("shutter"),
            "aperture": _number(field("fnum")),
            "ev": _number(field("ev")),
            "color_temperature": _number(field("ct")),
            "latitude": _number(field("latitude")),
            "longitude": _number(field("longitude")),
            "relative_altitude": _number(altitude_match.group(1)) if altitude_match else None,
            "absolute_altitude": _number(altitude_match.group(2)) if altitude_match else None,
            "eis": field("eis"),
        })

    samples.sort(key=lambda sample: sample["timestamp"])
    if not samples:
        raise ValueError(f"Nessun record SRT DJI valido trovato in {path}")
    if any(b["timestamp"] <= a["timestamp"] for a, b in zip(samples, samples[1:])):
        raise ValueError("I timestamp iniziali dei record SRT DJI devono essere crescenti.")
    return samples


def estimate_srt_fps(samples: list[dict]) -> float:
    """Average over the covered SRT interval so 33/34 ms rounding cancels out."""
    elapsed = samples[-1]["end_timestamp"] - samples[0]["timestamp"]
    return len(samples) / elapsed if elapsed > 0 else 1.0



def create_dji_altitude_widget(label: str, value: float | None, height: int, width: int) -> Image.Image:
    """Draw a DJI altitude value with its unit aligned to the value baseline."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_large, font_small = load_fonts_for_size(height)
    stroke_w = max(2, int(height * 0.04))
    x = int(width * 0.05)
    draw.text((x, int(height * 0.08)), label, fill=(160, 165, 175, 255), font=font_small)
    value_text = "----" if value is None else f"{value:.1f}"
    value_y = int(height * 0.30)
    draw.text((x, value_y), value_text, fill=(255, 255, 255, 255), font=font_large,
              stroke_width=stroke_w, stroke_fill=(0, 0, 0, 255))
    bbox = draw.textbbox((x, value_y), value_text, font=font_large, stroke_width=stroke_w)
    unit_font = ImageFont.truetype(str(FONT_PATH), int(height * 0.30))
    unit_y = bbox[3] - unit_font.getbbox("m")[3] - int(height * 0.01)
    unit_x = bbox[2] + max(5, int(height * 0.06))
    draw.text((unit_x, unit_y), "m", fill=(220, 225, 235, 255), font=unit_font,
              stroke_width=max(1, int(height * 0.015)), stroke_fill=(0, 0, 0, 255))
    return img


def create_dji_camera_widget(sample: dict, height: int, width: int) -> Image.Image:
    """Render camera telemetry on two uncluttered rows, including EIS and color temp."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    label_font = ImageFont.truetype(str(FONT_PATH), max(12, int(height * 0.18)))
    value_font = ImageFont.truetype(str(FONT_PATH), max(14, int(height * 0.30)))
    label_color = (160, 165, 175, 255)
    value_color = (255, 255, 255, 255)
    stroke_w = max(1, int(height * 0.012))

    iso = "—" if sample["iso"] is None else f"ISO {sample['iso']:.0f}"
    shutter = sample["shutter"] or "—"
    aperture = "—" if sample["aperture"] is None else f"f/{sample['aperture']:.1f}"
    ev = "—" if sample["ev"] is None else f"EV {sample['ev']:+.1f}"
    eis = sample["eis"] or "—"
    cct = sample["color_temperature"]
    color_temp = "—" if cct is None else f"{cct:.0f} K"

    rows = [
        (("ISO", iso), ("SHUTTER", shutter), ("APERTURE", aperture)),
        (("EV", ev), ("EIS", eis), ("COLOR", color_temp)),
    ]
    col_x = [int(width * 0.04), int(width * 0.36), int(width * 0.68)]
    row_y = [int(height * 0.08), int(height * 0.53)]
    for y, row in zip(row_y, rows):
        for x, (label, value) in zip(col_x, row):
            draw.text((x, y), label, fill=label_color, font=label_font)
            draw.text((x, y + int(height * 0.19)), value, fill=value_color, font=value_font,
                      stroke_width=stroke_w, stroke_fill=(0, 0, 0, 255))
    return img


def run_dji_widgets(args, parser, selected_widgets: list[str]) -> None:
    try:
        samples = parse_dji_srt(args.recording)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if "pitch" in selected_widgets:
        if args.video is None or not args.video.is_file():
            parser.error("Per il widget pitch DJI devi specificare un video valido con --video.")
        imu_samples, _ = extract_clip(args.video)
        rows, _ = process(imu_samples, "quat", 1.0, True, "auto", ())
        log_times = [row["sample"]["t"] for row in rows]
        max_log_idx = len(rows) - 1
        if max_log_idx < 0:
            parser.error("Il video non contiene campioni IMU utilizzabili per il widget pitch.")

    filtered = [
        sample for sample in samples
        if sample["timestamp"] >= args.start
        and (args.duration is None or sample["timestamp"] <= args.start + args.duration)
    ]
    if not filtered:
        parser.error("Nessun record SRT nell'intervallo temporale richiesto.")

    fps = args.fps or estimate_srt_fps(filtered)
    gps_samples = [
        sample for sample in samples
        if sample["latitude"] is not None and sample["longitude"] is not None
    ]
    size = args.size
    widths = {
        "rel_alt": int(size * 3.0), "abs_alt": int(size * 3.0),
        "camera": int(size * 4.8), "track": int(size * 1.55),
        "datetime": int(size * 4.72), "pitch": int(size * 2.36),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    try:
        for widget in selected_widgets:
            writers[widget] = WebMWriter(
                args.output_dir / f"{widget}.webm", widths[widget], size, fps, args.crf
            )

        for sample in filtered:
            if "rel_alt" in writers:
                value = sample["relative_altitude"]
                writers["rel_alt"].write(create_dji_altitude_widget(
                    "RELATIVE ALTITUDE", value, size, widths["rel_alt"],
                ))
            if "abs_alt" in writers:
                value = sample["absolute_altitude"]
                writers["abs_alt"].write(create_dji_altitude_widget(
                    "ABSOLUTE ALTITUDE", value, size, widths["abs_alt"],
                ))
            if "camera" in writers:
                writers["camera"].write(create_dji_camera_widget(
                    sample, size, widths["camera"],
                ))
            if "datetime" in writers:
                dt = sample["recording_datetime"]
                value = "----" if dt is None else dt.strftime("%d/%m/%Y %H:%M:%S")
                writers["datetime"].write(create_single_widget(
                    "", value, "", size, widths["datetime"],
                ))
            if "track" in writers:
                track_source = (
                    [item for item in filtered if item["latitude"] is not None and item["longitude"] is not None]
                    if args.track_mode == "window" else gps_samples
                )
                lats = [item["latitude"] for item in track_source]
                lons = [item["longitude"] for item in track_source]
                track_idx = (
                    min(range(len(track_source)), key=lambda i: abs(track_source[i]["timestamp"] - sample["timestamp"]))
                    if track_source else -1
                )
                writers["track"].write(create_track_widget(lats, lons, track_idx, size))
            if "pitch" in writers:
                current_time = sample["timestamp"]
                log_idx = bisect.bisect_left(log_times, current_time)
                if log_idx == 0:
                    best_idx = 0
                elif log_idx > max_log_idx:
                    best_idx = max_log_idx
                else:
                    before, after = log_times[log_idx - 1], log_times[log_idx]
                    best_idx = log_idx - 1 if abs(current_time - before) <= abs(current_time - after) else log_idx
                pitch = rows[best_idx]["pitch"]
                pitch_text = "----" if pitch is None else f"{pitch:.1f}"
                pitch_color = (70, 220, 100, 255) if pitch is None or pitch >= 0 else (240, 100, 90, 255)
                writers["pitch"].write(create_single_widget(
                    "CAMERA PITCH", pitch_text, "°", size, widths["pitch"], pitch_color,
                ))

        for writer in writers.values():
            writer.close()
        writers.clear()
    finally:
        for writer in writers.values():
            if writer.process.poll() is None:
                writer.process.terminate()
                writer.process.wait()

    print(f"Esportazione DJI SRT completata: {len(filtered)} frame @ {fps:.3f} FPS")
    for widget in selected_widgets:
        print(f"  - {args.output_dir / (widget + '.webm')}")


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
        description="Genera overlay WebM trasparenti da telemetria Avinox (.proto) o DJI (.srt)"
    )
    parser.add_argument("recording", nargs="?", type=Path, help="File telemetria .proto Avinox oppure .srt DJI")
    parser.add_argument("--output-dir", type=Path, default=Path("output_widgets"),
                        help="Cartella dei WebM generati")
    parser.add_argument("--list-widgets", action="store_true",
                        help="Mostra la lista dei widget disponibili ed esce")
    parser.add_argument(
        "--widgets", type=str,
        default=None,
        help="Widget da generare separati da virgola. Se omesso, usa i default della sorgente."
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
    parser.add_argument(
        "--video", type=Path, default=None,
        help="File video, only for the pitch widget"
    )
    args = parser.parse_args()

    if args.list_widgets:
        print("\n📋 Widget disponibili:")
        print("\nAvinox (.proto):")
        for name, desc in AVAILABLE_WIDGETS.items():
            print(f"  - {name: <12} : {desc}")
        print("\nDJI (.srt):")
        for name, desc in DJI_WIDGETS.items():
            print(f"  - {name: <12} : {desc}")
        print("\nEsempi: python tools/overlay_widgets.py ride.proto --widgets power,speed")
        print("        python tools/overlay_widgets.py drone.srt --widgets rel_alt,abs_alt,camera,track\n")
        return

    if not args.recording:
        parser.error("Specifica un file .proto Avinox o .srt DJI (oppure usa --list-widgets).")
    if not args.recording.is_file():
        parser.error(f"File di telemetria non trovato: {args.recording}")
    if args.size <= 0:
        parser.error("--size deve essere maggiore di zero.")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps deve essere maggiore di zero.")
    if not 0 <= args.crf <= 63:
        parser.error("--crf deve essere compreso tra 0 e 63.")
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg non trovato. Installa FFmpeg e assicurati che sia nel PATH.")

    source_suffix = args.recording.suffix.lower()
    if source_suffix == ".srt":
        widget_registry = DJI_WIDGETS
        default_widgets = "rel_alt,abs_alt,camera,track,datetime"
    elif source_suffix == ".proto":
        widget_registry = AVAILABLE_WIDGETS
        default_widgets = "speed,slope,pitch,alt,track,elevation,power,datetime"
    else:
        parser.error("Formato non supportato: usa un file .proto Avinox o .srt DJI.")

    requested_widgets = args.widgets if args.widgets is not None else default_widgets
    selected_widgets = list(dict.fromkeys(
        widget.strip().lower() for widget in requested_widgets.split(",") if widget.strip()
    ))
    for widget in selected_widgets:
        if widget not in widget_registry:
            parser.error(
                f"Widget '{widget}' non disponibile per la sorgente {source_suffix}. "
                "Usa --list-widgets per vedere quelli disponibili."
            )
    if not selected_widgets:
        parser.error("Seleziona almeno un widget con --widgets.")

    if source_suffix == ".srt":
        run_dji_widgets(args, parser, selected_widgets)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    decoded = parse(args.recording)

    if "pitch" in selected_widgets:
        if args.video is None or not args.video.is_file():
            parser.error(f"File not found: {args.video}")
        print("Processing IMU data from MP4...")
        samples,_ = extract_clip(args.video)
        print(f"Extracted {len(samples)} samples.")
        rows,_=process(samples, "quat", 1.0, True,"auto",())
        log_times = [r["sample"]["t"] for r in rows]
        max_log_idx = len(rows) - 1
        print(f"Processed {len(samples)} samples.")

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

    widget_widths = {
        "speed": int(args.size * 2.36),
        "slope": int(args.size * 2.36),
        "alt": int(args.size * 2.36),
        "pitch": int(args.size * 2.36),
        "track": int(args.size * 1.55),
        "elevation": int(args.size * 2.2),
        "gear": args.size,
        "power": int(args.size * 1.5),
        "datetime": int(args.size * 4.72),
    }
    
    writers: dict[str, WebMWriter] = {}
    try:
        # Start one encoder per widget: each resulting WebM remains independently
        # positionable in the downstream video editor.
        for widget in selected_widgets:
            width = widget_widths[widget]
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
                    create_single_widget("VELOCITY", speed_text, "km/h", height=args.size, width=widget_widths["speed"],)
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
                        "SLOPE", slope_text, "%", height=args.size, width=widget_widths["slope"], value_color=slope_color
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
                            width=widget_widths["datetime"],
                            value_color=(255, 255, 255, 255)
                        )
                    )


            if "pitch" in writers:
                current_time = idx_out / fps
                log_idx = bisect.bisect_left(log_times, current_time)
                if log_idx == 0:
                    best_idx = 0
                elif log_idx > max_log_idx:
                    best_idx = max_log_idx
                else:
                    t_before = log_times[log_idx - 1]
                    t_after = log_times[log_idx]
                    if abs(current_time - t_before) <= abs(current_time - t_after):
                        best_idx = log_idx - 1
                    else:
                        best_idx = log_idx
                pitch = rows[best_idx]["pitch"]
                pitch_text = f"{pitch:.1f}"
                pitch_color = (70, 220, 100, 255) if pitch >= 0 else (240, 100, 90, 255)
                writers["pitch"].write(
                    create_single_widget(
                        "CAMERA PITCH", pitch_text, "°", height=args.size, width=widget_widths["pitch"], value_color=pitch_color
                    )
                )


            if "alt" in writers:
                raw_altitude = sample.get("altitude", 0.0)
                altitude = float(raw_altitude) if raw_altitude is not None else None
                alt_text = "----" if altitude is None else f"{int(altitude)}"
                writers["alt"].write(
                    create_single_widget("ALTITUDE", alt_text, "m", height=args.size, width=widget_widths["alt"],)
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
