# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# ---------------------------------------------------------------------------
# Optical flow
# ---------------------------------------------------------------------------

def calculate_flow_features(
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    flow_width: int,
    flow_height_percent: float,
):
    """
    Calculate dense optical flow on the upper portion of the frame.

    Returns:
        flow
        feature dictionary
        resized flow-analysis frame
    """

    source_height, source_width = prev_frame.shape[:2]

    flow_height_percent = max(
        1.0,
        min(100.0, flow_height_percent),
    )

    analysis_height = max(
        1,
        int(
            source_height *
            flow_height_percent /
            100.0
        ),
    )

    scale = flow_width / float(source_width)

    analysis_height_scaled = max(
        1,
        int(round(analysis_height * scale)),
    )

    prev_crop = prev_frame[
        :analysis_height,
        :,
    ]

    current_crop = current_frame[
        :analysis_height,
        :,
    ]

    prev_small = cv2.resize(
        prev_crop,
        (
            flow_width,
            analysis_height_scaled,
        ),
        interpolation=cv2.INTER_AREA,
    )

    current_small = cv2.resize(
        current_crop,
        (
            flow_width,
            analysis_height_scaled,
        ),
        interpolation=cv2.INTER_AREA,
    )

    prev_gray = cv2.cvtColor(
        prev_small,
        cv2.COLOR_BGR2GRAY,
    )

    current_gray = cv2.cvtColor(
        current_small,
        cv2.COLOR_BGR2GRAY,
    )

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        current_gray,
        None,
        **FARNEBACK_PARAMS,
    )

    u = flow[..., 0]
    v = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(
        u,
        v,
        angleInDegrees=True,
    )

    # ---------------------------------------------------------------
    # Global flow statistics
    # ---------------------------------------------------------------

    flow_mean = float(np.mean(magnitude))
    flow_p90 = float(np.percentile(magnitude, 90))
    flow_std = float(np.std(magnitude))

    flow_x = float(np.mean(u))
    flow_y = float(np.mean(v))

    flow_coherence = math.hypot(flow_x, flow_y)

    # ---------------------------------------------------------------
    # Divergence and curl
    # ---------------------------------------------------------------

    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)

    divergence = du_dx + dv_dy

    curl = dv_dx - du_dy

    div_abs = np.abs(divergence)
    curl_abs = np.abs(curl)

    div_abs_mean = float(np.mean(div_abs))
    div_abs_p90 = float(np.percentile(div_abs, 90))
    div_std = float(np.std(divergence))

    div_pos_fraction = float(
        np.mean(divergence > 0)
    )

    curl_abs_mean = float(np.mean(curl_abs))
    curl_abs_p90 = float(np.percentile(curl_abs, 90))
    curl_std = float(np.std(curl))

    # ---------------------------------------------------------------
    # Normalized Divergence & Curl (Relative to Forward Speed)
    # ---------------------------------------------------------------
    # Adding a small epsilon (1e-6) prevents Division-By-Zero when stationary
    eps = 1e-6

    # Normalize by mean flow magnitude. The result is NOT dimensionless:
    # divergence/curl have units of 1/frame, while flow_mean has units of
    # pixels/frame, so the normalized quantities have units of 1/pixel.
    # The purpose is to describe spatial expansion/rotation relative to the
    # amount of motion, so the feature is less directly tied to flow speed.
    div_normalized_mean = div_abs_mean / (flow_mean + eps)

    curl_normalized_mean = curl_abs_mean / (flow_mean + eps)

    # ---------------------------------------------------------------
    # 3x3 spatial grid
    # ---------------------------------------------------------------

    features = {
        "flow_mean": flow_mean,
        "flow_p90": flow_p90,
        "flow_std": flow_std,
        "flow_x": flow_x,
        "flow_y": flow_y,
        "flow_coherence": flow_coherence,

        "div_normalized_mean": div_normalized_mean,
        "div_abs_p90": div_abs_p90,
        "div_std": div_std,
        "div_pos_fraction": div_pos_fraction,

        "curl_normalized_mean": curl_normalized_mean,
        "curl_abs_p90": curl_abs_p90,
        "curl_std": curl_std,

    }

    height, width = magnitude.shape

    for row in range(GRID_ROWS):
        y0 = int(row * height / GRID_ROWS)
        y1 = int((row + 1) * height / GRID_ROWS)

        for col in range(GRID_COLS):
            x0 = int(col * width / GRID_COLS)
            x1 = int((col + 1) * width / GRID_COLS)

            region_u = u[y0:y1, x0:x1]
            region_v = v[y0:y1, x0:x1]
            region_mag = magnitude[y0:y1, x0:x1]

            prefix = f"grid_{row}_{col}"

            features[f"{prefix}_x"] = float(
                np.mean(region_u)
            )

            features[f"{prefix}_y"] = float(
                np.mean(region_v)
            )

            features[f"{prefix}_abs_x"] = float(
                np.mean(np.abs(region_u))
            )

            features[f"{prefix}_abs_y"] = float(
                np.mean(np.abs(region_v))
            )

            features[f"{prefix}_mag"] = float(
                np.mean(region_mag)
            )

            features[f"{prefix}_p90"] = float(
                np.percentile(region_mag, 90)
            )

    return (
        flow,
        features,
        current_small,
    )

