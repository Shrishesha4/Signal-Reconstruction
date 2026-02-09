import numpy as np
import pandas as pd

def generate_safe_signal(
    n_points=15,
    x_start=0.0,
    x_end=10.0,
    noise_level=0.02,
    seed=42
):
    """
    Generates realistic, smooth signal data safe for:
    - Lagrange interpolation
    - Polynomial interpolation
    - Cubic spline
    - Linear interpolation
    """

    np.random.seed(seed)

    # Uniformly spaced x (important for Lagrange stability)
    x = np.linspace(x_start, x_end, n_points)

    # Base smooth signal (low-frequency, real-world-like)
    y_clean = (
        0.6 * np.sin(0.6 * x) +
        0.3 * np.cos(0.3 * x) +
        0.1 * x
    )

    # Small Gaussian noise (sensor-like)
    noise = np.random.normal(0, noise_level, size=n_points)
    y = y_clean + noise

    # Normalize amplitude (optional but safe)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    return pd.DataFrame({
        "x": np.round(x, 4),
        "y": np.round(y, 6)
    })


if __name__ == "__main__":
    df = generate_safe_signal(
        n_points=2000,      # <= 15 is ideal for Lagrange
        noise_level=0.015
    )

    # Save for frontend / CSV upload
    df.to_csv("safe_signal_data.csv", index=False)

    print("Generated safe signal data:")
    print(df)
