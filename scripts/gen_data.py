import numpy as np
import pandas as pd

def generate_realistic_signal(
    num_samples=1000,
    duration=10.0,
    noise_level=0.05,
    drift_strength=0.1,
    seed=42
):
    """
    Generates a realistic signal combining:
    - Base sinusoidal components
    - Slow drift (real-world sensor behavior)
    - Random noise
    """

    np.random.seed(seed)

    # Time axis
    t = np.linspace(0, duration, num_samples)

    # Base signal (multi-frequency)
    signal = (
        0.6 * np.sin(2 * np.pi * 0.5 * t) +
        0.3 * np.sin(2 * np.pi * 1.5 * t + 0.5) +
        0.2 * np.sin(2 * np.pi * 3.0 * t)
    )

    # Slow drift (e.g., temperature / sensor bias)
    drift = drift_strength * np.sin(2 * np.pi * 0.05 * t)

    # Noise (measurement noise)
    noise = noise_level * np.random.normal(size=num_samples)

    # Final signal
    y = signal + drift + noise

    return t, y


if __name__ == "__main__":
    # Generate signal
    t, y = generate_realistic_signal(
        num_samples=2000,
        duration=20,
        noise_level=0.07,
        drift_strength=0.15
    )

    # Save to CSV
    df = pd.DataFrame({
        "x": t,
        "y": y
    })

    df.to_csv("signal_data.csv", index=False)

    print("✅ Realistic signal data generated: signal_data.csv")
    print(f"Samples: {len(df)}")
