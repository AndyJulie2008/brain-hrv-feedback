# Muse 2 + Polar H10 Meditation and Arousal Biofeedback System

This project provides a real-time EEG-based and HRV-based biofeedback system built around the [Muse 2](https://choosemuse.com/products/muse-2) headset and Polar H10 (ASIN: B07PM54P4N on Amazon). It leverages Python and OSC (Open Sound Control) via the Mind Monitor app to track brainwave activity, helping users enter and sustain meditative or alert states through visual and auditory feedback. The Polar H10 supplies HRV metrics—specifically RMSSD (Root Mean Square of Successive Differences), which serves as a proxy for parasympathetic activity, reflecting vagus nerve engagement.

Arousal detection in this system is based on a decrease in the theta/beta ratio, reflecting increased high-frequency (beta) activity and reduced low-frequency (theta) activity, typically associated with alertness or stress.

Meditation detection in this system is based on an increase in the theta/alpha ratio, reflecting stronger low-frequency (theta) activity relative to alpha. This pattern is often associated with relaxed awareness, internal focus, and meditative states.

Thresholds of 30 ms and 50 ms in RMSSD are used as criteria to delineate three physiological states: values below 30 ms indicate a sympathetic‑dominant state (high arousal, stress, or anxiety) ("🔴 Anxious"), between 30–50 ms suggest a neutral/baseline state ("🟡 Neutral"), and above 50 ms reflect parasympathetic dominance, linked to relaxation and recovery ("🟢 Relaxed").
## ✨ Features

- Real-time EEG signal acquisition via Mind Monitor
- Real-time HRV tracking via Polar H10, with RMSSD plotted beneath brainwave graphs to visualize vagus nerve–linked parasympathetic activity
- Dynamic theta/alpha (θ/α) or theta/beta (θ/β) ratio-based feedback
- Visual biofeedback using a growing/shrinking circle, live EEG curves, and state transitions
- Built-in calibration and burn-in phases for robust baseline estimation
- Intuitive GUI interface powered by PyQtGraph and Pygame
- Multi-phase cognitive state detection:
  - Configuration → Burn-in → Calibration → Idle/Neutral → Transition → Meditation or Arousal

## 🧠 Cognitive State Flow

The system supports the following six key states:

| State | Screenshot | Comment |
|-------|------------|---------|
| 0. Configuration | ![](asset2/0.Configuration_update.PNG) | User sets session parameters and prepares headset |
| 1. Burn-in Phase | ![](asset2/1.Burnin_update.PNG) | User waits while the system stabilizes before calibration begins |
| 2. Calibration | ![](asset2/2.Calibration_update.PNG) | System calibrates thresholds for meditation and arousal |
| 3. Idle/Neutral | ![](asset2/3.Neutral_update.PNG) | No dominant mental state detected yet (neutral monitoring) |
| 4. Transition to Meditation | ![](asset2/4.Transit2meditation.PNG) | Emerging meditative EEG features (increased theta/alpha) cause the left feedback circle to contract. 
| 5. Meditation | ![](asset2/5.Meditation.PNG) | Stable meditative state detected (e.g., high theta/alpha) |
| 6. Transition to Arousal | ![](asset2/6.Transit2arousal.PNG) | Emerging arousal-related EEG features (decreased theta/beta) cause the right feedback circle to expand. |
| 7. Arousal | ![](asset2/7.Arousal.PNG) | Elevated arousal state detected (low theta/beta) |
| 8. Visualization | ![](asset2/8.Data_visualization.png) | Three‑panel display: (top) comparison of EEG brainwave bands, (middle) arousal vs. meditation, and (bottom) heart rate vs. RMSSD trends. |
> These screenshots illustrate the adaptive flow of the system as it dynamically tracks and responds to mental state shifts between relaxation and heightened arousal.

---

## How to Run

Follow these steps to set up and launch the system:

### 1. Create a Conda environment
conda create --name muse2biofeedback python=3.12.4 pip

### 2. Activate the created Conda environment
conda activate muse2biofeedback

### 3. Install required dependencies
pip install -r requirements.txt

### 4. Start the Program
**Once the environment is set up and all dependencies are installed, you can launch the system**

python muse2_polarH10_arousal_meditatation_biofeedback.py

### 5.Visualization
**After recording the data, you can visualize the results**

python muse2_polarH10_read.py


