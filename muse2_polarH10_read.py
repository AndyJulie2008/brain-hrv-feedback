import pandas as pd
import matplotlib.pyplot as plt

filename = "C:/RESEARCH_YW/EEG_muse/muse2_polarH10_recording_20250726_174646.csv"

plt.rcParams.update({
    "font.size": 16,          # default font size for text
    "axes.labelsize": 16,     # x/y label font size
    "axes.titlesize": 16,     # title font size
    "xtick.labelsize": 14,    # x-axis tick font size
    "ytick.labelsize": 14,    # y-axis tick font size
    "legend.fontsize": 12     # legend font size
})


def dB_to_raw_amplitude(val):
    return 10 ** (val / 10)

# size of sliding window for smoothing brain waves
window_size = 256 

# 1. Read file
df_orig = pd.read_csv(filename)
df = df_orig[df_orig['Status'] == 'Test']


# 2. color setting
valid_channels = ['af7', 'af8'] #,'tp9','tp10']
bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
colors = {
    'Delta': 'r',   # red
    'Theta': 'm',   # magenta
    'Alpha': 'c',   # cyan
    'Beta':  'g',   # green
    'Gamma': 'y'    # yellow
}

# 3. Compute average EEG using sliding window
smooth_band_avgs = {}
raw_band_avgs = {}
for band in bands:
    band_cols = [col for col in df.columns if any(col.startswith(f"{band}_{ch}") for ch in valid_channels)]

    raw_band_avgs[band] = df[band_cols].mean(axis=1)
    smooth_band_avgs[band] = raw_band_avgs[band] .rolling(window=window_size, min_periods=1).mean()  # average smoothing

raw_band_avgs['meditation'] = raw_band_avgs['Theta'] - raw_band_avgs['Alpha']
smooth_band_avgs['meditation'] =  raw_band_avgs['meditation'] .rolling(window=window_size, min_periods=1).mean()

raw_band_avgs['arousal'] = raw_band_avgs['Beta'] - raw_band_avgs['Theta'] 
smooth_band_avgs['arousal'] =  raw_band_avgs['arousal'] .rolling(window=window_size, min_periods=1).mean()

# 4. Convert second to minute
df['Time_min'] = (df['Time'] - df['Time'].iloc[0]) / 60
x_min = df['Time_min'].iloc[0] 
x_max = df['Time_min'].iloc[-1]   

# 5. Draw

#####################################
# Muse 2 (delta, theta, alpha, beta, gamma)
#####################################
plt.figure(figsize=(12, 6))
plt.subplot(3,1,1)
for i,band in enumerate(bands):
    # plt.subplot(5,1,i+1)
    plt.plot(df['Time_min'], smooth_band_avgs[band], color=colors[band], label=band)
    plt.xlabel("Time (minutes)")
    plt.ylabel("EEG Amplitude (dB)")
    plt.title("EEG Band Power Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
plt.xlim(x_min, x_max)
plt.legend(loc='upper left')

# --- subplot(3,1,2) ---
ax1 = plt.subplot(3,1,2)

# Left y-axis: Meditation (Theta/Alpha)
line1, = ax1.plot(df['Time_min'],
                  dB_to_raw_amplitude(smooth_band_avgs['meditation']),
                  color='red', label='Meditation (Theta/Alpha)')
ax1.set_ylabel("Meditation (Theta/Alpha)", color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Right y-axis: Arousal (Theta/Beta)
ax2 = ax1.twinx()
line2, = ax2.plot(df['Time_min'],
                  dB_to_raw_amplitude(smooth_band_avgs['arousal']),
                  color='blue', label='Arousal (Theta/Beta)')
ax2.set_ylabel("Arousal (Theta/Beta)", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Combine legends (from both axes)
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')

# Common title & grid
ax1.set_xlabel("Time (minutes)")
ax1.set_title("Meditation & Arousal Feedback")
ax1.grid(True)



#####################################
# Polar H10 (HR and RMSSD)
#####################################
# ======= Combined HR & RMSSD (dual y-axis) =======
ax3 = plt.subplot(3,1,3)

# Left y-axis: HR
line3, = ax3.plot(df['Time_min'], df['heart_rate'], color='blue', label='Heart Rate (bpm)')
ax3.axhline(y=110, color='red', linestyle='-.', linewidth=1.5)
ax3.axhline(y=90, color='blue', linestyle=':', linewidth=1.5)
ax3.axhline(y=70, color='blue', linestyle=':', linewidth=1.5)
ax3.axhline(y=50, color='red', linestyle='-.', linewidth=1.5)

ax3.set_yticks([50, 70, 90, 110])
ax3.set_ylabel("Heart Rate (bpm)", color='blue')
ax3.tick_params(axis='y', labelcolor='blue')

# Right y-axis: RMSSD
ax4 = ax3.twinx()
line4, = ax4.plot(df['Time_min'], df['RMSSD'], color='orange', label='RMSSD (ms)')
ax4.axhline(y=10, color='red', linestyle='-.', linewidth=1.5)
ax4.axhline(y=30, color='orange', linestyle='--', linewidth=1.5)
ax4.axhline(y=50, color='orange', linestyle='--', linewidth=1.5)
ax4.axhline(y=70, color='orange', linestyle='--', linewidth=1.5)
ax4.axhline(y=90, color='red', linestyle='-.', linewidth=1.5)

ax4.set_yticks([10, 30, 50, 70, 90])
ax4.set_ylabel("RMSSD (ms)", color='orange')
ax4.tick_params(axis='y', labelcolor='orange')

# Combine legends
# Combine legends (from both axes)
lines = [line3, line4]
labels = [line.get_label() for line in lines]
ax3.legend(lines, labels, loc='upper left')

plt.title("Heart rate (HR) & RMS of the Successive Differences (RMSSD)")
plt.tight_layout()
ax1.set_xlim(x_min, x_max)
ax3.set_xlim(x_min, x_max)

plt.show()