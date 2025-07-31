import sys
import csv
import time
import numpy as np
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pythonosc import dispatcher, osc_server
import threading
from datetime import datetime
from PyQt5.QtWidgets import QGraphicsEllipseItem 
import os
import pygame
from bleak import BleakScanner, BleakClient
import argparse

#####################################################################
# PolarH10
import asyncio
from bleak import BleakClient
import statistics

from qasync import QEventLoop

# UUID definitions
BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"  # Battery level
HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"       # Heart rate data
BODY_LOC_UUID = "00002a38-0000-1000-8000-00805f9b34fb" # Heart rate strap location
# Your Polar H10 address
H10_ADDRESS = "24:AC:AC:02:C7:5B"  
# Your Polar H10 Bluetooth address (fixed)
H10_ADDRESS = "24:AC:AC:02:C7:5B"
# Polar H10 Heart Rate Service UUID
HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# RR interval buffer (seconds)
rr_intervals = []
#####################################################################

class ParameterDialog(QtWidgets.QDialog):
    def __init__(self,font_size=18):
        super().__init__()
        self.setWindowTitle("Configure Dashboard Parameters")
        self.setFixedWidth(600)

        font = QtGui.QFont()
        font.setPointSize(font_size) 
        self.setFont(font)     

        layout = QtWidgets.QFormLayout()

        self.fields = {
            "RMSSD Display Window (sec)":(300, QtWidgets.QLineEdit()),
            "RMSSD maximum value": (50, QtWidgets.QLineEdit()),
            "Display Window (sec)": (20, QtWidgets.QLineEdit()),
            "Update Interval (ms)": (50, QtWidgets.QLineEdit()),
            "Arousal Duration Threshold": (10.0, QtWidgets.QLineEdit()),
            "Meditation Duration Threshold": (10.0, QtWidgets.QLineEdit()),
            "Sampling Duration (sec)": (30, QtWidgets.QLineEdit()),
            "Burn-in Period (sec)": (5, QtWidgets.QLineEdit()),
            "Smoothing Factor": (0.02, QtWidgets.QLineEdit()),
            "Arousal Threshold Factor": (2.0, QtWidgets.QLineEdit()),
            "Meditation Threshold Factor": (0.6, QtWidgets.QLineEdit()),
        }

        for label, (default, line_edit) in self.fields.items():
            line_edit.setText(str(default))
            line_edit.setFont(font)
            layout.addRow(QtWidgets.QLabel(label), line_edit)

        self.ok_button = QtWidgets.QPushButton("Start Dashboard")
        self.ok_button.setFont(font)
        self.ok_button.clicked.connect(self.accept)

        layout.addRow(self.ok_button)
        self.setLayout(layout)

    def get_parameters(self):
        def get_float(key):
            return float(self.fields[key][1].text())

        def get_int(key):
            return int(float(self.fields[key][1].text()))

        return {
            "display_window_sec": get_int("Display Window (sec)"),
            "update_interval_ms": get_int("Update Interval (ms)"),
            "arousal_duration_threshold": get_float("Arousal Duration Threshold"),
            "meditation_duration_threshold": get_float("Meditation Duration Threshold"),
            "sampling_duration": get_int("Sampling Duration (sec)"),
            "burn_in_period": get_int("Burn-in Period (sec)"),
            "smooth_factor": get_float("Smoothing Factor"),
            "threshold_factor_arousal": get_float("Arousal Threshold Factor"),
            "threshold_factor_meditat": get_float("Meditation Threshold Factor"),
            "rmssd_display_window_sec": get_float("RMSSD Display Window (sec)"),
            "rmssd_max":get_float('RMSSD maximum value')
        }



pygame.mixer.init()
music_dir = "C:/RESEARCH_YW/EEG_muse/stimuli/"
music_list = ["Hans Zimmer_time.mp3", "1_hrRain_Thunderstorm.mp3", "Relaxing_Ocean_Wave.mp3", "Interstellar.mp3","Destroyer_Of_Worlds.mp3","White_Noise_1min.mp3","Bandari_Silence_With_Sound_From_Nature.mp3", "Night_OCEAN_WAVES_3hrs.mp3"]
bg_music = pygame.mixer.Sound(os.path.join(music_dir,music_list[4])) # Reward for arousal
white_noise = pygame.mixer.Sound(os.path.join(music_dir,music_list[1])) # Noise
reward_music = pygame.mixer.Sound(os.path.join(music_dir,music_list[0])) # Reward for meditation
bg_channel = pygame.mixer.Channel(0)
noise_channel = pygame.mixer.Channel(1)

def play_music(music_file):
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(music_file,loop=-1)  # -1 denote endless loop

def stop_music():
    pygame.mixer.music.stop()


# === configuration ===
noise_gain = 1 #it should be 1 since volume range is [0 1]
music_vol = 1
channels = ['tp9', 'af7', 'af8', 'tp10']
fnirs_channels = ['af7', 'af8']

# Data input from Mind Monitor
alpha_buffers = {ch: deque(maxlen=500) for ch in channels}
theta_buffers = {ch: deque(maxlen=500) for ch in channels}
beta_buffers  = {ch: deque(maxlen=500) for ch in channels}
delta_buffers = {ch: deque(maxlen=500) for ch in channels}
gamma_buffers = {ch: deque(maxlen=500) for ch in channels}

# Timestamp
alpha_times = {ch: deque(maxlen=500) for ch in channels}
theta_times = {ch: deque(maxlen=500) for ch in channels}
beta_times  = {ch: deque(maxlen=500) for ch in channels}
delta_times = {ch: deque(maxlen=500) for ch in channels}
gamma_times = {ch: deque(maxlen=500) for ch in channels}

# average EEG and timestamp
avg_alpha_buffer = deque(maxlen=500)
avg_theta_buffer = deque(maxlen=500)
avg_beta_buffer  = deque(maxlen=500)
avg_delta_buffer = deque(maxlen=500)
avg_gamma_buffer = deque(maxlen=500)

avg_alpha_time_buffer = deque(maxlen=500)
avg_theta_time_buffer = deque(maxlen=500)
avg_beta_time_buffer  = deque(maxlen=500)
avg_delta_time_buffer = deque(maxlen=500)
avg_gamma_time_buffer = deque(maxlen=500)

hbo_buffers = {ch: deque(maxlen=500) for ch in fnirs_channels}
hbr_buffers = {ch: deque(maxlen=500) for ch in fnirs_channels}
hbo_time_buffer = {ch: deque(maxlen=500) for ch in channels}
hbr_time_buffer = {ch: deque(maxlen=500) for ch in channels}

gyro_data = [0.0, 0.0, 0.0]
acc_data = [0.0, 0.0, 0.0]
concentration = 0.0
mellow = 0.0
blink_timestamp = 0.0
jaw_timestamp = 0.0
BLINK_HOLD_SEC = 2
JAW_HOLD_SEC = 2
blink = False
jaw = False
lock = threading.Lock()
ratio = 0
avg_delta = 0
avg_theta = 0
avg_alpha = 0
avg_beta = 0
avg_gamma = 0
recording = False
csv_writer = None
csv_file = None

# === Helper ===
def trim_recent(timestamps, values, window_sec):
    now = time.time()
    x = np.array(timestamps)
    y = np.array(values)
    mask = x >= (now - window_sec)
    return x[mask] - now, y[mask]

# === Handlers ===
def alpha_handler(addr, tp9, af7, af8, tp10):
    now = time.time()
    tp9, af7, af8, tp10 = [10**(v / 10) for v in (tp9, af7, af8, tp10)]
    with lock:
        for ch, val in zip(channels, [tp9, af7, af8, tp10]):
            alpha_buffers[ch].append(val)
            alpha_times[ch].append(now)
            # print(f"{ch}: {val:.3f}")

        valid_alpha = [v for v in [af7, af8] ]

        avg_alpha = sum(valid_alpha) / len(valid_alpha) if valid_alpha else 0.0

        avg_alpha_buffer.append(avg_alpha)
        avg_alpha_time_buffer.append(now)

def theta_handler(addr, tp9, af7, af8, tp10):
    now = time.time()
    tp9, af7, af8, tp10 = [10**(v / 10) for v in (tp9, af7, af8, tp10)]
    with lock:
        for ch, val in zip(channels, [tp9, af7, af8, tp10]):
            theta_buffers[ch].append(val)
            theta_times[ch].append(now)
            # print(f"{ch}: {val:.3f}")

        valid_theta = [v for v in [af7, af8] ]

        avg_theta = sum(valid_theta) / len(valid_theta) if valid_theta else 0.0

        avg_theta_buffer.append(avg_theta)
        avg_theta_time_buffer.append(now)

def beta_handler(addr, tp9, af7, af8, tp10):
    now = time.time()
    tp9, af7, af8, tp10 = [10**(v / 10) for v in (tp9, af7, af8, tp10)]
    with lock:
        for ch, val in zip(channels, [tp9, af7, af8, tp10]):
            beta_buffers[ch].append(val)
            beta_times[ch].append(now)
            # print(f"{ch}: {val:.3f}")

        valid_beta = [v for v in [af7, af8] ] 

        avg_beta = sum(valid_beta) / len(valid_beta) if valid_beta else 0.0

        avg_beta_buffer.append(avg_beta)
        avg_beta_time_buffer.append(now)

def delta_handler(addr, tp9, af7, af8, tp10):
    now = time.time()
    tp9, af7, af8, tp10 = [10**(v / 10) for v in (tp9, af7, af8, tp10)]
    with lock:  
        for ch, val in zip(channels, [tp9, af7, af8, tp10]):
            delta_buffers[ch].append(val)
            delta_times[ch].append(now)
            # print(f"{ch}: {val:.3f}")

        valid_delta = [v for v in [af7, af8] ]

        avg_delta = sum(valid_delta) / len(valid_delta) if valid_delta else 0.0

        avg_delta_buffer.append(avg_delta)
        avg_delta_time_buffer.append(now)

def gamma_handler(addr, tp9, af7, af8, tp10):
    now = time.time()
    tp9, af7, af8, tp10 = [10**(v / 10) for v in (tp9, af7, af8, tp10)]
    with lock:
        for ch, val in zip(channels, [tp9, af7, af8, tp10]):
            gamma_buffers[ch].append(val)
            gamma_times[ch].append(now)
            # print(f"{ch}: {val:.3f}")

        valid_gamma = [v for v in [af7, af8] ]

        avg_gamma = sum(valid_gamma) / len(valid_gamma) if valid_gamma else 0.0

        avg_gamma_buffer.append(avg_gamma)
        avg_gamma_time_buffer.append(now)

def optics_handler(addr, *args):
    now = time.time()
    with lock:
        hbo_buffers['af7'].append((args[0] + args[1]) / 2)
        hbr_buffers['af7'].append((args[2] + args[3]) / 2)
        hbo_buffers['af8'].append((args[4] + args[5]) / 2)
        hbr_buffers['af8'].append((args[6] + args[7]) / 2)
        for ch in fnirs_channels:
            hbo_time_buffer[ch].append(now)
            hbr_time_buffer[ch].append(now)

def gyro_handler(addr, x, y, z):
    with lock:
        gyro_data[0], gyro_data[1], gyro_data[2] = x, y, z

def acc_handler(addr, x, y, z):
    with lock:
        acc_data[0], acc_data[1], acc_data[2] = x, y, z
        
def concentration_handler(addr, value):
    global concentration
    with lock:
        concentration = value

def mellow_handler(addr, value):
    global mellow
    with lock:
        mellow = value

def blink_handler(addr, val):
    global blink_timestamp
    global blink
    with lock:
        blink = bool(val)
        blink_timestamp = time.time()
        print(f'blink={blink}')

def jaw_handler(addr, val):
    global jaw_timestamp
    global jaw
    with lock:
        jaw = bool(val)
        jaw_timestamp = time.time()
        print(f'jaw={jaw}')

def z_score(array):
    if len(array) < 2:
        return np.zeros(len(array))
    mean = np.mean(array)
    std = np.std(array)
    std = std if std > 1e-6 else 1e-6
    return (array - mean) / std

def dB_to_raw_amplitude(val):
    return 10 ** (val / 10)

# === GUI ===
class MuseDashboard(QtWidgets.QMainWindow):
    def __init__(self,
                 display_window_sec=20,
                 update_interval_ms=100,
                 arousal_duration_threshold=10.0,
                 meditation_duration_threshold=10.0,
                 sampling_duration=30,
                 burn_in_period=5,
                 smooth_factor=0.02,
                 threshold_factor_arousal=2.0,
                 threshold_factor_meditat=0.6,
                 rmssd_display_window_sec = 300,
                 rmssd_max = 50):

        super().__init__()

        self.display_window_sec = display_window_sec
        self.update_interval_ms = update_interval_ms
        self.arousal_duration_threshold = arousal_duration_threshold
        self.meditation_duration_threshold = meditation_duration_threshold
        self.sampling_duration = sampling_duration
        self.burn_in_period = burn_in_period
        self.smooth_factor = smooth_factor
        self.threshold_factor_arousal = threshold_factor_arousal
        self.threshold_factor_meditat = threshold_factor_meditat
        self.rmssd_display_window_sec = rmssd_display_window_sec
        self.rmssd_max = rmssd_max
        self.setWindowTitle("Muse 2 Dashboard")
        self.resize(1400, 900)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        self.plotWidget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plotWidget, 4)

        self.infoPanel = QtWidgets.QVBoxLayout()
        layout.addLayout(self.infoPanel, 1)

        # Biofeedback
        self.meditat_history = deque(maxlen=50)
        self.arousal_history = deque(maxlen=50)
        self.current_volume = 0.0        
        self.target_volume = 0.0         
        self.volume_smoothing = 0.05 
        
        self.reward_mode_meditation = False
        self.meditation_start_time = None
        self.arousal_start_time = None

        self.reward_mode_arousal = False

        self.circle_meditat_threshold = 0.2 # default
        self.circle_arousal_threshold = 0.2 # default
        self.r_th_meditat = 0 
        self.r_th_arousal = 0

        # Initialize this parameter for adding a block to determine threshold (not hard code!)
        self.sampling_start_time = time.time()

        self.acc_meditat_time = 0 # total time for meditation
        self.acc_arousal_time = 0 # total time for meditation

        # === First row (Average EEG) ===
        self.eeg_avg_plot = self.plotWidget.addPlot(title="Average EEG", row=0, col=0, colspan=2)
        self.eeg_avg_plot.setXRange(-self.display_window_sec, 0)
        self.eeg_avg_plot.addLegend()
        self.eeg_avg_plot.setLabel('bottom', 'Time (s)', **{'color': '#FFFFFF', 'font-size': '12pt'})
        self.eeg_avg_plot.setLabel('left', 'Relative Power (a.u.)', **{'color': '#FFFFFF', 'font-size': '12pt'})
        self.avg_delta_curve = self.eeg_avg_plot.plot(pen='r', name='Delta')
        self.avg_theta_curve = self.eeg_avg_plot.plot(pen='m', name='Theta')
        self.avg_alpha_curve = self.eeg_avg_plot.plot(pen='c', name='Alpha')
        self.avg_beta_curve = self.eeg_avg_plot.plot(pen='g', name='Beta')
        self.avg_gamma_curve = self.eeg_avg_plot.plot(pen='y', name='Gamma')
        self.eeg_avg_plot.showGrid(x=True, y=True, alpha=0.3)

        # Add grid lines
        for sec in range(int(-self.display_window_sec), 1):  # e.g. -20 to 0
            vline = pg.InfiniteLine(pos=sec, angle=90,
                                    pen=pg.mkPen(color=(180, 180, 180), style=QtCore.Qt.DotLine))
            self.eeg_avg_plot.addItem(vline)

        # === Second row (RMSSD) ===
        self.current_radius_rmssd = 0.5
        self.hr_value = 0
        self.rmssd_value = 0.0  # default RMSSD value
        self.rmssd_plot = self.plotWidget.addPlot(title="RMSSD Trend", row=1, col=0, colspan=3)
        self.rmssd_plot.setXRange(-self.rmssd_display_window_sec, 0)
        self.rmssd_plot.setYRange(0, self.rmssd_max) # RMSSD range
        self.rmssd_plot.setLabel('bottom', 'Time (s)', **{'color': '#FFFFFF', 'font-size': '12pt'})
        self.rmssd_plot.setLabel('left', 'RMSSD (ms)', **{'color': '#FFFFFF', 'font-size': '12pt'})
        self.rmssd_plot.showGrid(x=True, y=True, alpha=0.3)
        self.rmssd_curve = self.rmssd_plot.plot(pen=pg.mkPen('b', width=2), name='RMSSD')
        line_30 = pg.InfiniteLine(pos=30, angle=0, pen=pg.mkPen(color='y', width=2, style=QtCore.Qt.DashLine))
        line_50 = pg.InfiniteLine(pos=50, angle=0, pen=pg.mkPen(color='g', width=2, style=QtCore.Qt.DashLine))
        self.rmssd_plot.addItem(line_30)
        self.rmssd_plot.addItem(line_50)
        for sec in range(int(-self.rmssd_display_window_sec), 1, 30):  # e.g. -20 to 0
            vline = pg.InfiniteLine(pos=sec, angle=90,
                                    pen=pg.mkPen(color=(180, 180, 180), style=QtCore.Qt.DotLine))
            self.rmssd_plot.addItem(vline)

        self.eeg_avg_plot.showGrid(x=True, y=True, alpha=0.3)
        # buffer
        self.rmssd_time_buffer = []
        self.rmssd_value_buffer = []


        # === Biofeedback circles ===
        # Meditation circle
        self.circle_plot = self.plotWidget.addPlot(row=2, col=0)
        self.circle_plot.setXRange(-1, 1)
        self.circle_plot.setYRange(-1, 1)
        self.circle_plot.hideAxis('bottom')
        self.circle_plot.hideAxis('left')

        self.circle_item = QGraphicsEllipseItem(-0.2, -0.2, 0.4, 0.4)
        self.circle_item.setPen(pg.mkPen('w'))
        self.circle_item.setBrush(pg.mkBrush(255, 0, 0, 180))
        self.circle_plot.addItem(self.circle_item)

        self.th_circle_item = QGraphicsEllipseItem(-0.2, -0.2, 0.4, 0.4)
        self.th_circle_item.setPen(pg.mkPen('w'))
        self.th_circle_item.setBrush(pg.mkBrush(180, 180, 180, 100))
        self.circle_plot.addItem(self.th_circle_item)
        self.meditation_title = pg.LabelItem(justify='center')
        self.meditation_title.setText("Meditation: r ∝ (α/θ)", color='w', size='30pt')
        self.plotWidget.addItem(self.meditation_title, row=3, col=0)

        # Arousal circle
        self.circle_arousal_plot = self.plotWidget.addPlot(row=2, col=1)
        self.circle_arousal_plot.setXRange(-1, 1)
        self.circle_arousal_plot.setYRange(-1, 1)
        self.circle_arousal_plot.hideAxis('bottom')
        self.circle_arousal_plot.hideAxis('left')

        self.circle_arousal_item = QGraphicsEllipseItem(-0.2, -0.2, 0.4, 0.4)
        self.circle_arousal_item.setPen(pg.mkPen('w'))
        self.circle_arousal_item.setBrush(pg.mkBrush(100, 100, 255, 180))
        self.circle_arousal_plot.addItem(self.circle_arousal_item)

        self.th_circle_arousal_item = QGraphicsEllipseItem(-0.2, -0.2, 0.4, 0.4)
        self.th_circle_arousal_item.setPen(pg.mkPen('w'))
        self.th_circle_arousal_item.setBrush(pg.mkBrush(180, 180, 180, 100))
        self.circle_arousal_plot.addItem(self.th_circle_arousal_item)
        self.arousal_title = pg.LabelItem(justify='center')
        self.arousal_title.setText("Arousal: r ∝ (β/θ)", color='w', size='30pt')
        self.plotWidget.addItem(self.arousal_title, row=3, col=1)


        

        self.labels = {}
        for key in ['Meditation','HRV', 'Arousal']:
            label = QtWidgets.QLabel(f"{key}:")
            self.infoPanel.addWidget(label)
            self.labels[key] = label

        self.btn_record = QtWidgets.QPushButton("Start Recording")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.infoPanel.addWidget(self.btn_record)

        self.btn_clear = QtWidgets.QPushButton("Clear Buffers")
        self.btn_clear.clicked.connect(self.clear_buffers)
        self.infoPanel.addWidget(self.btn_clear)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start(self.update_interval_ms)
        
        bg_channel.play(bg_music, loops=-1)
        bg_channel.set_volume(0) #mute bgm
        self.fade_in_music(bg_music, bg_channel, duration_ms=1000, target_volume=0) #music_vol #no bgm
        self.fade_in_music(white_noise, noise_channel, duration_ms=1000, target_volume=noise_gain)
        
        # measure the range of radius
        self.r_min = 10
        self.r_max = 0



    ############################# PolarH10 ##############################################
    async def connect_polar(self):
        
        print("Scan Polar H10...")
        devices = await BleakScanner.discover()
        polar_device = None
        for d in devices:
            if "Polar H10" in (d.name or ""):
                polar_device = d
                print(f"Found Polar H10: {d.address}")
                break

        if not polar_device:
            print("❌ Did not find Polar H10")
            return

        try:
            async with BleakClient(polar_device.address) as client:
                print("Connected to Polar H10")

                battery_data = await client.read_gatt_char(BATTERY_UUID)
                battery_level = int(battery_data[0])
                print(f"🔋 Battery: {battery_level}%")

                # Scan all services
                services = client.services
                print("🔍 Found services:")
                for service in services:
                    print(f"[Service] {service}")
                    for char in service.characteristics:
                        print(f"  [Characteristic] {char} ({char.uuid})")

                # Confirm HR_UUID 
                await client.start_notify(HR_UUID, self.hr_callback)
                while True:
                    await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n Cannot successfully connect Polar H10")

    def hr_callback(self, sender, data, win_len=100, min_count=2):
        """Input heart rate and RR (interval) to compute RMSSD"""
        if not hasattr(self, "rr_buffer"):
            self.rr_buffer = []
            
        flags = data[0]
        hr_16bit = flags & 0x01
        index = 1

        # 1. Parse heart rate
        if hr_16bit: # if hr > 255 (exceed 8-bit range)
            hr = int.from_bytes(data[index:index+2], byteorder='little') # 2 bits
            index += 2
            self.hr_value = hr
        else: #8 bit
            hr = data[index]
            index += 1
            self.hr_value = hr

        # 2. Parse RR intervals
        rr_list = []
        if flags & 0x10:
            while index < len(data):
                rr = int.from_bytes(data[index:index+2], byteorder='little') / 1024.0
                rr_list.append(rr)
                index += 2

        # Compute RMSSD
        rmssd = None

        if rr_list:
            self.rr_buffer.extend(rr_list)

        # Keep only the latest win_len RR intervals
        if len(self.rr_buffer) > win_len:
            self.rr_buffer = self.rr_buffer[-win_len:]

        # Keep only the latest 100 RR intervals
        is_sec = self.rr_buffer and max(self.rr_buffer) < 10 
        factor = 1000 if is_sec else 1

        if len(self.rr_buffer) > min_count:
            clean_rr = [rr for rr in self.rr_buffer if 0.3 < rr < 2.5]

            if len(clean_rr) > min_count:
                diffs = [j - i for i, j in zip(clean_rr[:-1], clean_rr[1:])]
                rmssd = (sum(x**2 for x in diffs) / len(diffs)) ** 0.5 * factor
                self.rmssd_value = rmssd
                now = time.time()
                self.rmssd_time_buffer.append(now)
                self.rmssd_value_buffer.append(rmssd)

                #  Limit the time window
                while self.rmssd_time_buffer and now - self.rmssd_time_buffer[0] > self.rmssd_display_window_sec:
                    self.rmssd_time_buffer.pop(0)
                    self.rmssd_value_buffer.pop(0)

        # 4. Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 5. Print
        if self.rmssd_value:
            print(f"⏱ {timestamp} ❤️ HR: {hr} bpm | RR: {rr_list} | RMSSD: {rmssd:.2f} ms" if rmssd else
                f"⏱ {timestamp} ❤️ HR: {hr} bpm | RR: {rr_list}")


    def rmssd_to_radius(self, rmssd, min_r=0.1, max_r=1.5):
        # Limit RMSSD range to 10–80 ms
        rmssd_clamped = max(10, min(rmssd, 80))

        # Normalize to 0–1 (higher value = more relaxed)
        norm = (rmssd_clamped - 10) / (80 - 10) 
        inverted = 1.0 - norm

        # Take square root to make differences in low RMSSD more pronounced: Use exponent to amplify
        r = min_r + (max_r - min_r) * (inverted ** 0.3)
        return r
    
    ############################# Muse 2 ##############################################

    def update_volume_smooth(self):
        # Linear interpolation
        self.current_volume += (self.target_volume - self.current_volume) * self.volume_smoothing
        self.current_volume = np.clip(self.current_volume, 0.0, 1.0)
        pygame.mixer.music.set_volume(self.current_volume)
        print(f"🎧 Volume: {self.current_volume:.2f} (→ {self.target_volume:.2f})")

    def fade_in_music(self, sound, channel, duration_ms=500, target_volume=1,steps=100000):
        self.fade_step = 0
        self.fade_volumes = np.linspace(0.0, target_volume, steps)
        self.fade_timer = QtCore.QTimer()
        self.fade_timer.setInterval(duration_ms // steps)

        def update_volume():
            if self.fade_step == 0:
                channel.set_volume(0.0)
                channel.play(sound, loops=-1)

            if self.fade_step < len(self.fade_volumes):
                vol = self.fade_volumes[self.fade_step]
                channel.set_volume(vol)
                self.fade_step += 1
            else:
                self.fade_timer.stop()

        self.fade_timer.timeout.connect(update_volume)
        self.fade_timer.start()

    def log_ratio_to_radius(self, log_ratio, max_log=1.5, baseline_r=0.5, scale=3, min_r=0.1, max_r=1.0):
        log_ratio = np.clip(log_ratio, -max_log, max_log)
        r = baseline_r - scale * (log_ratio / max_log)
        return np.clip(r, min_r, max_r)

    def radius_to_log_ratio(self, r, baseline_r=0.5, scale=2, max_log=1.5):
        log_ratio = (baseline_r - r) * (max_log / scale)
        # ratio = 10 ** (log_ratio / 10)
        return log_ratio

    def update_dashboard(self):
        global blink, blink_timestamp, jaw, jaw_timestamp
        blink_csv = False
        jaw_csv = False
        with lock:

            # Update PolarH10 RMSSD ==========================================================================
   
            if self.rmssd_value > 50:
                # self.circle_rmssd_item.setBrush(pg.mkBrush(0, 255, 0, 180))   # green (relax)
                state = "🟢 Relaxed"
                RMSSD_color = 'g' 

            elif self.rmssd_value > 30:
                # self.circle_rmssd_item.setBrush(pg.mkBrush(255, 255, 0, 180)) # yellow (neutral)
                state = "🟡 Neutral"
                RMSSD_color = 'y' 
            else:
                # self.circle_rmssd_item.setBrush(pg.mkBrush(255, 0, 0, 180))   # red (anxiety)
                state = "🔴 Anxious"
                RMSSD_color = 'r' 

            now = time.time()

            if self.rmssd_time_buffer:
                # Convert to relative time
                x_rmssd = [t - now for t in self.rmssd_time_buffer]
                y_rmssd = self.rmssd_value_buffer
                self.rmssd_curve.setData(x_rmssd, y_rmssd)
                self.rmssd_curve.setPen(pg.mkPen(color=RMSSD_color, width=2))
            self.labels['HRV'].setText(f"Heart rate: {self.hr_value} bpm  \
                                            \nRMSSD: {self.rmssd_value:.3f} ms  \
                                            \nState: → {state}")
            self.labels['HRV'].setStyleSheet("color: blue; font-size: 24px;")
            
            # Get blink and jaw information from Mind monitor ==================================================

            if blink and (time.time() - blink_timestamp > BLINK_HOLD_SEC):
                blink_csv = blink
                blink = False  # Automatically reset after exceeding the specified duration.
            else:
                blink_csv = False

            if jaw and (time.time() - jaw_timestamp > JAW_HOLD_SEC):
                jaw_csv = jaw
                jaw = False  # Automatically reset after exceeding the specified duration.  
            else:
                jaw_csv = False
            
            # Sampling for calibration ==========================================================================
            if time.time() - self.sampling_start_time < self.sampling_duration:
                # Ignore the data recorded in burn-in period
                if time.time() - self.sampling_start_time < self.burn_in_period:
                    remaining_burn_in = int(self.burn_in_period - (time.time() - self.sampling_start_time))
                    self.labels['Meditation'].setText(f"Status: Burn-in period \nWait: {remaining_burn_in}s")
                    self.labels['Meditation'].setStyleSheet("color: blue; font-size: 24px;")
                    self.th_circle_item.setRect(-self.r_th_meditat, -self.r_th_meditat, 2*self.r_th_meditat, 2*self.r_th_meditat) 

                    self.labels['Arousal'].setText(f"Status: Burn-in period \nWait: {remaining_burn_in}s")
                    self.labels['Arousal'].setStyleSheet("color: blue; font-size: 24px;")
                    self.circle_arousal_item.setRect(-self.r_th_arousal, -self.r_th_arousal, 2*self.r_th_arousal, 2*self.r_th_arousal) 

                    self.labels['HRV'].setText(f"Status: Burn-in period \nWait: {remaining_burn_in}s")
                    self.labels['HRV'].setStyleSheet("color: blue; font-size: 24px;")

                    if recording and csv_writer:
                        row = [time.time(), concentration, mellow] + \
                            [delta_buffers[ch][-1] if delta_buffers[ch] else 0.0 for ch in channels] + \
                            [theta_buffers[ch][-1] if theta_buffers[ch] else 0.0 for ch in channels] + \
                            [alpha_buffers[ch][-1] if alpha_buffers[ch] else 0.0 for ch in channels] + \
                            [beta_buffers[ch][-1] if beta_buffers[ch] else 0.0 for ch in channels] + \
                            [gamma_buffers[ch][-1] if gamma_buffers[ch] else 0.0 for ch in channels] + \
                            gyro_data + acc_data + [int(blink_csv), int(jaw_csv), 'Burn-in', self.hr_value, self.rmssd_value]
                        csv_writer.writerow(row)
                    return

                if recording and csv_writer:
                    row = [time.time(), concentration, mellow] + \
                        [delta_buffers[ch][-1] if delta_buffers[ch] else 0.0 for ch in channels] + \
                        [theta_buffers[ch][-1] if theta_buffers[ch] else 0.0 for ch in channels] + \
                        [alpha_buffers[ch][-1] if alpha_buffers[ch] else 0.0 for ch in channels] + \
                        [beta_buffers[ch][-1] if beta_buffers[ch] else 0.0 for ch in channels] + \
                        [gamma_buffers[ch][-1] if gamma_buffers[ch] else 0.0 for ch in channels] + \
                        gyro_data + acc_data + [int(blink_csv), int(jaw_csv), 'Calibration', self.hr_value, self.rmssd_value]
                    csv_writer.writerow(row)

                remaining = int(self.sampling_duration - (time.time() - self.sampling_start_time))
                
                if avg_alpha_buffer and avg_theta_buffer and avg_beta_buffer:
                    x_delta, y_delta = trim_recent(avg_delta_time_buffer, avg_delta_buffer, self.display_window_sec)
                    x_alpha, y_alpha = trim_recent(avg_alpha_time_buffer, avg_alpha_buffer, self.display_window_sec)
                    x_theta, y_theta = trim_recent(avg_theta_time_buffer, avg_theta_buffer, self.display_window_sec)
                    x_beta, y_beta = trim_recent(avg_beta_time_buffer, avg_beta_buffer, self.display_window_sec)
                    x_gamma, y_gamma = trim_recent(avg_gamma_time_buffer, avg_gamma_buffer, self.display_window_sec)

                    if len(x_delta) > 0:
                        self.avg_delta_curve.setData(x_delta, y_delta)
                    if len(x_alpha) > 0:
                        self.avg_alpha_curve.setData(x_alpha, y_alpha)
                    if len(x_theta) > 0:
                        self.avg_theta_curve.setData(x_theta, y_theta)
                    if len(x_beta) > 0:
                        self.avg_beta_curve.setData(x_beta, y_beta)
                    if len(x_gamma) > 0:
                        self.avg_gamma_curve.setData(x_gamma, y_gamma)

                    now = time.time()

                    for t_time, t_val, a_time, a_val, b_time, b_val in zip(avg_theta_time_buffer, avg_theta_buffer,
                                                            avg_alpha_time_buffer, avg_alpha_buffer,
                                                            avg_beta_time_buffer, avg_beta_buffer):
                        if now - t_time < self.sampling_duration and now - a_time < self.sampling_duration and a_val > 1e-6:
                            
                            # print(f"theta={t_val:.3f};beta={a_val:.3f}")
                            log_th_meditat_ratio = np.clip(t_val - a_val, -1.5, 1.5)
                            log_th_arousal_ratio = np.clip(t_val - b_val, -1.5, 1.5)

                            # store history for calculation of average ratio after calibration
                            self.meditat_history.append(log_th_meditat_ratio)
                            self.arousal_history.append(log_th_arousal_ratio)

                            # Exponential smoothing
                            if hasattr(self, 'last_th_meditat_ratio'):
                                smooth_th_meditat_ratio = self.smooth_factor * log_th_meditat_ratio + (1 - self.smooth_factor) * self.last_th_meditat_ratio
                            else: 
                                smooth_th_meditat_ratio = log_th_meditat_ratio
                            self.last_th_meditat_ratio = smooth_th_meditat_ratio

                            if hasattr(self, 'last_th_arousal_ratio'):
                                smooth_th_arousal_ratio = self.smooth_factor * log_th_arousal_ratio + (1 - self.smooth_factor) * self.last_th_arousal_ratio
                            else: 
                                smooth_th_arousal_ratio = log_th_arousal_ratio
                            self.last_th_arousal_ratio = smooth_th_arousal_ratio


                    self.r_meditat_temp = self.log_ratio_to_radius(self.last_th_meditat_ratio)
                    self.circle_meditat_threshold = self.radius_to_log_ratio(self.r_meditat_temp)
                    
                    self.r_arousal_temp = self.log_ratio_to_radius(self.last_th_arousal_ratio)
                    self.circle_arousal_threshold = self.radius_to_log_ratio(self.r_arousal_temp)

                    # Calculate the min and max radius of meditation circle
                    if self.r_min > self.r_meditat_temp:
                        self.r_min = self.r_meditat_temp
                    if self.r_max < self.r_meditat_temp:
                        self.r_max = self.r_meditat_temp

                    

                    noise_channel.set_volume(1)  #control noise range!

                    self.circle_item.setRect(-self.r_meditat_temp, -self.r_meditat_temp, 0, 0)  
                    self.th_circle_item.setRect(-self.r_meditat_temp, -self.r_meditat_temp, 2*self.r_meditat_temp, 2*self.r_meditat_temp)
                    
                    self.circle_arousal_item.setRect(-self.r_arousal_temp, -self.r_arousal_temp, 0, 0) 
                    self.th_circle_arousal_item.setRect(-self.r_arousal_temp, -self.r_arousal_temp, 2*self.r_arousal_temp, 2*self.r_arousal_temp)

                    self.labels['Meditation'].setText(f"Meditation panel \
                                                      \nStatus: Calibration \
                                                      \nWait: {remaining}s remaining \
                                                      \nThreshold          = {dB_to_raw_amplitude(self.circle_meditat_threshold):.3f} \
                                                      \nRadius (baseline) = {self.r_meditat_temp:.3f} \nTotal time: {(time.time() - self.sampling_start_time):.2f} \
                                                        \n\
                                                        \n============================================== \
                                                        \n     Calibration: Silent Reading \
                                                        \nPlease sit comfortably, relax your body, \
                                                        \nand clear your mind. When you're ready, \
                                                        \nsilently read the following poem in your mind:  \
                                                        \n============================================== \
                                                        \nAll that is gold does not glitter, \
                                                        \nNot all those who wander are lost; \
                                                        \nThe old that is strong does not wither, \
                                                        \nDeep roots are not reached by the frost. \
                                                        \nFrom the ashes a fire shall be woken, \
                                                        \nA light from the shadows shall spring; \
                                                        \nRenewed shall be blade that was broken, \
                                                        \nThe crownless again shall be king. \
                                                        \n—— J.R.R. Tolkien, \
                                                        \nThe Fellowship of the Ring (1954) \
                                                        \n==============================================")
                    
                    self.labels['Arousal'].setText(f"Arousal panel \
                                                   \nStatus: Calibration \
                                                   \nWait: {remaining}s remaining \
                                                   \nThreshold          = {dB_to_raw_amplitude(self.circle_arousal_threshold):.3f} \
                                                   \nRadius (baseline) = {self.r_arousal_temp:.3f} \nTotal time: {(time.time() - self.sampling_start_time):.2f} \
                                                   \nvolume = {noise_channel.get_volume():.3f}")

                    self.labels['Meditation'].setStyleSheet("color: gray; font-size: 24px;")
                    self.labels['Arousal'].setStyleSheet("color: gray; font-size: 24px;")

                return

            if not hasattr(self, 'theta_alpha_log_threshold'):
                threshold = np.median(self.meditat_history)

                self.r_th_meditat = self.log_ratio_to_radius(threshold)*self.threshold_factor_meditat
                self.theta_alpha_log_threshold = self.radius_to_log_ratio(self.r_th_meditat)

                self.circle_item.setRect(-self.r_th_meditat, -self.r_th_meditat, 0, 0) 
                self.th_circle_item.setRect(-self.r_th_meditat, -self.r_th_meditat, 2*self.r_th_meditat, 2*self.r_th_meditat)
                
                self.labels['Meditation'].setText(f"Meditation panel \nThreshold={dB_to_raw_amplitude(self.theta_alpha_log_threshold):.3f} dB \nRadius (base) = {self.r_th_meditat:.3f}")

                # determin the level that noise is zero!
                self.r_min = self.r_th_meditat # min(self.r_min,

            if not hasattr(self, 'theta_beta_log_threshold'):
                threshold = np.median(self.arousal_history)

                self.r_th_arousal = self.log_ratio_to_radius(threshold)*self.threshold_factor_arousal
                self.theta_beta_log_threshold = self.radius_to_log_ratio(self.r_th_arousal)

                self.circle_arousal_item.setRect(-self.r_th_arousal, -self.r_th_arousal, 0, 0) 
                self.th_circle_arousal_item.setRect(-self.r_th_arousal, -self.r_th_arousal, 2*self.r_th_arousal, 2*self.r_th_arousal) 
                
                self.labels['Arousal'].setText(f"Arousal panel \nThreshold={dB_to_raw_amplitude(self.theta_beta_log_threshold):.3f} dB \nRadius (base) = {self.r_th_arousal:.3f}\
                                                   \nvolume = {noise_channel.get_volume():.3f}")

            if avg_alpha_time_buffer:
                x_delta, y_delta = trim_recent(avg_delta_time_buffer, avg_delta_buffer, self.display_window_sec)
                x_alpha, y_alpha = trim_recent(avg_alpha_time_buffer, avg_alpha_buffer, self.display_window_sec)
                x_theta, y_theta = trim_recent(avg_theta_time_buffer, avg_theta_buffer, self.display_window_sec)
                x_beta, y_beta = trim_recent(avg_beta_time_buffer, avg_beta_buffer, self.display_window_sec)
                x_gamma, y_gamma = trim_recent(avg_gamma_time_buffer, avg_gamma_buffer, self.display_window_sec)

                if len(x_delta) > 0:
                    self.avg_delta_curve.setData(x_delta, y_delta)
                if len(x_alpha) > 0:
                    self.avg_alpha_curve.setData(x_alpha, y_alpha)
                if len(x_theta) > 0:
                    self.avg_theta_curve.setData(x_theta, y_theta)
                if len(x_beta) > 0:
                    self.avg_beta_curve.setData(x_beta, y_beta)
                if len(x_gamma) > 0:
                    # self.avg_gamma_curve.setData(x_gamma, y_gamma)
                    self.avg_gamma_curve.setData(x_gamma, y_gamma)

            ############## MEDITATION #########################
            try:
                if avg_alpha_buffer and avg_theta_buffer:
                    a = avg_alpha_buffer[-1]
                    t = avg_theta_buffer[-1]
                    log_ratio_meditat = np.clip(t - a, -1.5, 1.5) 

                    if hasattr(self, 'last_ratio_meditat'):
                        smooth_meditat_ratio = self.smooth_factor * log_ratio_meditat + (1 - self.smooth_factor) * self.last_ratio_meditat

                    else:
                        smooth_meditat_ratio = log_ratio_meditat
                    self.last_ratio_meditat = smooth_meditat_ratio

                    r_meditat = self.log_ratio_to_radius(self.last_ratio_meditat)
                    last_ratio_meditat = self.radius_to_log_ratio(r_meditat)

                    self.circle_item.setRect(-r_meditat, -r_meditat, 2*r_meditat, 2*r_meditat)
                    
                    vol_meditat = (r_meditat - self.r_min) / (self.r_max - self.r_min)
                    noise_channel.set_volume(vol_meditat*noise_gain)  #control noise range!
                    # print('noise volume:',noise_channel.get_volume())
                    if r_meditat < self.r_th_meditat:
                
                        self.circle_item.setBrush(pg.mkBrush(100, 100, 255, 150))  # blue
                        
                        if self.meditation_start_time is None:
                            self.meditation_start_time = time.time()                            
                        elif self.meditation_start_time:
                            
                            if time.time() - self.meditation_start_time > self.meditation_duration_threshold:  # get reward music!
                                reward_time_last = time.time() - self.meditation_start_time - self.meditation_duration_threshold
                                self.acc_meditat_time += reward_time_last # accumulate time for meditation
                                self.labels['Meditation'].setText(f"Meditation panel \
                                                                  \nStatus: Meditation \
                                                                  \nTime: {reward_time_last:.1f}s \
                                                                  \nThreshold         = {dB_to_raw_amplitude(self.theta_alpha_log_threshold):.3f} \
                                                                  \nratio (θ/α)       = {dB_to_raw_amplitude(last_ratio_meditat):.2f} \
                                                                  \nRadius (baseline) = {self.r_th_meditat:.3f} \
                                                                  \nRadius (now)      = {r_meditat:.3f} \
                                                                  \nTotal time: {(time.time() - self.sampling_start_time):.2f} \
                                                                  \nTotal meditation time: {self.acc_meditat_time:.2f}")
                                
                                self.labels['Meditation'].setStyleSheet("color: blue; font-size: 24px;")

                                if not self.reward_mode_meditation:
                                    bg_channel.stop()
                                    noise_channel.stop()
            
                                    self.fade_in_music(reward_music, bg_channel, duration_ms=1000, target_volume=music_vol) 

                                    self.reward_mode_meditation = True
                            else: # trasition period
                                trigger_remain_time = self.meditation_duration_threshold - (time.time() - self.meditation_start_time) # when it goes negative, trigger start!
                                self.labels['Meditation'].setText(f"Meditation panel \
                                                                  \nStatus: Transition \
                                                                  \nWait: {trigger_remain_time:.1f}s  \
                                                                  \nThreshold         = {dB_to_raw_amplitude(self.theta_alpha_log_threshold):.3f} \
                                                                  \nratio (θ/α)       = {dB_to_raw_amplitude(last_ratio_meditat):.2f} \
                                                                  \nRadius (baseline) = {self.r_th_meditat:.3f} \
                                                                  \nRadius (now)      = {r_meditat:.3f} \
                                                                  \nTotal time: {(time.time() - self.sampling_start_time):.2f} \
                                                                  \nTotal meditation time: {self.acc_meditat_time:.2f}")
                                
                                self.labels['Meditation'].setStyleSheet("color: green; font-size: 24px;")
                            
                    else:
                        self.labels['Meditation'].setText(f"Meditation panel \
                                                                  \nStatus: Active Mind \
                                                                  \nThreshold         = {dB_to_raw_amplitude(self.theta_alpha_log_threshold):.3f} \
                                                                  \nratio (θ/α)       = {dB_to_raw_amplitude(last_ratio_meditat):.2f} \
                                                                  \nRadius (baseline) = {self.r_th_meditat:.3f} \
                                                                  \nRadius (now)      = {r_meditat:.3f} \
                                                                  \nTotal time: {(time.time() - self.sampling_start_time):.2f} \
                                                                  \nTotal meditation time: {self.acc_meditat_time:.2f}")
                        
                        self.labels['Meditation'].setStyleSheet("color: red; font-size: 24px;")
                        self.circle_item.setBrush(pg.mkBrush(255, 0, 0, 180))  # Red

                        self.meditation_start_time = None
                        if self.reward_mode_meditation:
                            # change the reward music back to background music
                            bg_channel.stop()
                            self.fade_in_music(bg_music, bg_channel, duration_ms=1000, target_volume=0) # music_vol mute bgm
                            bg_channel.set_volume(0) # music_vol mute BGM
                            self.fade_in_music(white_noise, bg_channel, duration_ms=1000, target_volume=vol_meditat*noise_gain)
                            noise_channel.set_volume(vol_meditat*noise_gain)  #control noise range!
                            # print('noise volume after reward:',noise_channel.get_volume())

                            self.reward_mode_meditation = False
                
            except Exception as e:
                print(f"[Circle Feedback Error] {e}")
            
             ############## Arousal #########################
            try:
                if avg_theta_buffer and avg_beta_buffer:
                    t = avg_theta_buffer[-1]
                    b = avg_beta_buffer[-1]
                    log_ratio_arousal = np.clip(t - b, -1.5, 1.5) 
                    
                    if hasattr(self, 'last_ratio_arousal'):
                        smooth_arousal_ratio = self.smooth_factor * log_ratio_arousal + (1 - self.smooth_factor) * self.last_ratio_arousal

                    else:
                        smooth_arousal_ratio = log_ratio_arousal
                    self.last_ratio_arousal = smooth_arousal_ratio

                    r_arousal = self.log_ratio_to_radius(self.last_ratio_arousal)
                    last_ratio_arousal = self.radius_to_log_ratio(r_arousal)
                    # print(f"record arousal: {last_ratio_arousal:.3f}, r ={r_arousal}")

                    self.circle_arousal_item.setRect(-r_arousal, -r_arousal, 2*r_arousal, 2*r_arousal)
                    
                    vol_arousal = (r_arousal-0.1) 
                    if r_arousal > self.r_th_arousal: # Here it should be larger not smaller
                        self.circle_arousal_item.setBrush(pg.mkBrush(255, 0, 0, 180))  # red

                        if self.arousal_start_time is None:
                            self.arousal_start_time = time.time()                            
                        elif self.arousal_start_time:
                            
                            
                            if time.time() - self.arousal_start_time > self.arousal_duration_threshold:  # get reward music!
                                reward_time_last = time.time() - self.arousal_start_time - self.arousal_duration_threshold
                                self.acc_arousal_time += reward_time_last
                                self.labels['Arousal'].setText(f"Arousal panel \
                                                                  \nStatus: Arousal \
                                                                  \nTime: {reward_time_last:.1f}s \
                                                                  \nThreshold         = {dB_to_raw_amplitude(self.theta_beta_log_threshold):.3f} \
                                                                  \nratio (θ/β)       = {dB_to_raw_amplitude(last_ratio_arousal):.2f} \
                                                                  \nRadius (baseline) = {self.r_th_arousal:.3f} \
                                                                  \nRadius (now)      = {r_arousal:.3f} \
                                                                  \nTotal time: {(time.time() - self.sampling_start_time):.2f} \
                                                                  \nTotal arousal time: {self.acc_arousal_time:.2f}\
                                                                  \nvolume = {noise_channel.get_volume():.3f}")
                            
                                
                                self.labels['Arousal'].setStyleSheet("color: blue; font-size: 24px;")

                                if not self.reward_mode_arousal:
                                    bg_channel.stop()
                                    noise_channel.stop()            
                                    self.fade_in_music(bg_music, bg_channel, duration_ms=1000, target_volume=music_vol) 

                                    self.reward_mode_arousal = True
                            else: # trasition period
                                trigger_remain_time = self.arousal_duration_threshold - (time.time() - self.arousal_start_time)
                                self.labels['Arousal'].setText(f"Arousal panel \
                                                                  \nStatus: Transition \nWait: {trigger_remain_time:.1f}s  \
                                                                  \nThreshold         = {dB_to_raw_amplitude(self.theta_beta_log_threshold):.3f} \
                                                                  \nratio (θ/β)       = {dB_to_raw_amplitude(last_ratio_arousal):.2f} \
                                                                  \nRadius (baseline) = {self.r_th_arousal:.3f} \
                                                                  \nRadius (now)      = {r_arousal:.3f} \
                                                                  \nTotal time: {(time.time() - self.sampling_start_time):.2f} \
                                                                  \nTotal arousal time: {self.acc_arousal_time:.2f}\
                                                                  \nvolume = {noise_channel.get_volume():.3f}")
                                
                                self.labels['Arousal'].setStyleSheet("color: green; font-size: 24px;")
                    else:
                        self.labels['Arousal'].setText(f"Arousal panel \
                                                                  \nStatus: Active Mind \
                                                                  \nThreshold         = {dB_to_raw_amplitude(self.theta_beta_log_threshold):.3f} \
                                                                  \nratio (θ/β)       = {dB_to_raw_amplitude(last_ratio_arousal):.2f} \
                                                                  \nRadius (baseline) = {self.r_th_arousal:.3f} \
                                                                  \nRadius (now)      = {r_arousal:.3f} \
                                                                  \nTotal time: {(time.time() - self.sampling_start_time):.2f} \
                                                                  \nTotal arousal time: {self.acc_arousal_time:.2f}\
                                                                  \nvolume = {noise_channel.get_volume():.3f}")
                        
                        self.labels['Arousal'].setStyleSheet("color: red; font-size: 24px;")
                        self.circle_arousal_item.setBrush(pg.mkBrush(100, 100, 255, 150))  # blue

                        self.arousal_start_time = None
                        if self.reward_mode_arousal:
                            # change the reward music back to background music
                            bg_channel.stop()
                            self.fade_in_music(bg_music, bg_channel, duration_ms=1000, target_volume=0) # music_vol mute bgm
                            bg_channel.set_volume(0) # music_vol mute BGM
                            self.fade_in_music(white_noise, bg_channel, duration_ms=1000, target_volume=vol_arousal*noise_gain)
                            noise_channel.set_volume(vol_arousal*noise_gain)  #control noise range!
                            print('noise volume after reward:',noise_channel.get_volume())

                            self.reward_mode_arousal = False
                
            except Exception as e:
                print(f"[Circle Feedback Error] {e}")

            if recording and csv_writer:
                row = [time.time(), concentration, mellow] + \
                        [delta_buffers[ch][-1] if delta_buffers[ch] else 0.0 for ch in channels] + \
                        [theta_buffers[ch][-1] if theta_buffers[ch] else 0.0 for ch in channels] + \
                        [alpha_buffers[ch][-1] if alpha_buffers[ch] else 0.0 for ch in channels] + \
                        [beta_buffers[ch][-1] if beta_buffers[ch] else 0.0 for ch in channels] + \
                        [gamma_buffers[ch][-1] if gamma_buffers[ch] else 0.0 for ch in channels] + \
                        gyro_data + acc_data + [int(blink_csv), int(jaw_csv), 'Test', self.hr_value, self.rmssd_value]
                csv_writer.writerow(row)

    def toggle_recording(self):
        global recording, csv_writer, csv_file
        if not recording:
            filename = f"muse2_polarH10_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_file = open(os.path.join('C:/RESEARCH_YW/EEG_muse/csv',filename), "w", newline='')
            csv_writer = csv.writer(csv_file)
            headers = ['Time', 'Concentration', 'Mellow'] + \
                      [f"Delta_{ch}" for ch in channels] + \
                      [f"Theta_{ch}" for ch in channels] + \
                      [f"Alpha_{ch}" for ch in channels] + \
                      [f"Beta_{ch}" for ch in channels] + \
                      [f"Gamma_{ch}" for ch in channels] + \
                      ['Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Blink', 'Jaw', 'Status','heart_rate','RMSSD']
            csv_writer.writerow(headers)
            recording = True
            self.btn_record.setText("Stop Recording")
        else:
            recording = False
            if csv_file:
                csv_file.close()
            self.btn_record.setText("Start Recording")

    def clear_buffers(self):
        with lock:
            for d in list(alpha_buffers.values()) + list(theta_buffers.values()) + \
                     list(hbo_buffers.values()) + list(hbr_buffers.values()) + \
                     list(alpha_times.values()) + list(theta_times.values()) + \
                     list(hbo_time_buffer.values()) + list(hbr_time_buffer.values()):
                d.clear()

# === OSC Server ===
def start_osc_server(ip="192.168.68.72", port=5000):
    disp = dispatcher.Dispatcher()
    disp.map("/muse/elements/alpha_absolute", alpha_handler)
    disp.map("/muse/elements/theta_absolute", theta_handler)
    disp.map("/muse/elements/beta_absolute", beta_handler)
    disp.map("/muse/elements/delta_absolute", delta_handler)
    disp.map("/muse/elements/gamma_absolute", gamma_handler)
    disp.map("/muse/optics", optics_handler)
    disp.map("/muse/gyroscope", gyro_handler)
    disp.map("/muse/accelerometer", acc_handler)
    disp.map("/muse/algorithm/concentration", concentration_handler)
    disp.map("/muse/algorithm/mellow", mellow_handler)
    disp.map("/muse/elements/blink", blink_handler)
    disp.map("/muse/elements/jaw_clench", jaw_handler)
    server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
    print(f"🎧 OSC server listening on {ip}:{port}")
    server.serve_forever()

# === 启动 ===
def run_app_shell():
    parser = argparse.ArgumentParser(description="EEG Biofeedback Dashboard Parameters")

    parser.add_argument("--display_window_sec", type=float, default=20.0, help="Time window (sec) for displaying EEG curves")
    parser.add_argument("--update_interval_ms", type=int, default=50, help="GUI update interval in milliseconds")
    parser.add_argument("--arousal_duration_threshold", type=float, default=5.0, help="Duration (s) required to trigger arousal reward")
    parser.add_argument("--meditation_duration_threshold", type=float, default=5.0, help="Duration (s) required to trigger arousal reward")
    parser.add_argument("--sampling_duration", type=float, default=30.0, help="Total duration (s) of calibration sampling")
    parser.add_argument("--burn_in_period", type=float, default=5.0, help="Initial burn-in period to discard unstable EEG")
    parser.add_argument("--smooth_factor", type=float, default=0.02, help="Exponential smoothing factor for ratio")
    parser.add_argument("--threshold_factor_arousal", type=float, default=2.0, help="Arousal threshold scaling factor")
    parser.add_argument("--threshold_factor_meditat", type=float, default=0.6, help="Meditation threshold scaling factor")
    parser.add_argument("--rmssd_display_window_sec", type=float, default=0.6, help="Time window (sec) for displaying RMSSD curves")
    parser.add_argument("--rmssd_max", type=float, default=50, help="RMSSD maximum value")

    
    args = parser.parse_args()

    threading.Thread(target=start_osc_server, daemon=True).start()
    app = QtWidgets.QApplication(sys.argv)

    win = MuseDashboard(
        display_window_sec=args.display_window_sec,
        update_interval_ms=args.update_interval_ms,
        arousal_duration_threshold=args.arousal_duration_threshold,
        meditation_duration_threshold=args.meditation_duration_threshold,
        sampling_duration=args.sampling_duration,
        burn_in_period=args.burn_in_period,
        smooth_factor=args.smooth_factor,
        threshold_factor_arousal=args.threshold_factor_arousal,
        threshold_factor_meditat=args.threshold_factor_meditat,
        rmssd_display_window_sec=args.rmssd_display_window_sec,
        rmssd_max=args.rmssd_display_window_sec,
    )
    
    win.show()
    sys.exit(app.exec_())


def run_app_gui():
    app = QtWidgets.QApplication(sys.argv)

    param_dialog = ParameterDialog()
    param_dialog.setWindowTitle("Configure Dashboard Settings")

    # Set font size
    font = QtGui.QFont()
    font.setPointSize(14)
    app.setFont(font)

    if param_dialog.exec_() == QtWidgets.QDialog.Accepted:
        params = param_dialog.get_parameters()

        # Start OSC thread
        threading.Thread(target=start_osc_server, daemon=True).start()

        win = MuseDashboard(**params)
        win.show()

        # Use qasync to integrate Qt’s event loop with asyncio
        loop = QEventLoop(app)
        asyncio.set_event_loop(loop)

        # Directly create coroutine task
        asyncio.ensure_future(win.connect_polar())

        # Run asyncio loop together with the Qt application
        with loop:   
            loop.run_forever()

if __name__ == "__main__":
    run_app_gui()