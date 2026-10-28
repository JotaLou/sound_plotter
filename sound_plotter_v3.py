import queue
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

SAMPLE_RATE = 44_100
BLOCK_SIZE = 8192
UPDATE_MS = 100
DISPLAY_FLOOR_DBFS = -120.0

audio_queue = queue.Queue(maxsize=1)
stream = None


def get_input_devices():
    """Return (device index, device info) pairs for input-capable devices."""
    return [
        (index, device)
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
    ]


input_devices = get_input_devices()
if not input_devices:
    raise RuntimeError("No audio input devices were found.")

# Use device names as the visible dropdown entries.
device_by_name = {}
device_names = []
for index, device in input_devices:
    name = f"{device['name']} (device {index})"
    device_names.append(name)
    device_by_name[name] = index


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    block = indata[:, 0].copy()
    try:
        audio_queue.put_nowait(block)
    except queue.Full:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
        audio_queue.put_nowait(block)


def start_stream(device_index):
    """Stop the previous stream, clear queued audio, and start the selected input."""
    global stream

    if stream is not None:
        stream.stop()
        stream.close()
        stream = None

    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    stream = sd.InputStream(
        device=device_index,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    )
    stream.start()


window = tk.Tk()
window.title("Live microphone spectrum")

controls = ttk.Frame(window, padding=8)
controls.pack(fill=tk.X)

ttk.Label(controls, text="Microphone:").pack(side=tk.LEFT, padx=(0, 8))

device_selector = ttk.Combobox(
    controls,
    values=device_names,
    state="readonly",
    width=50,
)
device_selector.pack(side=tk.LEFT, fill=tk.X, expand=True)
device_selector.current(0)

figure = Figure(figsize=(9, 5), dpi=100)
ax = figure.add_subplot(111)

frequencies = np.fft.rfftfreq(BLOCK_SIZE, d=1 / SAMPLE_RATE)
window_function = np.hanning(BLOCK_SIZE)

(line,) = ax.plot(
    frequencies,
    np.full_like(frequencies, DISPLAY_FLOOR_DBFS),
    label="Live",
)
(max_line,) = ax.plot(
    frequencies,
    np.full_like(frequencies, DISPLAY_FLOOR_DBFS),
    "--",
    label="Max hold",
)

ax.set_xscale("log")
ax.set_xlim(20, SAMPLE_RATE / 2)
ax.set_ylim(DISPLAY_FLOOR_DBFS, 0)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Level (dBFS)")
ax.set_title("Live microphone spectrum")
ax.grid(True, which="both")
ax.legend()

canvas = FigureCanvasTkAgg(figure, master=window)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

max_hold = np.full(len(frequencies), DISPLAY_FLOOR_DBFS)


def on_device_selected(_event=None):
    global max_hold

    selected_name = device_selector.get()
    device_index = device_by_name[selected_name]

    try:
        start_stream(device_index)
        max_hold = np.full(len(frequencies), DISPLAY_FLOOR_DBFS)
        print(f"Using: {selected_name}")
    except Exception as error:
        print(f"Could not start that microphone: {error}")


device_selector.bind("<<ComboboxSelected>>", on_device_selected)


def update_plot():
    global max_hold

    block = None
    while True:
        try:
            block = audio_queue.get_nowait()
        except queue.Empty:
            break

    if block is not None:
        if len(block) < BLOCK_SIZE:
            block = np.pad(block, (0, BLOCK_SIZE - len(block)))
        else:
            block = block[:BLOCK_SIZE]

        spectrum = np.abs(np.fft.rfft(block * window_function))
        spectrum *= 2.0 / np.sum(window_function)
        spectrum[0] /= 2.0
        if BLOCK_SIZE % 2 == 0:
            spectrum[-1] /= 2.0

        dbfs = 20 * np.log10(np.maximum(spectrum, 1e-12))
        dbfs[frequencies < 20] = DISPLAY_FLOOR_DBFS

        max_hold = np.maximum(max_hold, dbfs)
        line.set_ydata(dbfs)
        max_line.set_ydata(max_hold)
        canvas.draw_idle()

    window.after(UPDATE_MS, update_plot)

def close_window():
    global stream

    if stream is not None:
        stream.stop()
        stream.close()
    window.destroy()

device_selector.current(0)
on_device_selected()
window.protocol("WM_DELETE_WINDOW", close_window)
window.after(UPDATE_MS, update_plot)
window.mainloop()
