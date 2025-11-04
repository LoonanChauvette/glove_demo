from glove_demo.streaming import AudioStream, Replica
from psychtoolbox import PsychPortAudio
import numpy as np
import soundfile as sf
import tkinter as tk
from pathlib import Path


class QuickSinUI:
    def __init__(self, root):
        self.root = root
        self.root.title("QuickSIN")
        self.root.geometry("500x500")

        self._all_devices = PsychPortAudio('GetDevices')

        self.quicksin_dir = Path(__file__).resolve().parents[2] / "quicksin_data"
        apis  = sorted({d['HostAudioAPIName'] for d in self._all_devices})
        chs   = sorted({d['NrOutputChannels'] for d in self._all_devices})
        lists = sorted(f.name for f in self.quicksin_dir.iterdir() if f.is_file())

        # State
        self.var_api = tk.StringVar(value=apis[0] if apis else "")
        self.var_ch  = tk.IntVar(value=chs[0] if chs else 0)
        self.var_dev_label = tk.StringVar(value="— select —")
        self.var_dev_id    = tk.IntVar(value=-1)  # <- keep selected device_id
        self.var_list      = tk.StringVar(value=lists[1] if len(lists) > 1 else lists[0] if lists else "")
        self.var_snr       = tk.DoubleVar(value=0.0)
        self.var_gain      = tk.DoubleVar(value=1.0)
        self.var_routing   = []

        # UI
        tk.Label(self.root, text="Channels").pack(anchor="w", padx=8, pady=(8, 0))
        tk.OptionMenu(self.root, self.var_ch, *chs, command=self._on_filter_change).pack(fill="x", padx=8)

        tk.Label(self.root, text="API").pack(anchor="w", padx=8, pady=(8, 0))
        tk.OptionMenu(self.root, self.var_api, *apis, command=self._on_filter_change).pack(fill="x", padx=8)

        tk.Label(self.root, text="Device").pack(anchor="w", padx=8, pady=(8, 0))
        self.device_menu = tk.OptionMenu(self.root, self.var_dev_label, "— select —")
        self.device_menu.pack(fill="x", padx=8)

        tk.Label(self.root, text="List").pack(anchor="w", padx=8, pady=(8, 0))
        tk.OptionMenu(self.root, self.var_list, *lists, command=self._load_stimuli).pack(fill="x", padx=8)

        tk.Label(self.root, text="SNR").pack(anchor="w", padx=8, pady=(8, 0))
        tk.Scale(self.root, from_=-20, to=20, orient="horizontal", variable=self.var_snr).pack(fill="x", padx=8)

        tk.Label(self.root, text="Gain").pack(anchor="w", padx=8, pady=(8, 0))
        tk.Entry(self.root, textvariable=self.var_gain).pack(fill="x", padx=8)

        tk.Button(self.root, text="Start", command=self._start).pack(fill="x", padx=8, pady=(8, 0))
        tk.Button(self.root, text="Stop", command=self._stop).pack(fill="x", padx=8, pady=(8, 0))

        self.routing_frame = tk.Frame(self.root)
        self.routing_frame.pack(fill="x", padx=8, pady=(8, 0))

        self._build_routing_ui()

    def _on_filter_change(self, *_):
        self._refresh_devices()
        self._build_routing_ui()

    def _build_routing_ui(self):
        # clear old widgets
        for widget in self.routing_frame.winfo_children():
            widget.destroy()

        self.var_routing = []
        num_channels = self.var_ch.get()
        num_columns = 4

        for ch in range(num_channels):
            var = tk.StringVar(value="silence")
            self.var_routing.append(var)

            row = ch // num_columns
            col = ch % num_columns

            channel_container_frame = tk.Frame(self.routing_frame)
            channel_container_frame.grid(row=row, column=col, padx=2, pady=2, sticky="ew") # Less padding for density

            tk.Label(channel_container_frame, text=f"Ch {ch+1}:").pack(side="left")
            option_menu = tk.OptionMenu(channel_container_frame, var, "speech", "noise", "sin", "silence")
            option_menu.pack(side="left", fill="x", expand=True, padx=(0, 5)) # Pad on the right of menu

    def _load_stimuli(self, *_):
        self.audio, self.sr = sf.read('quicksin_data/' + self.var_list.get())
        self.speech = self.audio[:, 0] * float(self.var_gain.get())
        self.noise = self.audio[:, 1]
        self.sin = mix(self.speech, self.noise, self.var_snr.get())

    def _make_routed_signal(self, routing, n_channels):
        n_samples = len(self.speech)
        out = np.zeros((n_samples, n_channels), dtype=np.float32)

        for ch, source in enumerate(routing):
            if source == "speech":
                out[:, ch] = self.speech
            elif source == "noise":
                out[:, ch] = self.noise
            elif source == "sin":
                out[:, ch] = self.sin
        return out

    def _refresh_devices(self):
        api = self.var_api.get()
        ch  = self.var_ch.get()

        devices = []
        for d in self._all_devices:
            if d['HostAudioAPIName'] != api and d['NrOutputChannels'] != ch:
                continue
            if "Voicemeeter" in d['DeviceName']:
                continue

            devices.append(d)


        menu = self.device_menu["menu"]
        menu.delete(0, "end")

        if not devices:
            self.var_dev_label.set("No devices found")
            self.var_dev_id.set(-1)
            menu.add_command(label="No devices found",
                             command=lambda: (self.var_dev_label.set("No devices found"),
                                              self.var_dev_id.set(-1)))
            return

        # Populate and set first as default
        self.var_dev_label.set(devices[0]['DeviceName'])
        self.var_dev_id.set(devices[0]['DeviceIndex'])

        for d in devices:
            name, idx = d['DeviceName'], d['DeviceIndex']
            menu.add_command(
                label=name,
                command=lambda n=name, i=idx: (self.var_dev_label.set(n), self.var_dev_id.set(i))
            )

    def get_selected(self):
        return (self.var_api.get(),
                self.var_ch.get(),
                self.var_dev_label.get(),
                self.var_dev_id.get())

    def _start(self):
        self._load_stimuli()
        self.primary = AudioStream() 
        self.primary.open(
            device = self.var_dev_id.get(),
            api = self.var_api.get(),
            stream_type = 'primary', 
            lat_class = 'exclusive', 
            sample_rate = self.sr,
            channels = self.var_ch.get()
        )

        self.stream = Replica(primary = self.primary)
        if self.var_ch.get() == 2:
            print('Opening 2-channel stream')
            self.stream.open(channels = 2, selectchannels=[[1.,2.]])
        elif self.var_ch.get() == 8:
            print('Opening 8-channel stream')
            self.stream.open(channels = 8, selectchannels=[[1.,2.,3.,4.,5.,6.,7.,8.]])
        else:
            # error
            print('Not a valid number of channels')
            return
        
        self.primary.start()

        routing = [v.get() for v in self.var_routing[:self.var_ch.get()]]
        signal  = self._make_routed_signal(routing, self.var_ch.get())
        _ = self.stream.play_sound(signal)

    def _stop(self):
        self.stream.close()
        self.primary.close()


def rms(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x**2))

def create_mask(x: np.ndarray, thr: float, win: int) -> np.ndarray:
    pos_x = abs(x)
    peak = float(np.max(pos_x)) + 1e-12
    cutoff = peak * (10.0 ** (thr / 20.0))
    
    mask = np.zeros(len(pos_x), dtype=bool)

    for i in range(0, len(pos_x), win):
        if np.mean(pos_x[i:i+win]) >= cutoff:
            mask[i:i+win] = True
    return mask

def mix_and_calibrate(speech: np.ndarray, noise: np.ndarray, snr_db: float, sr: int) -> np.ndarray:
    if noise.shape != speech.shape:
        raise ValueError(f"Noise shape {noise.shape} must match speech shape {speech.shape}")

    window = int(0.2 * sr)
    mask = create_mask(speech, -45, window)

    rms_speech = rms(speech[mask])
    rms_noise = rms(noise)

    snr_linear = 10**(snr_db/20)  # dB → linear amplitude ratiox#
    target_rms_noise = rms_speech / snr_linear

    scaling_factor = target_rms_noise / (rms_noise + 1e-12)
    noise_scaled = noise * scaling_factor

    mix = speech + noise_scaled

    max_val = np.max(np.abs(mix))
    if max_val > 1.0:
        mix = mix / max_val

    return mix

def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if speech.shape != noise.shape:
        raise ValueError(f"Noise shape {noise.shape} must match speech shape {speech.shape}")

    # Convert dB → linear amplitude ratio
    snr_linear = 10 ** (snr_db / 20)
    noise_scaled = noise / snr_linear

    mix = speech + noise_scaled

    # Normalize to avoid clipping
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix = mix / peak

    return mix


   
if __name__ == '__main__':
    root = tk.Tk()
    ui = QuickSinUI(root)

    api, ch, dev_name, dev_id = ui.get_selected()

    root.mainloop()