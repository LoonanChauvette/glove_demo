from glove_demo.streaming import AudioStream, Replica
from psychtoolbox import PsychPortAudio
import numpy as np
import soundfile as sf
import sys
import tkinter as tk
from pathlib import Path


class QuickSinUI:
    def __init__(self, root):
        self.root = root
        self.root.title("QuickSIN")
        self.root.geometry("500x500")

        self._all_devices = PsychPortAudio("GetDevices")

        self.quicksin_dir = self._get_data_path()
        self.mri_dir = self._get_mri_path()
        apis = sorted({d["HostAudioAPIName"] for d in self._all_devices})
        chs = sorted({d["NrOutputChannels"] for d in self._all_devices})
        lists = sorted(f.name for f in self.quicksin_dir.iterdir() if f.is_file())
        mri_files = self._scan_mri_files()

        # State
        self.var_api = tk.StringVar(value=apis[0] if apis else "")
        self.var_ch = tk.IntVar(value=chs[0] if chs else 0)
        self.var_dev_label = tk.StringVar(value="— select —")
        self.var_dev_id = tk.IntVar(value=-1)  # <- keep selected device_id
        self.var_list = tk.StringVar(
            value=lists[1] if len(lists) > 1 else lists[0] if lists else ""
        )
        self.var_snr = tk.DoubleVar(value=0.0)
        self.var_gain = tk.DoubleVar(value=1.0)
        self.var_routing = []
        self.var_noise_type = tk.StringVar(value="Original")
        self.var_mri_file = tk.StringVar(value=mri_files[0] if mri_files else "")

        # UI
        tk.Label(self.root, text="Channels").pack(anchor="w", padx=8, pady=(8, 0))
        tk.OptionMenu(
            self.root, self.var_ch, *chs, command=self._on_filter_change
        ).pack(fill="x", padx=8)

        tk.Label(self.root, text="API").pack(anchor="w", padx=8, pady=(8, 0))
        tk.OptionMenu(
            self.root, self.var_api, *apis, command=self._on_filter_change
        ).pack(fill="x", padx=8)

        tk.Label(self.root, text="Device").pack(anchor="w", padx=8, pady=(8, 0))
        self.device_menu = tk.OptionMenu(self.root, self.var_dev_label, "— select —")
        self.device_menu.pack(fill="x", padx=8)

        tk.Label(self.root, text="List").pack(anchor="w", padx=8, pady=(8, 0))
        tk.OptionMenu(
            self.root, self.var_list, *lists, command=self._load_stimuli
        ).pack(fill="x", padx=8)

        tk.Label(self.root, text="Noise Type").pack(anchor="w", padx=8, pady=(8, 0))
        tk.OptionMenu(
            self.root,
            self.var_noise_type,
            "Original",
            "MRI",
            command=self._on_noise_type_change,
        ).pack(fill="x", padx=8)

        tk.Label(self.root, text="MRI File").pack(anchor="w", padx=8, pady=(8, 0))
        self.mri_file_menu = tk.OptionMenu(
            self.root, self.var_mri_file, *mri_files if mri_files else ["No MRI files"]
        )
        self.mri_file_menu.pack(fill="x", padx=8)
        if self.var_noise_type.get() == "Original":
            self.mri_file_menu.config(state="disabled")

        tk.Label(self.root, text="SNR").pack(anchor="w", padx=8, pady=(8, 0))
        tk.Scale(
            self.root, from_=-20, to=20, orient="horizontal", variable=self.var_snr
        ).pack(fill="x", padx=8)

        tk.Label(self.root, text="Gain").pack(anchor="w", padx=8, pady=(8, 0))
        tk.Entry(self.root, textvariable=self.var_gain).pack(fill="x", padx=8)

        tk.Button(self.root, text="Start", command=self._start).pack(
            fill="x", padx=8, pady=(8, 0)
        )
        tk.Button(self.root, text="Stop", command=self._stop).pack(
            fill="x", padx=8, pady=(8, 0)
        )

        self.routing_frame = tk.Frame(self.root)
        self.routing_frame.pack(fill="x", padx=8, pady=(8, 0))

        self._build_routing_ui()

    def _get_data_path(self):
        if getattr(sys, "frozen", False):
            return Path(sys._MEIPASS) / "quicksin_data"
        return Path(__file__).resolve().parents[2] / "quicksin_data"

    def _get_mri_path(self):
        if getattr(sys, "frozen", False):
            return Path(sys._MEIPASS) / "MRI_data"
        return Path(__file__).resolve().parents[2] / "MRI_data"

    def _scan_mri_files(self):
        """Scan MRI_data folder for .wav files"""
        if not self.mri_dir.exists():
            return []
        return sorted(
            f.name
            for f in self.mri_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".wav"
        )

    def _on_filter_change(self, *_):
        self._refresh_devices()
        self._build_routing_ui()

    def _on_noise_type_change(self, *_):
        """Handle noise type selection change"""
        if self.var_noise_type.get() == "Original":
            self.mri_file_menu.config(state="disabled")
        else:  # MRI
            self.mri_file_menu.config(state="normal")

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
            channel_container_frame.grid(
                row=row, column=col, padx=2, pady=2, sticky="ew"
            )  # Less padding for density

            tk.Label(channel_container_frame, text=f"Ch {ch + 1}:").pack(side="left")
            option_menu = tk.OptionMenu(
                channel_container_frame, var, "speech", "noise", "sin", "silence"
            )
            option_menu.pack(
                side="left", fill="x", expand=True, padx=(0, 5)
            )  # Pad on the right of menu

    def _load_mri_noise(self, target_length):
        """Load MRI noise and match length to speech via looping"""
        mri_path = self.mri_dir / self.var_mri_file.get()
        mri_noise, mri_sr = sf.read(str(mri_path))

        # Handle stereo MRI files by taking first channel
        if len(mri_noise.shape) > 1:
            mri_noise = mri_noise[:, 0]

        # Resample if sample rates don't match
        if mri_sr != self.sr:
            raise ValueError(
                f"MRI file sample rate ({mri_sr}) doesn't match audio sample rate ({self.sr})"
            )

        # Loop/tile if shorter, trim if longer
        if len(mri_noise) < target_length:
            num_repeats = int(np.ceil(target_length / len(mri_noise)))
            mri_noise = np.tile(mri_noise, num_repeats)[:target_length]
        else:
            mri_noise = mri_noise[:target_length]

        return mri_noise

    def _normalize_mri_noise(self, mri_noise, original_noise):
        """Normalize MRI noise to match RMS of original noise"""
        rms_original = rms(original_noise)
        rms_mri = rms(mri_noise)
        return mri_noise * (rms_original / (rms_mri + 1e-12))

    def _load_stimuli(self, *_):
        self.audio, self.sr = sf.read("quicksin_data/" + self.var_list.get())
        self.speech = self.audio[:, 0] * float(self.var_gain.get())

        # Load noise based on selected type
        if self.var_noise_type.get() == "Original":
            self.noise = self.audio[:, 1]
        else:  # MRI
            original_noise = self.audio[:, 1]
            self.noise = self._load_mri_noise(len(self.speech))
            self.noise = self._normalize_mri_noise(self.noise, original_noise)

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
        ch = self.var_ch.get()

        devices = []
        for d in self._all_devices:
            if d["HostAudioAPIName"] != api and d["NrOutputChannels"] != ch:
                continue
            if "Voicemeeter" in d["DeviceName"]:
                continue

            devices.append(d)

        menu = self.device_menu["menu"]
        menu.delete(0, "end")

        if not devices:
            self.var_dev_label.set("No devices found")
            self.var_dev_id.set(-1)
            menu.add_command(
                label="No devices found",
                command=lambda: (
                    self.var_dev_label.set("No devices found"),
                    self.var_dev_id.set(-1),
                ),
            )
            return

        # Populate and set first as default
        self.var_dev_label.set(devices[0]["DeviceName"])
        self.var_dev_id.set(devices[0]["DeviceIndex"])

        for d in devices:
            name, idx = d["DeviceName"], d["DeviceIndex"]
            menu.add_command(
                label=name,
                command=lambda n=name, i=idx: (
                    self.var_dev_label.set(n),
                    self.var_dev_id.set(i),
                ),
            )

    def get_selected(self):
        return (
            self.var_api.get(),
            self.var_ch.get(),
            self.var_dev_label.get(),
            self.var_dev_id.get(),
        )

    def _start(self):
        self._load_stimuli()
        self.primary = AudioStream()
        self.primary.open(
            device=self.var_dev_id.get(),
            api=self.var_api.get(),
            stream_type="primary",
            lat_class="exclusive",
            sample_rate=self.sr,
            channels=self.var_ch.get(),
        )

        self.stream = Replica(primary=self.primary)
        if self.var_ch.get() == 2:
            print("Opening 2-channel stream")
            self.stream.open(channels=2, selectchannels=[[1.0, 2.0]])
        elif self.var_ch.get() == 8:
            print("Opening 8-channel stream")
            self.stream.open(
                channels=8, selectchannels=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
            )
        else:
            # error
            print("Not a valid number of channels")
            return

        self.primary.start()

        routing = [v.get() for v in self.var_routing[: self.var_ch.get()]]
        signal = self._make_routed_signal(routing, self.var_ch.get())
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
        if np.mean(pos_x[i : i + win]) >= cutoff:
            mask[i : i + win] = True
    return mask


def mix_and_calibrate(
    speech: np.ndarray, noise: np.ndarray, snr_db: float, sr: int
) -> np.ndarray:
    if noise.shape != speech.shape:
        raise ValueError(
            f"Noise shape {noise.shape} must match speech shape {speech.shape}"
        )

    window = int(0.2 * sr)
    mask = create_mask(speech, -45, window)

    rms_speech = rms(speech[mask])
    rms_noise = rms(noise)

    snr_linear = 10 ** (snr_db / 20)  # dB → linear amplitude ratiox#
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
        raise ValueError(
            f"Noise shape {noise.shape} must match speech shape {speech.shape}"
        )

    # Convert dB → linear amplitude ratio
    snr_linear = 10 ** (snr_db / 20)
    noise_scaled = noise / snr_linear

    mix = speech + noise_scaled

    # Normalize to avoid clipping
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix = mix / peak

    return mix


def main():
    root = tk.Tk()
    ui = QuickSinUI(root)

    api, ch, dev_name, dev_id = ui.get_selected()

    root.mainloop()


if __name__ == "__main__":
    main()
