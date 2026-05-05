from glove_demo.streaming import AudioStream, Replica
from psychtoolbox import PsychPortAudio
import numpy as np
import soundfile as sf
import sys
import tkinter as tk
from pathlib import Path
import librosa


class QuickSinUI:
    def __init__(self, root):
        self.root = root
        self.root.title("QuickSIN Audio Testing")
        self.root.geometry("700x750")

        self._all_devices = PsychPortAudio("GetDevices")

        self.quicksin_dir = self._get_data_path()
        self.mri_dir = self._get_mri_path()
        self.songs_dir = self._get_songs_path()
        apis = sorted({d["HostAudioAPIName"] for d in self._all_devices})
        chs = sorted({d["NrOutputChannels"] for d in self._all_devices})
        lists = sorted(f.name for f in self.quicksin_dir.iterdir() if f.is_file())
        mri_files = self._scan_mri_files()
        songs = self._scan_songs()

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
        self.var_mode = tk.StringVar(value="Speech")
        self.var_song = tk.StringVar(value=songs[0] if songs else "")

        # Main container with padding
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Audio Device Configuration Frame
        device_frame = tk.LabelFrame(main_frame, text="Audio Device", padx=10, pady=5)
        device_frame.pack(fill="x", pady=(0, 10))

        tk.Label(device_frame, text="API:").grid(row=0, column=0, sticky="w", pady=2)
        tk.OptionMenu(
            device_frame, self.var_api, *apis, command=self._on_filter_change
        ).grid(row=0, column=1, sticky="ew", pady=2)

        tk.Label(device_frame, text="Channels:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        tk.OptionMenu(
            device_frame, self.var_ch, *chs, command=self._on_filter_change
        ).grid(row=1, column=1, sticky="ew", pady=2)

        tk.Label(device_frame, text="Device:").grid(row=2, column=0, sticky="w", pady=2)
        self.device_menu = tk.OptionMenu(device_frame, self.var_dev_label, "— select —")
        self.device_menu.grid(row=2, column=1, sticky="ew", pady=2)

        device_frame.columnconfigure(1, weight=1)

        # Stimulus Configuration Frame
        stimulus_frame = tk.LabelFrame(main_frame, text="Stimulus", padx=10, pady=5)
        stimulus_frame.pack(fill="x", pady=(0, 10))

        tk.Label(stimulus_frame, text="Mode:").grid(row=0, column=0, sticky="w", pady=2)

        # Toggle buttons for mode
        mode_toggle_frame = tk.Frame(stimulus_frame)
        mode_toggle_frame.grid(row=0, column=1, sticky="w", pady=2)

        self.speech_btn = tk.Radiobutton(
            mode_toggle_frame,
            text="Speech",
            variable=self.var_mode,
            value="Speech",
            command=self._on_mode_change,
            indicatoron=False,
            width=12,
            selectcolor="#4CAF50",
            relief="raised",
        )
        self.speech_btn.pack(side="left", padx=(0, 5))

        self.song_btn = tk.Radiobutton(
            mode_toggle_frame,
            text="Song",
            variable=self.var_mode,
            value="Song",
            command=self._on_mode_change,
            indicatoron=False,
            width=12,
            selectcolor="#4CAF50",
            relief="raised",
        )
        self.song_btn.pack(side="left")

        self.list_label = tk.Label(stimulus_frame, text="List:")
        self.list_label.grid(row=1, column=0, sticky="w", pady=2)
        self.list_menu = tk.OptionMenu(
            stimulus_frame, self.var_list, *lists, command=self._load_stimuli
        )
        self.list_menu.grid(row=1, column=1, sticky="ew", pady=2)

        self.song_label = tk.Label(stimulus_frame, text="Song:")
        self.song_menu = tk.OptionMenu(
            stimulus_frame,
            self.var_song,
            *songs if songs else ["No songs"],
            command=self._load_song_callback,
        )
        if self.var_mode.get() == "Speech":
            self.song_label.grid_forget()
            self.song_menu.grid_forget()
        else:
            self.song_label.grid(row=1, column=0, sticky="w", pady=2)
            self.song_menu.grid(row=1, column=1, sticky="ew", pady=2)

        stimulus_frame.columnconfigure(1, weight=1)

        # Noise Configuration Frame
        noise_frame = tk.LabelFrame(
            main_frame, text="Noise Configuration", padx=10, pady=5
        )
        noise_frame.pack(fill="x", pady=(0, 10))

        tk.Label(noise_frame, text="Noise Type:").grid(
            row=0, column=0, sticky="w", pady=2
        )

        # Toggle buttons for noise type
        noise_toggle_frame = tk.Frame(noise_frame)
        noise_toggle_frame.grid(row=0, column=1, sticky="w", pady=2)

        self.original_noise_btn = tk.Radiobutton(
            noise_toggle_frame,
            text="Original",
            variable=self.var_noise_type,
            value="Original",
            command=self._on_noise_type_change,
            indicatoron=False,
            width=12,
            selectcolor="#2196F3",
            relief="raised",
        )
        self.original_noise_btn.pack(side="left", padx=(0, 5))

        self.mri_noise_btn = tk.Radiobutton(
            noise_toggle_frame,
            text="MRI",
            variable=self.var_noise_type,
            value="MRI",
            command=self._on_noise_type_change,
            indicatoron=False,
            width=12,
            selectcolor="#2196F3",
            relief="raised",
        )
        self.mri_noise_btn.pack(side="left")

        tk.Label(noise_frame, text="MRI File:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.mri_file_menu = tk.OptionMenu(
            noise_frame,
            self.var_mri_file,
            *mri_files if mri_files else ["No MRI files"],
        )
        self.mri_file_menu.grid(row=1, column=1, sticky="ew", pady=2)
        if self.var_noise_type.get() == "Original":
            self.mri_file_menu.config(state="disabled")

        tk.Label(noise_frame, text="SNR (dB):").grid(
            row=2, column=0, sticky="w", pady=2
        )
        snr_scale = tk.Scale(
            noise_frame, from_=-20, to=20, orient="horizontal", variable=self.var_snr
        )
        snr_scale.grid(row=2, column=1, sticky="ew", pady=2)

        tk.Label(noise_frame, text="Gain:").grid(row=3, column=0, sticky="w", pady=2)
        tk.Entry(noise_frame, textvariable=self.var_gain, width=10).grid(
            row=3, column=1, sticky="w", pady=2
        )

        noise_frame.columnconfigure(1, weight=1)

        # Channel Routing Frame
        routing_outer_frame = tk.LabelFrame(
            main_frame, text="Channel Routing", padx=10, pady=5
        )
        routing_outer_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.routing_frame = tk.Frame(routing_outer_frame)
        self.routing_frame.pack(fill="both", expand=True)

        # Control Buttons Frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill="x")

        tk.Button(
            button_frame,
            text="Start",
            command=self._start,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            height=2,
        ).pack(side="left", fill="x", expand=True, padx=(0, 5))
        tk.Button(
            button_frame,
            text="Stop",
            command=self._stop,
            bg="#f44336",
            fg="white",
            font=("Arial", 10, "bold"),
            height=2,
        ).pack(side="left", fill="x", expand=True, padx=(5, 0))

        self._build_routing_ui()
        self._refresh_devices()

    def _get_data_path(self):
        if getattr(sys, "frozen", False):
            return Path(sys._MEIPASS) / "quicksin_data"
        return Path(__file__).resolve().parents[2] / "quicksin_data"

    def _get_mri_path(self):
        if getattr(sys, "frozen", False):
            return Path(sys._MEIPASS) / "MRI_data"
        return Path(__file__).resolve().parents[2] / "MRI_data"

    def _get_songs_path(self):
        if getattr(sys, "frozen", False):
            return Path(sys._MEIPASS) / "songs"
        return Path(__file__).resolve().parents[2] / "songs"

    def _scan_mri_files(self):
        """Scan MRI_data folder for .wav files"""
        if not self.mri_dir.exists():
            return []
        return sorted(
            f.name
            for f in self.mri_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".wav"
        )

    def _scan_songs(self):
        """Scan songs folder for audio files"""
        if not self.songs_dir.exists():
            return []
        return sorted(
            f.name
            for f in self.songs_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".m4a", ".mp3", ".wav"]
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

    def _on_mode_change(self, *_):
        """Handle mode selection change between Speech and Song"""
        if self.var_mode.get() == "Speech":
            # Show list label/menu, hide song label/menu
            self.list_label.grid(row=1, column=0, sticky="w", pady=2)
            self.list_menu.grid(row=1, column=1, sticky="ew", pady=2)
            self.song_label.grid_forget()
            self.song_menu.grid_forget()
        else:  # Song mode
            # Hide list label/menu, show song label/menu
            self.list_label.grid_forget()
            self.list_menu.grid_forget()
            self.song_label.grid(row=1, column=0, sticky="w", pady=2)
            self.song_menu.grid(row=1, column=1, sticky="ew", pady=2)

        # Rebuild routing UI for new mode
        self._build_routing_ui()

    def _build_routing_ui(self):
        # clear old widgets
        for widget in self.routing_frame.winfo_children():
            widget.destroy()

        self.var_routing = []
        num_channels = self.var_ch.get()
        num_columns = 2  # 2 columns for better fit with up to 8 channels

        # Determine routing options based on mode
        if self.var_mode.get() == "Speech":
            routing_options = ["silence", "speech", "noise", "sin"]
        else:  # Song mode
            routing_options = ["silence", "song", "noise", "son"]

        for ch in range(num_channels):
            var = tk.StringVar(value="silence")
            self.var_routing.append(var)

            row = ch // num_columns
            col = ch % num_columns

            tk.Label(
                self.routing_frame, text=f"Channel {ch + 1}:", anchor="e", width=12
            ).grid(row=row, column=col * 2, sticky="e", padx=(5, 5), pady=3)
            option_menu = tk.OptionMenu(self.routing_frame, var, *routing_options)
            option_menu.config(width=10)
            option_menu.grid(
                row=row, column=col * 2 + 1, sticky="w", padx=(0, 15), pady=3
            )

        # Configure column weights for proper spacing
        for i in range(num_columns * 2):
            if i % 2 == 1:  # Option menu columns
                self.routing_frame.columnconfigure(i, weight=1)

    def _load_mri_noise(self, target_length, target_sr=None):
        """Load MRI noise and match length to speech via looping"""
        mri_path = self.mri_dir / self.var_mri_file.get()
        mri_noise, mri_sr = sf.read(str(mri_path))

        # Handle stereo MRI files by taking first channel
        if len(mri_noise.shape) > 1:
            mri_noise = mri_noise[:, 0]

        # Use target_sr if provided, otherwise use self.sr
        if target_sr is None:
            target_sr = self.sr

        # Resample if sample rates don't match
        if mri_sr != target_sr:
            mri_noise = librosa.resample(mri_noise, orig_sr=mri_sr, target_sr=target_sr)

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

    def _load_song_callback(self, *_):
        """Callback when a song is selected from the dropdown"""
        # This will be called when the user selects a song
        # The actual loading happens in _start() method
        pass

    def _convert_to_wav(self, audio_path):
        """Convert audio file to WAV format using librosa"""
        # Load the audio file (librosa can handle m4a, mp3, etc.)
        # librosa.load returns (audio_data, sample_rate)
        audio_data, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)

        # librosa.load returns shape (n_samples,) for mono or (2, n_samples) for stereo
        # We need to transpose if stereo to match soundfile convention (n_samples, n_channels)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.T

        return audio_data, sample_rate

    def _load_song(self):
        """Load selected song file and prepare song-in-noise mix"""
        song_path = self.songs_dir / self.var_song.get()

        # Check if file needs conversion (not WAV)
        if song_path.suffix.lower() in [".m4a", ".mp3"]:
            song_audio, song_sr = self._convert_to_wav(song_path)
        else:
            song_audio, song_sr = sf.read(str(song_path))

        # Handle stereo by averaging channels to mono
        if len(song_audio.shape) > 1:
            song_audio = np.mean(song_audio, axis=1)

        # Determine target sample rate based on noise type
        original_noise = None
        if self.var_noise_type.get() == "Original":
            # Load speech list to get the noise channel and sample rate
            if hasattr(self, "audio") and self.audio is not None:
                original_noise = self.audio[:, 1]
                target_sr = self.sr
            else:
                # Load speech list just to get the noise
                self.audio, target_sr = sf.read(
                    str(self.quicksin_dir / self.var_list.get())
                )
                self.sr = target_sr
                original_noise = self.audio[:, 1]
        else:  # MRI
            # Get target sample rate from MRI file
            mri_path = self.mri_dir / self.var_mri_file.get()
            mri_info = sf.info(str(mri_path))
            target_sr = mri_info.samplerate
            self.sr = target_sr

        # Resample song if needed
        if song_sr != target_sr:
            song_audio = librosa.resample(
                song_audio, orig_sr=song_sr, target_sr=target_sr
            )

        # Apply gain
        self.song = song_audio * float(self.var_gain.get())

        # Load noise based on selected type
        if self.var_noise_type.get() == "Original":
            # Match noise length to song by looping/trimming
            if original_noise is not None:
                if len(original_noise) < len(self.song):
                    num_repeats = int(np.ceil(len(self.song) / len(original_noise)))
                    self.noise = np.tile(original_noise, num_repeats)[: len(self.song)]
                else:
                    self.noise = original_noise[: len(self.song)]
        else:  # MRI
            self.noise = self._load_mri_noise(len(self.song), target_sr)
            # Normalize to match song RMS
            target_rms = rms(self.song) * 0.5
            rms_noise = rms(self.noise)
            self.noise = self.noise * (target_rms / (rms_noise + 1e-12))

        # Create song-in-noise mix
        self.son = mix(self.song, self.noise, self.var_snr.get())

    def _load_stimuli(self, *_):
        self.audio, self.sr = sf.read(str(self.quicksin_dir / self.var_list.get()))
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
        # Determine signal length based on mode
        if self.var_mode.get() == "Speech":
            n_samples = len(self.speech)
        else:  # Song mode
            n_samples = len(self.song)

        out = np.zeros((n_samples, n_channels), dtype=np.float32)

        for ch, source in enumerate(routing):
            if source == "speech":
                out[:, ch] = self.speech
            elif source == "noise":
                out[:, ch] = self.noise
            elif source == "sin":
                out[:, ch] = self.sin
            elif source == "song":
                out[:, ch] = self.song
            elif source == "son":
                out[:, ch] = self.son
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
        # Load stimuli based on mode
        if self.var_mode.get() == "Speech":
            self._load_stimuli()
        else:  # Song mode
            self._load_song()

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
