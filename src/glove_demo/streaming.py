from __future__ import annotations
import time
import numpy as np
from typing import Union, List
from psychtoolbox import PsychPortAudio, GetSecs

class AudioStream():
    """
    Wrapper around PsychportAudio/PsychToolBox with added utility functions.

    Attributes:
        pahandle (any): Handle to the audio stream.
        latency_classes (dict): Dictionary mapping latency class names to their respective numerical values.

    Methods:
        open: Opens an audio stream with specified device, type, latency class, sample rate, and channels.
        set_volume: Sets the volume of the audio stream.
        get_status: Returns the current status of the audio stream.
        wait: Waits until the audio stream is inactive.
        warm_up: Plays a short silent sound to warm up the audio stream.
        play_sound: Plays a given numpy array of audio data.
        list_devices: Lists available audio devices and their details.

    Documentation:
        PsychToolBox: https://psychtoolbox.org/docs/PsychPortAudio
        PsychoPy: https://psychopy.org/api/sound/playback.html 
        
    Warnings:
        1. PsychPortAudio is not thread-safe in exclusive mode or greater.
        2. Exclusive mode requires that exclusive control be enabled on the device.
        3. Exclusive mode requires that hardware accelaration and signal enhancement be disabled on the device.

    Usage:
        >>> from streaming import AudioStream
        >>> stream = AudioStream()
        >>> stream.list_devices()           # Find the name of the device you want to use
        >>> stream.open(
        >>>     device = -1,                # -1 = default device
        >>>     type = 'playback',          # 'playback' to use as is, or 'primary' to setup replica devices
        >>>     lat_class = 'aggressive',   # aggressive is recommended for experiments
        >>>     sample_rate = 48000,
        >>>     channels = 2
        >>> )
        >>> stream.set_volume(0.5)
        >>> stream.warm_up(sample_rate = 48000)
        >>> stream.wait()
        >>> audio = read('example.wav')
        >>> start_time = stream.play_sound(audio_data = audio)
    """

    def __init__(self) -> None:
        self.pahandle = None
        self.n_channels = None
        self.sample_rate = None
        self.latency_classes = dict(unimportant = 0, default = 1, exclusive = 2, aggressive = 3, critical = 4) 

    def open(self, 
             device: Union[int, str] = -1,
             api: str = 'Windows WASAPI',
             stream_type: str = "playback", 
             lat_class: str = "aggressive",
             sample_rate: int = 48000,
             channels: int = 8) -> None:
        
        self.sample_rate = sample_rate
        self.n_channels = channels
        deviceid = device if isinstance(device, int) else self._set_device(device, api)
        mode = 1 if stream_type == 'playback' else 1+8
        reqlatencyclass = self.latency_classes[lat_class]
        
        try:
            self.pahandle = PsychPortAudio('Open', deviceid, mode, reqlatencyclass, sample_rate, channels) 
        except Exception as e:
            raise Exception(f"Failed to open audio stream: {e}")
    
    def close(self) -> None:
        print('closing stream')
        PsychPortAudio('Close')

    def set_volume(self, volume: float) -> None:
        PsychPortAudio('Volume', self.pahandle, volume)

    def get_status(self) -> dict:
        return PsychPortAudio('GetStatus', self.pahandle)

    def wait(self) -> None:
        status = self.get_status()
        while status['Active']:
            status = self.get_status()
            time.sleep(0.01)

    def warm_up(self, sample_rate) -> None:
        self.play_sound(np.zeros(int(sample_rate * 0.1)))

    def start(self):
        PsychPortAudio('Start', self.pahandle)

    def play_sound(self, audio_data: np.ndarray) -> float:
        buffer = PsychPortAudio('CreateBuffer', self.pahandle, audio_data)
        PsychPortAudio('FillBuffer', self.pahandle, buffer)
        start_time = PsychPortAudio('Start', self.pahandle, 1, GetSecs()+0.1, 1) 
        return start_time

    
    def _set_device(self, dname: str, api: str) -> int:
        for d in PsychPortAudio('GetDevices'):
            if d['DeviceName'] == dname and d['HostAudioAPIName'] == api:
                return d['DeviceIndex']
        print(f"Device {dname} not found. Using default device.")
        return -1
    
    def list_devices(self):
        devices = PsychPortAudio('GetDevices')
        for d in devices:
            print(f"Device index: {d['DeviceIndex']}, Name: {d['DeviceName']}, API: {d['HostAudioAPIName']}, N channels: {d['NrOutputChannels']}")


class Replica(AudioStream):
    def __init__(self, primary: AudioStream) -> None:
        self.primary = primary
        self.pahandle = None
        self.n_channels = self.primary.n_channels

    def open(self, channels: int = 8, selectchannels: List[float] = []) -> None:     
        mode = 1 # Only playback is implemented
        self.n_channels = channels
        #PsychPortAudio('Start', self.primary.pahandle)
        self.pahandle = PsychPortAudio('OpenSlave', self.primary.pahandle, mode, channels, selectchannels) 

    def on_repeat(self, sound: np.ndarray) -> None:
        buffer = PsychPortAudio('CreateBuffer', self.pahandle, sound)
        PsychPortAudio('FillBuffer', self.pahandle, buffer)
        PsychPortAudio('Start', self.pahandle, 0)