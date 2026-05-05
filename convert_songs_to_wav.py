"""
Pre-convert all M4A/MP3 audio files in the songs directory to WAV format.

This script helps avoid potential issues with real-time conversion during playback
on the 8-channel platform by converting all audio files to WAV format ahead of time.
"""

import librosa
import soundfile as sf
from pathlib import Path
import sys


def get_songs_path():
    """Get the path to the songs directory"""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "songs"  # type: ignore
    return Path(__file__).resolve().parent / "songs"


def convert_to_wav(input_path, output_path, target_sr=48000):
    """
    Convert audio file to WAV format using librosa and resample to target sample rate

    Args:
        input_path: Path to input audio file (M4A, MP3, etc.)
        output_path: Path to output WAV file
        target_sr: Target sample rate (default: 48000 Hz)
    """
    print(f"Converting: {input_path.name}")

    # Load the audio file (librosa can handle m4a, mp3, etc.)
    # sr=target_sr resamples to target sample rate, mono=False preserves stereo
    audio_data, sample_rate = librosa.load(str(input_path), sr=target_sr, mono=False)

    # librosa.load returns shape (n_samples,) for mono or (2, n_samples) for stereo
    # We need to transpose if stereo to match soundfile convention (n_samples, n_channels)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.T

    # Write as WAV file
    sf.write(str(output_path), audio_data, sample_rate)
    print(f"  > Created: {output_path.name}")
    print(f"  > Sample rate: {sample_rate} Hz")
    print(f"  > Shape: {audio_data.shape}")


def main():
    """Convert all M4A and MP3 files in songs directory to WAV at 48000 Hz"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert audio files to WAV format at 48000 Hz"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-conversion even if WAV files already exist",
    )
    args = parser.parse_args()

    songs_dir = get_songs_path()

    if not songs_dir.exists():
        print(f"Error: Songs directory not found at {songs_dir}")
        return

    # Find all audio files that need conversion
    audio_extensions = [".m4a", ".mp3", ".mp4"]
    files_to_convert = []

    for ext in audio_extensions:
        files_to_convert.extend(songs_dir.glob(f"*{ext}"))

    if not files_to_convert:
        print("No M4A, MP3, or MP4 files found to convert.")
        return

    print(f"Found {len(files_to_convert)} file(s) to convert")
    print(f"Target sample rate: 48000 Hz")
    if args.force:
        print("Force mode: Will overwrite existing WAV files")
    print()

    for audio_file in files_to_convert:
        # Create output filename with .wav extension
        output_file = audio_file.with_suffix(".wav")

        # Check if WAV already exists
        if output_file.exists() and not args.force:
            print(f"Skipping {audio_file.name} (WAV already exists)")
            continue

        try:
            convert_to_wav(audio_file, output_file, target_sr=48000)
            print()
        except Exception as e:
            print(f"Error converting {audio_file.name}: {e}\n")

    print("Conversion complete!")
    print("All WAV files are now at 48000 Hz sample rate.")
    print(
        "\nNote: Original M4A/MP3 files are preserved. You can delete them if desired."
    )


if __name__ == "__main__":
    main()
