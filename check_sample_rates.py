"""Check sample rates of all audio files in the project"""

import soundfile as sf
from pathlib import Path
import sys


def get_paths():
    """Get paths to all audio directories"""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore
    else:
        base = Path(__file__).resolve().parent

    return {
        "songs": base / "songs",
        "quicksin": base / "quicksin_data",
        "mri": base / "MRI_data",
    }


def check_directory(dir_path, dir_name):
    """Check sample rates of all audio files in a directory"""
    if not dir_path.exists():
        print(f"\n{dir_name}: Directory not found")
        return

    print(f"\n{dir_name}:")
    print("-" * 60)

    audio_files = (
        list(dir_path.glob("*.wav"))
        + list(dir_path.glob("*.m4a"))
        + list(dir_path.glob("*.mp3"))
    )

    if not audio_files:
        print("  No audio files found")
        return

    for audio_file in sorted(audio_files):
        try:
            info = sf.info(str(audio_file))
            print(f"  {audio_file.name}")
            print(f"    Sample rate: {info.samplerate} Hz")
            print(f"    Channels: {info.channels}")
            print(f"    Duration: {info.duration:.2f} seconds")
        except Exception as e:
            print(f"  {audio_file.name}: Error - {e}")


def main():
    paths = get_paths()

    print("=" * 60)
    print("AUDIO FILE SAMPLE RATE CHECK")
    print("=" * 60)

    check_directory(paths["songs"], "Songs")
    check_directory(paths["quicksin"], "QuickSIN Data")
    check_directory(paths["mri"], "MRI Data")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
