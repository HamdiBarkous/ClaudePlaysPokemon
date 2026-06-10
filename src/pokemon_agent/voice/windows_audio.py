"""Plays TTS clips through Windows audio from inside WSL.

WSLg's Pulse bridge stutters (fast silence drop-outs) no matter the client —
the same wav file plays clean on Windows. So on WSL we hand playback to a
persistent powershell.exe process using Media.SoundPlayer: clips are written
as wav files to the Windows temp folder and played synchronously on the
Windows side. One PowerShell process serves the whole session, so the ~1s
startup cost is paid once, not per line.
"""

import logging
import os
import subprocess
import wave

logger = logging.getLogger(__name__)


def is_wsl() -> bool:
    return "WSL_DISTRO_NAME" in os.environ or os.path.exists(
        "/proc/sys/fs/binfmt_misc/WSLInterop"
    )


class WindowsAudioPlayer:
    """Sequential wav playback on Windows via a persistent PowerShell pipe."""

    def __init__(self):
        self._proc = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-Command", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        win_tmp = self._ask("Write-Output $env:TEMP")
        drive, rest = win_tmp.split(":", 1)
        self._win_wav = win_tmp + "\\pokemon_tts.wav"
        self._wsl_wav = f"/mnt/{drive.lower()}{rest.replace(chr(92), '/')}/pokemon_tts.wav"

    def _ask(self, command: str) -> str:
        """Run one command in the persistent PowerShell, return its last line."""
        self._proc.stdin.write(command + "; Write-Output __DONE__\n")
        self._proc.stdin.flush()
        lines = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("powershell audio bridge died")
            line = line.strip()
            if line == "__DONE__":
                return lines[-1] if lines else ""
            if line:
                lines.append(line)

    def play(self, audio, sample_rate: int) -> None:
        """Write the clip where Windows can see it and play it synchronously."""
        with wave.open(self._wsl_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(audio.tobytes())
        self._ask(f"(New-Object Media.SoundPlayer '{self._win_wav}').PlaySync()")

    def stop(self) -> None:
        try:
            self._proc.kill()
        except Exception:
            pass
