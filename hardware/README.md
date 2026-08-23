# Hardware

- **Compute**: NVIDIA Jetson Orin Nano 8GB Developer Kit, JetPack 6.2.1.
- **Audio in**: SHOOPE USB microphone.
- **Audio out**: USB speaker, played via `plughw:0,0`.
- **Display**: Waveshare 2.8" resistive touch TFT LCD, primary ask/interact trigger.
- **Touch sensor**: TTP223 capacitive module, repurposed as a no-look "repeat last
  answer" control. Confirmed to sense through a thin bamboo slice (up to ~5-6mm, no
  air gap) with a thin plastic film glued underneath for flush contact.
- **RTC**: DS3231 + AT24C32 EEPROM module.
- **Enclosure**: ~17x16cm footprint, mic on a raised turret, bamboo trim panel
  (local MSME fabrication), LCD slot reassigned to a power bank slot.

## Known constraint

The Jetson Orin Nano's default CMA (contiguous memory allocator) pool is only 256MB,
separate from general RAM. This caused a hardware-level NvMap/CMA crash when the
original TTS model (OmniVoice, ~2.3GB + 770MB combined) tried to load alongside ASR.
See `tts/README.md` for the fix.

## Setup notes

- Flashed via Balena Etcher, no SDK Manager required (firmware was already
  6.x-compatible).
- SSH access over Ethernet.
- Installed via Ansible playbook (`suno-sutra-sw`), fixes needed for a git-lfs pointer
  file issue, a hardcoded path, and an incompatible pinmux task.
- Hugging Face auth: `hf auth login` (the older `huggingface-cli login` is deprecated).
