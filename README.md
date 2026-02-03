# Baby Cry Detector

A real-time audio-based baby cry detection system using frequency analysis. Monitors audio input, detects crying patterns, and can send emergency notifications when crying persists.

## Features

- **Real-time cry detection** using FFT frequency analysis
- **Smoothed detection** to reduce false positives (requires multiple confirmations)
- **Episode tracking** with automatic reset after silence
- **Optional audio recording** of crying episodes
- **Pushover notifications** for emergency alerts (production version)
- **Auto-reconnect** on audio device disconnect

## Scripts

### `cry_detector.py` (Production)

Designed for deployment on a Raspberry Pi with a USB microphone. Features:
- USB audio device auto-detection
- Pushover emergency notifications (requires acknowledgment)
- Configurable alert and silence windows
- Saves recordings to USB drive

### `cry_detector_local.py` (Development/Testing)

For local testing on a laptop/desktop. Features:
- Uses default system microphone
- Configurable thresholds via command line
- No external notification dependencies

## Installation

```bash
pip install pyaudio numpy pushover  # pushover only needed for production
```

### macOS Note
You may need to grant microphone permission to Terminal in System Preferences > Privacy & Security > Microphone.

### Raspberry Pi Note
Install PortAudio first:
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio numpy pushover
```

## Usage

### Local Testing

```bash
# Basic monitoring
python cry_detector_local.py

# With recording enabled
python cry_detector_local.py --record

# Custom thresholds (more sensitive)
python cry_detector_local.py -v 500 -r 0.15

# Custom alert/reset windows
python cry_detector_local.py --alert 60 --reset 20

# Specify audio device
python cry_detector_local.py -d 0
```

**Options:**
- `-d, --device` - Audio input device index
- `-v, --volume` - Volume threshold (default: 1000)
- `-r, --ratio` - Cry frequency ratio threshold (default: 0.2)
- `--record` - Enable recording of crying episodes
- `--alert` - Seconds of crying before alert (default: 30)
- `--reset` - Seconds of silence before episode reset (default: 10)

### Production (Raspberry Pi)

```bash
# Basic monitoring
python cry_detector.py

# With recording
python cry_detector.py --record

# With Pushover notifications
python cry_detector.py --pushover

# Custom alert/reset windows (in minutes)
python cry_detector.py --alert 5 --reset 3

# All options
python cry_detector.py --record --pushover --alert 10 --reset 5
```

**Options:**
- `--record` - Enable recording of crying episodes
- `--pushover` - Enable Pushover emergency notifications
- `--alert` - Minutes of crying before alert (default: 10)
- `--reset` - Minutes of silence before episode reset (default: 5)

### Pushover Setup (Production)

Create a `config.py` file with your Pushover credentials:

```python
PUSHOVER_USER_KEY = "your_user_key"
PUSHOVER_API_TOKEN = "your_api_token"
```

## Detection Algorithm

1. Audio is captured in chunks and analyzed using FFT
2. Cry detection triggers based on volume and frequency characteristics
3. Smoothing requires multiple confirmations to reduce false positives
4. Alert triggers after sustained crying exceeds the alert window
5. Episode resets after silence exceeds the reset window

## License

MIT
