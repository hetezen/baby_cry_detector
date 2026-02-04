# Baby Cry Detector

A real-time audio-based baby cry detection system using frequency analysis. Monitors audio input, detects crying patterns, and can send emergency notifications when crying persists.

## Features

- **Real-time cry detection** using FFT frequency analysis
- **Three-level detection** to reduce false positives:
  - Chunk-level detection (is_crying_now)
  - Smoothed detection (multiple chunks positive)
  - Sustained detection (crying for min_cry_duration)
- **Brief sound filtering** - ignores coughs and short sounds under the minimum cry duration
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
- Shorter default windows for faster testing

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

# Custom minimum cry duration (filters brief sounds)
python cry_detector_local.py --min-cry 10

# Specify audio device
python cry_detector_local.py -d 0
```

**Options:**
- `-d, --device` - Audio input device index
- `-v, --volume` - Volume threshold
- `-r, --ratio` - Cry frequency ratio threshold
- `--record` - Enable recording of crying episodes
- `--alert` - Seconds of crying before alert
- `--reset` - Seconds of silence before episode reset
- `--min-cry` - Seconds of sustained crying before confirming episode

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

# Custom minimum cry duration (in seconds)
python cry_detector.py --min-cry 10

# All options
python cry_detector.py --record --pushover --alert 10 --reset 5 --min-cry 10
```

**Options:**
- `--record` - Enable recording of crying episodes
- `--pushover` - Enable Pushover emergency notifications
- `--alert` - Minutes of crying before alert
- `--reset` - Minutes of silence before episode reset
- `--min-cry` - Seconds of sustained crying before confirming episode

### Pushover Setup (Production)

Create a `config.py` file with your Pushover credentials:

```python
PUSHOVER_USER_KEY = "your_user_key"
PUSHOVER_API_TOKEN = "your_api_token"
```

## Detection Algorithm

1. **Chunk Analysis**: Audio is captured in chunks and analyzed using FFT
2. **Chunk Detection** (`is_crying_now`): Each chunk is evaluated for volume and cry frequency characteristics
3. **Smoothed Detection** (`smoothed_crying`): Requires multiple positive detections in recent chunks to reduce noise
4. **Sustained Detection** (`crying`): Requires smoothed_crying to persist for `min_cry_duration` seconds (filters brief sounds like coughs)
5. **Gap Tolerance**: Brief silences within a crying episode don't reset the sustained detection
6. **Alert**: Triggers after sustained crying exceeds the alert window
7. **Reset**: Episode resets after silence exceeds the reset window

## License

MIT
