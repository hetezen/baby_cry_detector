#!/usr/bin/env python3
"""
Local development version of cry detector.
Uses default microphone input for testing on laptop.
Removes Pushover notifications and USB-specific features.
"""
import pyaudio
import numpy as np
from collections import deque
import time
from datetime import datetime
import os

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

# Audio settings
RATE = 44100  # Sample rate
CHUNK = 8192  # Chunks for frequency resolution (~186ms per chunk)
FORMAT = pyaudio.paInt16

# Cry detection parameters - ADJUSTED FOR MORE SENSITIVITY
CRY_FREQ_MIN = 400   # Hz - minimum frequency for baby cry (wider range)
CRY_FREQ_MAX = 2000   # Hz - maximum frequency for baby cry (wider range)
VOLUME_THRESHOLD = 800  # Lower threshold to catch quieter cries
CRY_RATIO_THRESHOLD = 0.40  # Threshold for cry energy ratio
SMOOTHING_WINDOW = 5  # Number of recent chunks to consider
CRY_CONFIRMATION_COUNT = 2  # Need this many positive detections in window
ALERT_WINDOW = 30  # Shorter for testing: alert after 30 seconds
RESET_WINDOW = 10  # Shorter for testing: reset after 10 seconds of silence
MIN_CRY_DURATION = 5  # Seconds of sustained crying before announcing episode (filters brief sounds)
SILENCE_GAP = 5  # Seconds of silence within crying that resets potential cry detection

# Recording settings (local directory)
RECORDINGS_DIR = os.path.expanduser('~/Documents/babymonitor_recordings')
RECORDING_GRACE_PERIOD = 5  # Keep recording for 5 seconds after crying stops
MIN_RECORDING_DURATION = 3  # Shorter for testing: save recordings at least 3 seconds
MAX_RECORDING_FRAMES = 5000  # ~15 minutes at 44100/8192 (~5.4 chunks/sec) - prevents unbounded memory

class CryDetectorLocal:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.initial_start_time = None  # When confirmed crying episode began
        self.last_cry_time = None  # Most recent cry detection
        self.recent_detections = deque(maxlen=SMOOTHING_WINDOW)
        self.alert_sent = False  # Track if we've already alerted for this episode
        self.potential_cry_start_time = None  # When crying first detected (before confirmation)

        # Recording state
        self.is_recording = False
        self.current_episode_frames = []
        self.episode_start_time = None
        self.last_detected_cry_time = None  # For grace period
        self.chunk_count = 0  # For deterministic debug logging

        # Configurable thresholds (can be overridden before start())
        self.volume_threshold = VOLUME_THRESHOLD
        self.cry_ratio_threshold = CRY_RATIO_THRESHOLD
        self.enable_recording = False
        self.alert_window = ALERT_WINDOW
        self.reset_window = RESET_WINDOW
        self.min_cry_duration = MIN_CRY_DURATION
        self.silence_gap = SILENCE_GAP

        # Auto-stop settings
        self.stop_time = None  # datetime.time object, e.g. 07:00

    def list_audio_devices(self):
        """List all available audio input devices"""
        print(f"\n{YELLOW}Available audio input devices:{RESET}")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                default_marker = " (default)" if i == self.audio.get_default_input_device_info()['index'] else ""
                print(f"  [{i}] {info['name']}{default_marker}")
        print()

    def start(self, device_index=None):
        """Start the audio stream using specified device or default microphone"""
        self.list_audio_devices()

        if device_index is None:
            # Use default input device
            default_info = self.audio.get_default_input_device_info()
            device_index = int(default_info['index'])
            print(f"{GREEN}✓ Using default input device: {default_info['name']} (index {device_index}){RESET}")
        else:
            info = self.audio.get_device_info_by_index(device_index)
            print(f"{GREEN}✓ Using specified device: {info['name']} (index {device_index}){RESET}")

        self.stream = self.audio.open(
            format=FORMAT,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        print(f"\n{GREEN}Cry detector (LOCAL) started. Monitoring audio...{RESET}")
        print(f"Configuration: {CRY_FREQ_MIN}-{CRY_FREQ_MAX} Hz, Chunk: {CHUNK} samples (~{CHUNK/RATE*1000:.0f}ms)")
        print(f"Volume threshold: {self.volume_threshold}, Ratio threshold: {self.cry_ratio_threshold}")
        print(f"Confirmation: {CRY_CONFIRMATION_COUNT}/{SMOOTHING_WINDOW} chunks")
        print(f"Min cry duration: {self.min_cry_duration} seconds (filters brief sounds)")
        print(f"Silence gap: {self.silence_gap} seconds (resets potential cry)")
        print(f"Alert window: {self.alert_window} seconds")
        print(f"Reset after {self.reset_window} seconds of silence")
        if self.enable_recording:
            print(f"{YELLOW}Recording: Saving to {RECORDINGS_DIR}{RESET}")
            print(f"{YELLOW}Recording grace period: {RECORDING_GRACE_PERIOD}s, Min duration: {MIN_RECORDING_DURATION}s{RESET}")
        else:
            print(f"{YELLOW}Recording: Disabled (use --record to enable){RESET}")
        if self.stop_time:
            print(f"{GREEN}Auto-stop: {self.stop_time.strftime('%H:%M')}{RESET}")
        else:
            print(f"{YELLOW}Auto-stop: Disabled (use --stop-at HH:MM to enable){RESET}")
        print(f"\n{YELLOW}Press Ctrl+C to stop{RESET}\n")

    def analyze_audio(self, audio_data):
        """Analyze audio chunk for crying"""
        # Convert to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # Check volume level
        volume = np.abs(audio_np).mean()

        # Perform FFT to get frequency spectrum
        fft = np.fft.rfft(audio_np)
        freqs = np.fft.rfftfreq(len(audio_np), 1/RATE)
        magnitudes = np.abs(fft)

        # Find dominant frequency
        dominant_freq_idx = np.argmax(magnitudes)
        dominant_freq = freqs[dominant_freq_idx]

        # Calculate energy in cry frequency band
        cry_band_mask = (freqs >= CRY_FREQ_MIN) & (freqs <= CRY_FREQ_MAX)
        cry_energy = np.sum(magnitudes[cry_band_mask])
        total_energy = np.sum(magnitudes)
        cry_ratio = cry_energy / total_energy if total_energy > 0 else 0

        # Detect cry based on multiple criteria
        has_volume = volume > self.volume_threshold
        in_cry_freq = CRY_FREQ_MIN <= dominant_freq <= CRY_FREQ_MAX
        has_cry_energy = cry_ratio > self.cry_ratio_threshold

        is_crying_now = has_volume and (in_cry_freq or has_cry_energy)

        # Debug output - all positive events, every 10th negative event
        timestamp = datetime.now().strftime("%H:%M:%S")
        if is_crying_now:
            # Always print when crying detected
            print(f"{RED}[{timestamp}] Vol: {volume:.0f}, Freq: {dominant_freq:.1f} Hz, Ratio: {cry_ratio:.2f}, Cry: TRUE{RESET}")
        elif self.chunk_count % 10 == 0:
            # Print every 10th non-crying chunk for deterministic logging
            print(f"[{timestamp}] Vol: {volume:.0f}, Freq: {dominant_freq:.1f} Hz, Ratio: {cry_ratio:.2f}, Cry: False")

        return is_crying_now, volume, dominant_freq, cry_ratio

    def save_episode_recording(self):
        """Save the current episode's audio recording"""
        import wave

        if not self.current_episode_frames or self.episode_start_time is None:
            return

        # Calculate duration
        duration = len(self.current_episode_frames) * CHUNK / RATE

        # Only save if duration meets minimum threshold
        if duration < MIN_RECORDING_DURATION:
            print(f"{YELLOW}⏭ Skipping short recording ({duration:.1f}s < {MIN_RECORDING_DURATION}s){RESET}")
            return

        # Create recordings directory if it doesn't exist
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

        # Generate filename with episode start timestamp
        timestamp = datetime.fromtimestamp(self.episode_start_time).strftime("%Y%m%d_%H%M%S")
        filename = f'{RECORDINGS_DIR}/episode_{timestamp}_{duration:.0f}s.wav'

        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.current_episode_frames))
            print(f"{GREEN}✓ Saved episode recording: {filename} ({len(self.current_episode_frames)} chunks, {duration:.1f}s){RESET}")
        except Exception as e:
            print(f"{RED}✗ Failed to save episode recording: {e}{RESET}")

    def monitor(self):
        """Main monitoring loop"""
        try:
            while True:
                # Read audio chunk with reconnection on failure
                try:
                    audio_data = self.stream.read(CHUNK, exception_on_overflow=False)
                except (IOError, OSError) as e:
                    print(f"{RED}✗ Audio stream error: {e}{RESET}")
                    print(f"{YELLOW}Attempting to reconnect...{RESET}")
                    self.cleanup_stream()
                    time.sleep(2)
                    try:
                        self.start()
                        continue
                    except Exception as reconnect_error:
                        print(f"{RED}✗ Reconnection failed: {reconnect_error}{RESET}")
                        time.sleep(5)
                        continue

                self.chunk_count += 1
                current_time = time.time()

                # Check if we should auto-stop
                if self.stop_time:
                    now = datetime.now().time()
                    if now.hour == self.stop_time.hour and now.minute == self.stop_time.minute and now.second < 2:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"\n{GREEN}[{timestamp}] Auto-stop time reached ({self.stop_time.strftime('%H:%M')}). Shutting down...{RESET}")
                        break

                # Analyze for crying
                is_crying_now, volume, freq, ratio = self.analyze_audio(audio_data)

                # Add to recent detections for smoothing
                self.recent_detections.append(is_crying_now)

                # Count positive detections in recent window
                cry_count = sum(self.recent_detections)

                # Smoothed detection (immediate, before min_cry_duration check)
                smoothed_crying = cry_count >= CRY_CONFIRMATION_COUNT

                # Track potential cry start time for sustained detection
                if smoothed_crying:
                    if self.potential_cry_start_time is None:
                        self.potential_cry_start_time = current_time
                    self.last_cry_time = current_time
                else:
                    # Check if brief silence should reset potential cry
                    if self.potential_cry_start_time is not None and self.last_cry_time is not None:
                        silence_duration = current_time - self.last_cry_time
                        if silence_duration >= self.silence_gap:
                            # Brief sound ended - not sustained crying
                            brief_duration = self.last_cry_time - self.potential_cry_start_time
                            if brief_duration < self.min_cry_duration:
                                print(f"{YELLOW}⏭ Ignored brief sound ({brief_duration:.1f}s < {self.min_cry_duration}s){RESET}")
                            self.potential_cry_start_time = None

                # CRYING requires sustained smoothed detection for min_cry_duration
                crying = False
                if smoothed_crying and self.potential_cry_start_time is not None:
                    sustained_duration = current_time - self.potential_cry_start_time
                    if sustained_duration >= self.min_cry_duration:
                        crying = True

                # Update last detected cry time when confirmed crying
                if crying:
                    self.last_detected_cry_time = current_time

                # Handle recording with grace period (only for confirmed crying)
                if self.enable_recording and crying:
                    if not self.is_recording:
                        # Start recording this episode
                        self.is_recording = True
                        self.episode_start_time = current_time
                        self.current_episode_frames = []
                        print(f"{YELLOW}📹 Started recording episode{RESET}")

                    # Add audio chunk to current episode (with memory limit)
                    if len(self.current_episode_frames) < MAX_RECORDING_FRAMES:
                        self.current_episode_frames.append(audio_data)

                elif self.enable_recording and self.is_recording:
                    # Not currently crying but we're recording
                    # Continue recording during grace period
                    grace_elapsed = current_time - self.last_detected_cry_time if self.last_detected_cry_time else 0

                    if grace_elapsed < RECORDING_GRACE_PERIOD:
                        # Still within grace period - keep recording (with memory limit)
                        if len(self.current_episode_frames) < MAX_RECORDING_FRAMES:
                            self.current_episode_frames.append(audio_data)
                    else:
                        # Grace period expired - stop recording
                        self.is_recording = False
                        print(f"{YELLOW}⏹ Stopped recording episode (grace period expired){RESET}")
                        # Save the episode recording
                        self.save_episode_recording()
                        # Clear for next episode
                        self.current_episode_frames = []
                        self.episode_start_time = None

                # Handle confirmed crying detection (alert logic)
                if crying:
                    # Set initial start time if this is a new episode
                    if self.initial_start_time is None:
                        self.initial_start_time = self.potential_cry_start_time
                        self.alert_sent = False
                        sustained_duration = current_time - self.potential_cry_start_time
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"\n{RED}[{timestamp}] 🙁 Baby started crying! (sustained for {sustained_duration:.1f}s, Vol: {volume:.0f}, Freq: {freq:.1f} Hz){RESET}")

                    # Check if we should alert
                    elapsed_time = current_time - self.initial_start_time
                    if elapsed_time >= self.alert_window and not self.alert_sent:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"\n{RED}[{timestamp}] 🚨 ALERT! Crying for {elapsed_time:.0f} seconds!{RESET}")
                        print(f"{RED}   (Episode started at {datetime.fromtimestamp(self.initial_start_time).strftime('%H:%M:%S')}){RESET}")
                        self.alert_sent = True

                else:
                    # Not crying - check if we should reset confirmed episode
                    if self.initial_start_time is not None and self.last_cry_time is not None:
                        silence_duration = current_time - self.last_cry_time

                        # Reset after silence window
                        if silence_duration >= self.reset_window:
                            episode_duration = self.last_cry_time - self.initial_start_time
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"\n{GREEN}[{timestamp}] ☺️ Baby settled! Episode duration: {episode_duration:.1f} seconds{RESET}")
                            print(f"{GREEN}   (Silent for {silence_duration:.1f} seconds){RESET}")

                            # Reset everything
                            self.initial_start_time = None
                            self.last_cry_time = None
                            self.potential_cry_start_time = None
                            self.alert_sent = False

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Stopping cry detector...{RESET}")

            # Save any ongoing recording
            if self.is_recording and self.current_episode_frames:
                print(f"{YELLOW}Saving final episode recording...{RESET}")
                self.save_episode_recording()

            if self.initial_start_time is not None and self.last_cry_time is not None:
                episode_duration = self.last_cry_time - self.initial_start_time
                print(f"Final episode duration: {episode_duration:.1f} seconds")
        finally:
            self.cleanup()

    def cleanup_stream(self):
        """Close audio stream without terminating PyAudio"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def cleanup(self):
        """Clean up audio resources"""
        self.cleanup_stream()
        self.audio.terminate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Local cry detector for development/testing')
    parser.add_argument('-d', '--device', type=int, default=None,
                        help='Audio input device index (default: system default)')
    parser.add_argument('-v', '--volume', type=int, default=VOLUME_THRESHOLD,
                        help=f'Volume threshold (default: {VOLUME_THRESHOLD})')
    parser.add_argument('-r', '--ratio', type=float, default=CRY_RATIO_THRESHOLD,
                        help=f'Cry ratio threshold (default: {CRY_RATIO_THRESHOLD})')
    parser.add_argument('--record', action='store_true', default=False,
                        help='Enable recording of crying episodes (default: disabled)')
    parser.add_argument('--alert', type=int, default=ALERT_WINDOW,
                        help=f'Seconds of crying before alert (default: {ALERT_WINDOW})')
    parser.add_argument('--reset', type=int, default=RESET_WINDOW,
                        help=f'Seconds of silence before episode reset (default: {RESET_WINDOW})')
    parser.add_argument('--min-cry', type=int, default=MIN_CRY_DURATION,
                        help=f'Seconds of sustained crying before announcing episode (default: {MIN_CRY_DURATION})')
    parser.add_argument('--silence-gap', type=int, default=SILENCE_GAP,
                        help=f'Seconds of silence within crying that resets detection (default: {SILENCE_GAP})')
    parser.add_argument('--stop-at', type=str, default=None,
                        help='Time to auto-stop the script (HH:MM format, e.g. 07:00)')
    args = parser.parse_args()

    detector = CryDetectorLocal()
    detector.volume_threshold = args.volume
    detector.cry_ratio_threshold = args.ratio
    detector.enable_recording = args.record
    detector.alert_window = args.alert
    detector.reset_window = args.reset
    detector.min_cry_duration = args.min_cry
    detector.silence_gap = args.silence_gap
    if args.stop_at:
        try:
            h, m = args.stop_at.split(':')
            detector.stop_time = datetime.strptime(f"{h}:{m}", "%H:%M").time()
        except ValueError:
            print(f"Invalid --stop-at format: '{args.stop_at}'. Use HH:MM (e.g. 07:00)")
            exit(1)
    detector.start(device_index=args.device)
    detector.monitor()
