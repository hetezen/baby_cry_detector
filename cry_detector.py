#!/usr/bin/env python3
import os
import sys

# Suppress JACK and ALSA warnings
os.environ['JACK_NO_START_SERVER'] = '1'

import pyaudio
import numpy as np


def _create_pyaudio_silently():
    """Create PyAudio instance with stderr suppressed on Linux to hide JACK warnings."""
    if not sys.platform.startswith('linux'):
        return pyaudio.PyAudio()

    # Redirect stderr to /dev/null during PyAudio initialization
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    try:
        audio = pyaudio.PyAudio()
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)
    return audio


from collections import deque
import time
import wave
from datetime import datetime
from pushover import Client

# Import Pushover credentials from config file
try:
    from config import PUSHOVER_USER_KEY, PUSHOVER_API_TOKEN
except ImportError:
    PUSHOVER_USER_KEY = None
    PUSHOVER_API_TOKEN = None
    print("Warning: config.py not found. Pushover notifications disabled.")

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

# Audio settings
RATE = 44100  # Sample rate
CHUNK = 4096  # Larger chunks for better frequency resolution
FORMAT = pyaudio.paInt16

# Cry detection parameters - ADJUSTED FOR MORE SENSITIVITY
CRY_FREQ_MIN = 400   # Hz - minimum frequency for baby cry (wider range)
CRY_FREQ_MAX = 2000   # Hz - maximum frequency for baby cry (wider range)
VOLUME_THRESHOLD = 1000  # Lower threshold to catch quieter cries
CRY_RATIO_THRESHOLD = 0.20  # Lower threshold for cry energy ratio
SMOOTHING_WINDOW = 5  # Number of recent chunks to consider
CRY_CONFIRMATION_COUNT = 2  # Need this many positive detections in window
ALERT_WINDOW = 600  # Alert if crying actively happening at 10 minutes (wall clock time)
RESET_WINDOW = 300  # Reset after 5 minutes of silence

# Recording settings
RECORDINGS_DIR = '/media/tinybaby/ESD-USB/recordings'
RECORDING_GRACE_PERIOD = 10  # Keep recording for 10 seconds after crying stops
MIN_RECORDING_DURATION = 10  # Only save recordings at least 10 seconds long
MAX_RECORDING_FRAMES = 10000  # ~15 minutes at 44100/4096 (~10 chunks/sec) - prevents unbounded memory

# Pushover emergency notification settings
PUSHOVER_RETRY = 30   # Retry every 30 seconds if not acknowledged
PUSHOVER_EXPIRE = 3600  # Give up after 1 hour
ENABLE_PUSHOVER = True and PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN  # Auto-disable if credentials missing

class CryDetector:
    def __init__(self):
        self.audio = _create_pyaudio_silently()
        self.stream = None
        self.initial_start_time = None  # When crying episode first began
        self.last_cry_time = None  # Most recent cry detection
        self.recent_detections = deque(maxlen=SMOOTHING_WINDOW)
        self.alert_sent = False  # Track if we've already alerted for this episode
        
        # Recording state - only record during crying episodes
        self.is_recording = False
        self.current_episode_frames = []
        self.episode_start_time = None
        self.last_detected_cry_time = None  # For grace period
        self.chunk_count = 0  # For deterministic debug logging
        
        # Configurable options (can be overridden before start())
        self.enable_recording = False
        self.enable_pushover = False
        self.alert_window = ALERT_WINDOW
        self.reset_window = RESET_WINDOW

        # Initialize Pushover client
        if PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN:
            try:
                self.pushover_client = Client(PUSHOVER_USER_KEY, api_token=PUSHOVER_API_TOKEN)
                print(f"{GREEN}✓ Pushover client initialized{RESET}")
            except Exception as e:
                print(f"{YELLOW}⚠ Pushover initialization failed: {e}{RESET}")
                self.pushover_client = None
        else:
            self.pushover_client = None
            print(f"{YELLOW}⚠ Pushover credentials not found - notifications disabled{RESET}")

    def find_usb_audio_device(self):
        """Find USB audio device index by name"""
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if 'USB' in info['name'] and info['maxInputChannels'] > 0:
                print(f"{GREEN}✓ Found USB audio device: {info['name']} (index {i}){RESET}")
                return i
        return None

    def start(self):
        """Start the audio stream"""
        device_index = self.find_usb_audio_device()
        if device_index is None:
            print(f"{RED}✗ No USB audio device found. Available devices:{RESET}")
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  [{i}] {info['name']}")
            raise RuntimeError("No USB audio device found")

        self.stream = self.audio.open(
            format=FORMAT,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        print("Cry detector started. Monitoring audio...")
        print(f"Configuration: {CRY_FREQ_MIN}-{CRY_FREQ_MAX} Hz")
        print(f"Volume threshold: {VOLUME_THRESHOLD}, Ratio threshold: {CRY_RATIO_THRESHOLD}")
        print(f"Confirmation: {CRY_CONFIRMATION_COUNT}/{SMOOTHING_WINDOW} chunks")
        print(f"Alert window: {self.alert_window/60:.0f} minutes")
        print(f"Reset after {self.reset_window/60:.0f} minutes of silence")
        if self.enable_pushover and self.pushover_client:
            print(f"{GREEN}Pushover: Enabled (retry every {PUSHOVER_RETRY}s, expire after {PUSHOVER_EXPIRE/60:.0f} min){RESET}")
        else:
            print(f"{YELLOW}Pushover: Disabled (use --pushover to enable){RESET}")
        if self.enable_recording:
            print(f"{GREEN}Recording: Saving to {RECORDINGS_DIR}{RESET}")
            print(f"{GREEN}Recording grace period: {RECORDING_GRACE_PERIOD}s, Min duration: {MIN_RECORDING_DURATION}s{RESET}")
        else:
            print(f"{YELLOW}Recording: Disabled (use --record to enable){RESET}")
        
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
        has_volume = volume > VOLUME_THRESHOLD
        in_cry_freq = CRY_FREQ_MIN <= dominant_freq <= CRY_FREQ_MAX
        has_cry_energy = cry_ratio > CRY_RATIO_THRESHOLD
        
        is_crying_now = has_volume and (in_cry_freq or has_cry_energy)
        
        # Debug output - all positive events, 10% of negative events
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if is_crying_now:
            # Always print when crying detected
            print(f"{RED}[{timestamp}] [Monitor] Vol: {volume:.0f}, Freq: {dominant_freq:.1f} Hz, Ratio: {cry_ratio:.2f}, Cry: TRUE{RESET}")
        elif self.chunk_count % 10 == 0:
            # Print every 10th non-crying chunk for deterministic logging
            print(f"[{timestamp}] [Monitor] Vol: {volume:.0f}, Freq: {dominant_freq:.1f} Hz, Ratio: {cry_ratio:.2f}, Cry: False")
        
        return is_crying_now, volume, dominant_freq, cry_ratio
    
    def save_episode_recording(self):
        """Save the current episode's audio recording"""
        if not self.current_episode_frames or self.episode_start_time is None:
            return
        
        # Calculate duration
        duration = len(self.current_episode_frames) * CHUNK / RATE
        
        # Only save if duration meets minimum threshold
        if duration < MIN_RECORDING_DURATION:
            print(f"{YELLOW}⏭ Skipping short recording ({duration:.1f}s < {MIN_RECORDING_DURATION}s){RESET}")
            return
        
        # Create recordings directory on USB drive if it doesn't exist
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
                
                # Analyze for crying
                is_crying_now, volume, freq, ratio = self.analyze_audio(audio_data)
                
                # Add to recent detections for smoothing
                self.recent_detections.append(is_crying_now)
                
                # Count positive detections in recent window
                cry_count = sum(self.recent_detections)
                
                # Determine if baby is crying based on smoothed detection
                crying = cry_count >= CRY_CONFIRMATION_COUNT
                
                # Update last detected cry time when crying is detected
                if crying:
                    self.last_detected_cry_time = current_time
                
                # Handle recording with grace period
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
                
                # Handle crying detection (alert logic)
                if crying:
                    # Update last cry time
                    self.last_cry_time = current_time
                    
                    # Set initial start time if this is a new episode
                    if self.initial_start_time is None:
                        self.initial_start_time = current_time
                        self.alert_sent = False
                        print(f"\n{RED}🍼 Baby started crying! (Vol: {volume:.0f}, Freq: {freq:.1f} Hz){RESET}")
                    
                    # Check if we should alert
                    elapsed_time = current_time - self.initial_start_time
                    if elapsed_time >= self.alert_window and not self.alert_sent:
                        print(f"\n{RED}🚨 ALERT! Baby has been crying for {elapsed_time/60:.1f} minutes!{RESET}")
                        print(f"{RED}   (Episode started at {datetime.fromtimestamp(self.initial_start_time).strftime('%H:%M:%S')}, currently crying){RESET}")
                        
                        # Send EMERGENCY Pushover notification
                        if self.pushover_client and self.enable_pushover:
                            try:
                                self.pushover_client.send_message(
                                    f"Baby has been crying for {elapsed_time/60:.1f} minutes! Please check on the baby.",
                                    title="🚨 BABY MONITOR EMERGENCY",
                                    priority=2,  # Emergency - requires acknowledgment
                                    retry=PUSHOVER_RETRY,    # Retry every 30 seconds
                                    expire=PUSHOVER_EXPIRE   # Give up after 1 hour
                                )
                                print(f"{GREEN}✓ Emergency notification sent! (will retry every {PUSHOVER_RETRY}s until acknowledged){RESET}")
                            except Exception as e:
                                print(f"{RED}✗ Failed to send notification: {e}{RESET}")
                        
                        self.alert_sent = True
                
                else:
                    # Not currently crying - check if we should reset episode
                    if self.last_cry_time is not None:
                        silence_duration = current_time - self.last_cry_time
                        
                        # Reset after silence exceeds reset_window
                        if silence_duration >= self.reset_window:
                            if self.initial_start_time is not None:
                                episode_duration = self.last_cry_time - self.initial_start_time
                                print(f"\n{GREEN}✓ Baby settled! Episode duration: {episode_duration/60:.1f} minutes{RESET}")
                                print(f"{GREEN}   (Silent for {silence_duration/60:.1f} minutes){RESET}")
                            
                            # Reset everything
                            self.initial_start_time = None
                            self.last_cry_time = None
                            self.alert_sent = False
                        
        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Stopping cry detector...{RESET}")
            
            # Save any ongoing recording
            if self.is_recording and self.current_episode_frames:
                print(f"{YELLOW}Saving final episode recording...{RESET}")
                self.save_episode_recording()
            
            if self.initial_start_time is not None and self.last_cry_time is not None:
                episode_duration = self.last_cry_time - self.initial_start_time
                print(f"Final episode duration: {episode_duration/60:.1f} minutes")
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

    parser = argparse.ArgumentParser(description='Baby cry detector with USB audio and Pushover notifications')
    parser.add_argument('--record', action='store_true', default=False,
                        help='Enable recording of crying episodes (default: disabled)')
    parser.add_argument('--pushover', action='store_true', default=False,
                        help='Enable Pushover emergency notifications (default: disabled)')
    parser.add_argument('--alert', type=int, default=10,
                        help='Minutes of crying before alert (default: 10)')
    parser.add_argument('--reset', type=int, default=5,
                        help='Minutes of silence before episode reset (default: 5)')
    args = parser.parse_args()

    detector = CryDetector()
    detector.enable_recording = args.record
    detector.enable_pushover = args.pushover and PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN
    detector.alert_window = args.alert * 60  # Convert minutes to seconds
    detector.reset_window = args.reset * 60
    detector.start()
    detector.monitor()
