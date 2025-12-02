import cv2
import numpy as np
import argparse
import time
import threading
import json
import os
from collections import deque
try:
    import winsound  # Windows audio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Audio feedback not available on this system")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  Machine Learning not available. Install scikit-learn for ML features.")
    print("   pip install scikit-learn")

# Morse code dictionary
MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '-----': '0', '--..--': ',', '.-.-.-': '.', '..--..': '?', 
    '-.-.--': '!', '-....-': '-', '-..-.': '/', '.--.-.': '@',
    '-.--.': '(', '-.--.-': ')'
}

# Audio feedback system
class AudioFeedback:
    def __init__(self, enabled=True):
        self.enabled = enabled and AUDIO_AVAILABLE
        
    def play_dot(self):
        if self.enabled:
            threading.Thread(target=lambda: winsound.Beep(800, 100), daemon=True).start()
            
    def play_dash(self):
        if self.enabled:
            threading.Thread(target=lambda: winsound.Beep(600, 300), daemon=True).start()
            
    def play_letter_complete(self):
        if self.enabled:
            threading.Thread(target=lambda: winsound.Beep(1000, 150), daemon=True).start()
            
    def play_error(self):
        if self.enabled:
            threading.Thread(target=lambda: winsound.Beep(300, 200), daemon=True).start()

# Settings management
class Settings:
    def __init__(self, filename="morse_settings.json"):
        self.filename = filename
        self.defaults = {
            "unit_time": 0.2,
            "confidence": 0.1,
            "audio_enabled": True,
            "show_guide": True,
            "auto_calibrate": True
        }
        self.settings = self.load()
        
    def load(self):
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    loaded = json.load(f)
                return {**self.defaults, **loaded}
        except Exception:
            pass
        return self.defaults.copy()
        
    def save(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Could not save settings: {e}")
            
    def get(self, key):
        return self.settings.get(key, self.defaults.get(key))
        
    def set(self, key, value):
        self.settings[key] = value
        self.save()

# Morse timing constants (in seconds) with auto-calibration
class MorseTiming:
    def __init__(self, unit_time=0.2, auto_calibrate=True):
        self.unit = unit_time
        self.auto_calibrate = auto_calibrate
        self.tap_history = deque(maxlen=20)  # Store recent tap durations
        self.update_timing()
        
    def update_timing(self):
        self.dot_max = self.unit * 1.5  # Max duration for a dot
        self.dash_min = self.unit * 2.5  # Min duration for a dash
        self.dash_max = self.unit * 6.0  # Max duration for a dash
        self.letter_gap = self.unit * 3.0  # Gap between letters
        self.word_gap = self.unit * 7.0  # Gap between words
        self.timeout = self.unit * 10.0  # Max silence before auto-decode
        
    def add_tap_duration(self, duration):
        """Add tap duration for auto-calibration"""
        if self.auto_calibrate:
            self.tap_history.append(duration)
            if len(self.tap_history) >= 10:
                self._auto_calibrate()
                
    def _auto_calibrate(self):
        """Automatically adjust timing based on user's tapping pattern"""
        if len(self.tap_history) < 10:
            return
            
        durations = sorted(self.tap_history)
        # Assume shortest taps are dots, longest are dashes
        dots = [d for d in durations if d < durations[len(durations)//2]]
        dashes = [d for d in durations if d > durations[len(durations)//2]]
        
        if dots and dashes:
            avg_dot = sum(dots) / len(dots)
            avg_dash = sum(dashes) / len(dashes)
            
            # Update unit time based on average dot duration
            new_unit = avg_dot * 1.2  # Slightly more generous than fastest dots
            if 0.1 <= new_unit <= 0.5:  # Reasonable bounds
                self.unit = new_unit
                self.update_timing()
                print(f"Auto-calibrated: unit time = {self.unit:.2f}s")


class MorseDecoder:
    def __init__(self, timing, audio_feedback=None):
        self.timing = timing
        self.audio = audio_feedback or AudioFeedback()
        self.current_signal = ''
        self.decoded_text = ''
        self.last_release_time = None
        self.click_start_time = None
        self.is_clicking = False
        self.stats = {
            'total_taps': 0,
            'dots': 0,
            'dashes': 0,
            'letters': 0,
            'words': 0
        }
        
    def reset(self):
        self.current_signal = ''
        self.decoded_text = ''
        self.last_release_time = None
        self.click_start_time = None
        self.is_clicking = False
        
    def process_click_start(self):
        """Called when click/tap begins"""
        self.click_start_time = time.time()
        
        # Check gap before this click for letter/word separation
        if self.last_release_time:
            gap_duration = self.click_start_time - self.last_release_time
            if gap_duration >= self.timing.word_gap:
                # Word separation
                self._decode_current_signal()
                self.decoded_text += ' '
            elif gap_duration >= self.timing.letter_gap:
                # Letter separation
                self._decode_current_signal()
                
        self.is_clicking = True
        
    def process_click_end(self):
        """Called when click/tap ends"""
        if not self.is_clicking or not self.click_start_time:
            return
            
        click_duration = time.time() - self.click_start_time
        self.last_release_time = time.time()
        
        # Add to timing calibration
        self.timing.add_tap_duration(click_duration)
        
        # Classify as dot or dash
        if click_duration <= self.timing.dot_max:
            self.current_signal += '.'
            self.audio.play_dot()
            self.stats['dots'] += 1
        elif click_duration >= self.timing.dash_min and click_duration <= self.timing.dash_max:
            self.current_signal += '-'
            self.audio.play_dash()
            self.stats['dashes'] += 1
        else:
            # Invalid duration - play error sound
            self.audio.play_error()
            
        self.stats['total_taps'] += 1
        self.is_clicking = False
        
    def check_timeout(self):
        """Check if we should auto-decode due to inactivity"""
        if (self.last_release_time and 
            time.time() - self.last_release_time >= self.timing.timeout):
            self._decode_current_signal()
            
    def _decode_current_signal(self):
        """Decode the current signal and add to decoded text"""
        if self.current_signal:
            letter = MORSE_CODE.get(self.current_signal, '?')
            self.decoded_text += letter
            self.audio.play_letter_complete()
            self.stats['letters'] += 1
            if letter == '?':
                print(f"Unknown pattern: {self.current_signal}")
            self.current_signal = ''
            
    def get_status(self):
        """Get current status for display"""
        return {
            'current_signal': self.current_signal,
            'decoded_text': self.decoded_text,
            'is_clicking': self.is_clicking,
            'stats': self.stats.copy(),
            'timing': {
                'unit': self.timing.unit,
                'dot_max': self.timing.dot_max,
                'dash_min': self.timing.dash_min
            }
        }
        
    def process_word_gap(self):
        """Process word separation"""
        self._decode_current_signal()
        self.decoded_text += ' '
        self.stats['words'] += 1


def draw_morse_guide(frame, show_guide=True):
    """Draw Morse code reference guide on the frame"""
    if not show_guide:
        return frame
        
    guide_items = [
        ("E: ·", "T: -", "A: ·-", "I: ··", "N: -·"),
        ("S: ···", "H: ····", "R: ·-·", "D: -··", "L: ·-··"),
        ("SOS: ··· --- ···", "HELP: ···· · ·-·· ·--·"),
        ("Quick tap = ·", "Long tap = -", "0.6s gap = letter", "1.4s gap = word")
    ]
    
    y_start = frame.shape[0] - 120
    for i, row in enumerate(guide_items):
        y_pos = y_start + (i * 25)
        x_pos = 10
        for item in row:
            cv2.putText(frame, item, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            x_pos += len(item) * 8 + 20
    
    return frame

def parse_args():
    p = argparse.ArgumentParser(description="Virtual Morse Code System - ML-powered table-tap Morse decoder")
    p.add_argument("--camera", type=int, default=0, help="Camera device index")
    p.add_argument("--confidence", type=float, help="Minimum confidence threshold for click detection")
    p.add_argument("--unit-time", type=float, help="Morse code unit time in seconds (dot duration)")
    p.add_argument("--no-audio", action="store_true", help="Disable audio feedback")
    p.add_argument("--no-guide", action="store_true", help="Hide Morse code reference guide")
    p.add_argument("--no-auto-calibrate", action="store_true", help="Disable automatic timing calibration")
    p.add_argument("--no-ml", action="store_true", help="Disable ML detection, use traditional method")
    p.add_argument("--ml-samples", type=int, default=10, help="Number of training samples per class for ML (default: 10)")
    p.add_argument("--reset-settings", action="store_true", help="Reset all saved settings to defaults")
    return p.parse_args()


class ROIDrawer:
    def __init__(self, win_name):
        self.win_name = win_name
        self.drawing = False
        self.start = None
        self.end = None
        self.current = None
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self._mouse_cb)

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end = (x, y)

    def draw(self, frame):
        out = frame.copy()
        if self.start and self.end:
            cv2.rectangle(out, self.start, self.end, (0, 255, 0), 2)
        return out

    def get_roi(self):
        if not self.start or not self.end:
            return None
        x1, y1 = self.start
        x2, y2 = self.end
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        return (x1, y1, x2, y2)


def collect_ml_training_data(cap, roi, samples_per_class=10):
    """Collect multiple training samples for ML model"""
    # Validate ROI before use
    if roi is None:
        print("Error: ROI not set. Please select a region and confirm.")
        return None, None, None
    if roi is None or not isinstance(roi, tuple) or len(roi) != 4:
        cap.release()
        cv2.destroyAllWindows()
        print("No ROI selected. Exiting.")
        return
    x1, y1, x2, y2 = roi
    hovering_frames = []
    clicking_frames = []
    
    # Collect hovering samples
    print(f"\n🧠 === ML TRAINING: Collect {samples_per_class} HOVERING samples ===")
    print("Position your hand ABOVE the table (hovering, not touching)")
    print("Press SPACE to capture each sample. Move hand slightly between captures.")
    
    count = 0
    while count < samples_per_class:
        ret, frame = cap.read()
        if not ret:
            continue
            
        roi_frame = frame[y1:y2, x1:x2]
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        progress_text = f"HOVERING Samples: {count}/{samples_per_class} - Press SPACE"
        cv2.putText(display_frame, progress_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, "Move hand slightly between captures", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("ML Training Data Collection", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space key
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
            hovering_frames.append(gray_blur)
            count += 1
            print(f"✓ Hovering sample {count}/{samples_per_class} captured!")
            time.sleep(0.5)  # Brief pause
        elif key == 27:  # ESC
            return None, None, None
    
    # Collect clicking samples
    print(f"\n🧠 === ML TRAINING: Collect {samples_per_class} CLICKING samples ===")
    print("Press your hand DOWN on the table (touching/clicking)")
    print("Press SPACE to capture each sample. Vary pressure and position slightly.")
    
    count = 0
    while count < samples_per_class:
        ret, frame = cap.read()
        if not ret:
            continue
            
        roi_frame = frame[y1:y2, x1:x2]
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        progress_text = f"CLICKING Samples: {count}/{samples_per_class} - Press SPACE"
        cv2.putText(display_frame, progress_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, "Vary pressure and position slightly", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("ML Training Data Collection", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space key
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
            clicking_frames.append(gray_blur)
            count += 1
            print(f"✓ Clicking sample {count}/{samples_per_class} captured!")
            time.sleep(0.5)  # Brief pause
        elif key == 27:  # ESC
            return None, None, None
    
    cv2.destroyWindow("ML Training Data Collection")
    
    # Train the ML model
    ml_detector = MLClickDetector()
    if ml_detector.train(hovering_frames, clicking_frames):
        print("\n🎯 === ML Model Ready for Detection! ===")
        return None, None, ml_detector  # Return ML detector instead of reference frames
    else:
        print("\n❌ === ML Training Failed, falling back to traditional method ===")
        # Fallback: use first samples as references
        hovering_ref = hovering_frames[0].astype(np.float32) if hovering_frames else None
        clicking_ref = clicking_frames[0].astype(np.float32) if clicking_frames else None
        return hovering_ref, clicking_ref, None


def capture_reference_states(cap, roi):
    """Legacy function - now redirects to ML training or traditional method"""
    if ML_AVAILABLE:
        return collect_ml_training_data(cap, roi, samples_per_class=10)
    else:
        # Fallback to traditional two-sample method
        return capture_reference_states_traditional(cap, roi)


def capture_reference_states_traditional(cap, roi):
    """Traditional two-sample reference capture (fallback)"""
    x1, y1, x2, y2 = roi
    
    # Capture hovering state
    print("\n=== STEP 1: Capture 'Hand Hovering' Reference ===")
    print("Position your hand ABOVE the table (hovering, not touching)")
    print("Press SPACE when ready to capture hovering state...")
    
    hovering_ref = None
    while hovering_ref is None:
        ret, frame = cap.read()
        if not ret:
            continue
        roi_frame = frame[y1:y2, x1:x2]
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, "Hand HOVERING - Press SPACE to capture", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Reference Capture", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space key
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            hovering_ref = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
            print("✓ Hovering state captured!")
        elif key == 27:  # ESC
            return None, None, None
    
    # Capture clicking state
    print("\n=== STEP 2: Capture 'Hand Clicking' Reference ===")
    print("Press your hand DOWN on the table (touching/clicking)")
    print("Press SPACE when ready to capture clicking state...")
    
    clicking_ref = None
    while clicking_ref is None:
        ret, frame = cap.read()
        if not ret:
            continue
        roi_frame = frame[y1:y2, x1:x2]
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, "Hand CLICKING - Press SPACE to capture", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Reference Capture", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space key
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            clicking_ref = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
            print("✓ Clicking state captured!")
        elif key == 27:  # ESC
            return None, None, None
    
    cv2.destroyWindow("Reference Capture")
    print("\n=== Ready for Detection! ===")
    return hovering_ref, clicking_ref, None


class MLClickDetector:
    """Machine Learning-based click detection using multiple training samples"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_size = None
        self.training_accuracy = 0.0
        
    def extract_features(self, frame):
        """Extract features from a frame for ML classification"""
        # Resize to consistent size
        resized = cv2.resize(frame, (32, 32))
        
        # Multiple feature types
        features = []
        
        # 1. Pixel intensities (flattened)
        pixels = resized.flatten() / 255.0
        features.extend(pixels)
        
        # 2. Statistical features
        features.extend([
            np.mean(resized),
            np.std(resized),
            np.min(resized),
            np.max(resized),
            np.median(resized)
        ])
        
        # 3. Edge density (Canny)
        edges = cv2.Canny(resized, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # 4. Texture features (Local Binary Pattern approximation)
        grad_x = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        features.extend([
            np.mean(gradient_mag),
            np.std(gradient_mag)
        ])
        
        return np.array(features)
    
    def train(self, hovering_frames, clicking_frames):
        """Train the ML model on collected samples"""
        if not ML_AVAILABLE:
            print("❌ Cannot train ML model - scikit-learn not available")
            return False
            
        print(f"🧠 Training ML model on {len(hovering_frames)} hovering + {len(clicking_frames)} clicking samples...")
        
        # Extract features
        X = []
        y = []
        
        # Hovering samples (label 0)
        for frame in hovering_frames:
            features = self.extract_features(frame)
            X.append(features)
            y.append(0)
            
        # Clicking samples (label 1)
        for frame in clicking_frames:
            features = self.extract_features(frame)
            X.append(features)
            y.append(1)
            
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            print("❌ No training data collected")
            return False
            
        self.feature_size = X.shape[1]
        print(f"📊 Feature vector size: {self.feature_size}")
        
        # Split data
        if len(X) >= 6:  # Need at least 6 samples for train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y  # Use all data for both
            
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        self.training_accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        print(f"✅ ML model trained! Accuracy: {self.training_accuracy:.2%}")
        
        return True
    
    def predict(self, frame):
        """Predict if frame shows clicking or hovering"""
        if not self.is_trained:
            return False, 0.0
            
        try:
            features = self.extract_features(frame).reshape(1, -1)
            # Guard against uninitialized scaler/model
            if self.scaler is None or self.model is None:
                return False, 0.0
            features_scaled = self.scaler.transform(features)
            
            # Get prediction and confidence
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            is_clicking = prediction == 1
            confidence = max(probabilities)  # Confidence is the highest probability
            
            return is_clicking, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return False, 0.0
    
    def get_model_info(self):
        """Get information about the trained model"""
        return {
            'is_trained': self.is_trained,
            'feature_size': self.feature_size,
            'training_accuracy': self.training_accuracy,
            'ml_available': ML_AVAILABLE
        }


def detect_click_state(current_frame, hovering_ref=None, clicking_ref=None, ml_detector=None):
    """Determine if current frame shows clicking or hovering (ML or traditional)"""
    if ml_detector and ml_detector.is_trained:
        # Use ML prediction
        is_clicking, confidence = ml_detector.predict(current_frame)
        return is_clicking, confidence, 0.0, 0.0  # Legacy scores not needed for ML
    else:
        # Fallback to traditional method
        if hovering_ref is None or clicking_ref is None:
            return False, 0.0, 0.0, 0.0
            
        hover_diff = cv2.absdiff(current_frame, cv2.convertScaleAbs(hovering_ref))
        click_diff = cv2.absdiff(current_frame, cv2.convertScaleAbs(clicking_ref))
        
        hover_score = np.mean(hover_diff)
        click_score = np.mean(click_diff)
        
        is_clicking = click_score < hover_score
        confidence = abs(hover_score - click_score) / max(hover_score, click_score) if max(hover_score, click_score) > 0 else 0.0
        
        return is_clicking, confidence, hover_score, click_score


def main():
    args = parse_args()
    
    # Load settings
    settings = Settings()
    if args.reset_settings:
        settings.settings = settings.defaults.copy()
        settings.save()
        print("Settings reset to defaults")
        return
    
    # Apply command line overrides
    if args.confidence is not None:
        settings.set('confidence', args.confidence)
    if args.unit_time is not None:
        settings.set('unit_time', args.unit_time)
    if args.no_audio:
        settings.set('audio_enabled', False)
    if args.no_guide:
        settings.set('show_guide', False)
    if args.no_auto_calibrate:
        settings.set('auto_calibrate', False)
        
    # Ensure all settings have valid values
    if settings.get('confidence') is None:
        settings.set('confidence', 0.1)
    if settings.get('unit_time') is None:
        settings.set('unit_time', 0.2)
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Cannot open camera. Please check camera connection.")
        return

    # Initialize Morse code system with settings (safe defaults)
    unit_time_val = settings.get('unit_time')
    if unit_time_val is None:
        unit_time_val = 0.18
    auto_calibrate_val = settings.get('auto_calibrate')
    if auto_calibrate_val is None:
        auto_calibrate_val = True
    audio_enabled_val = settings.get('audio_enabled')
    if audio_enabled_val is None:
        audio_enabled_val = True

    timing = MorseTiming(
        unit_time=unit_time_val,
        auto_calibrate=bool(auto_calibrate_val)
    )
    audio = AudioFeedback(enabled=bool(audio_enabled_val))
    decoder = MorseDecoder(timing, audio)
    
    win = "Virtual Morse Code System v2.0"
    drawer = ROIDrawer(win)
    
    print("=== Virtual Morse Code System v2.0 ===")
    print("🎯 Advanced table-tap Morse decoder with audio feedback!")
    print(f"📋 Dot: Quick tap (< {timing.dot_max:.1f}s) | Dash: Long tap ({timing.dash_min:.1f}s-{timing.dash_max:.1f}s)")
    print(f"⏱️  Letter gap: {timing.letter_gap:.1f}s | Word gap: {timing.word_gap:.1f}s")
    print(f"🔊 Audio: {'ON' if audio.enabled else 'OFF'} | Auto-calibrate: {'ON' if timing.auto_calibrate else 'OFF'}")
    print("📊 Press 'r' to reset, 's' for stats, 'g' to toggle guide, ESC to quit\n")

    print("Draw an ROI around the table area by dragging with the mouse. Press 'c' to confirm ROI.")
    roi = None
    hovering_ref = None
    clicking_ref = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = drawer.draw(frame)
        cv2.putText(display, "Drag mouse to select ROI. Press 'c' to confirm.", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            r = drawer.get_roi()
            if r:
                roi = r
                result = capture_reference_states(cap, roi)
                if not result or not isinstance(result, tuple) or len(result) != 3:
                    print("Failed to capture reference states or train ML model")
                    roi = None
                    continue
                hovering_ref, clicking_ref, ml_detector = result
                if (hovering_ref is None and clicking_ref is None and ml_detector is None):
                    print("Failed to capture reference states or train ML model")
                    roi = None
                else:
                    if ml_detector and ml_detector.is_trained:
                        model_info = ml_detector.get_model_info()
                        print(f"🧠 ML model ready! Accuracy: {model_info['training_accuracy']:.2%}")
                        print("🎯 Starting ML-powered detection. Press ESC to quit.")
                    else:
                        print("📊 Traditional detection ready. Press ESC to quit.")
                    break
            else:
                print("Invalid ROI, select a larger region.")
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    x1, y1, x2, y2 = roi
    last_state = False  # Track previous clicking state
    min_confidence = settings.get('confidence')  # Minimum confidence threshold for detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi_frame = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect current state (ML or traditional)
        try:
            is_clicking, confidence, hover_score, click_score = detect_click_state(
                gray_blur, hovering_ref, clicking_ref, ml_detector
            )
            
            # Ensure confidence is a valid number
            if confidence is None or not isinstance(confidence, (int, float)):
                confidence = 0.0
                
        except Exception as e:
            # Fallback values if detection fails
            is_clicking, confidence, hover_score, click_score = False, 0.0, 0.0, 0.0
            print(f"Detection error: {e}")
        
        # Process Morse code based on state transitions
        # Ensure min_confidence has a numeric default
        safe_min_conf = min_confidence if isinstance(min_confidence, (int, float)) else 0.6
        current_clicking = is_clicking and confidence > safe_min_conf
        
        if current_clicking and not last_state:
            # Click started
            decoder.process_click_start()
        elif not current_clicking and last_state:
            # Click ended
            decoder.process_click_end()
            
        # Check for timeout (auto-decode)
        decoder.check_timeout()
        
        last_state = current_clicking
        
        # Process word gap detection
        if (decoder.last_release_time and 
            time.time() - decoder.last_release_time >= timing.word_gap and
            decoder.current_signal):
            decoder.process_word_gap()
        
        # Get current status
        status = decoder.get_status()

        # Visual feedback
        if current_clicking:
            state_text = "⚡ TAPPING"
            state_color = (0, 0, 255)  # Red
            cv2.circle(roi_frame, (roi_frame.shape[1]//2, roi_frame.shape[0]//2), 25, (0, 0, 255), -1)
            cv2.putText(roi_frame, "TAP!", (roi_frame.shape[1]//2-30, roi_frame.shape[0]//2+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            state_text = "🎯 Ready"
            state_color = (0, 255, 0)  # Green

        # Create enhanced display
        display_frame = frame.copy()
        
        # Add Morse guide if enabled
        if settings.get('show_guide'):
            display_frame = draw_morse_guide(display_frame, True)
        
        # Main status with emoji
        cv2.putText(display_frame, state_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 2)
        
        # Current signal with visual dots/dashes
        signal_display = status['current_signal'].replace('.', '●').replace('-', '━') or '(none)'
        cv2.putText(display_frame, f"📡 Signal: {signal_display}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Decoded text with smart truncation
        decoded_text = status['decoded_text'] or '(tap Morse patterns)'
        if len(decoded_text) > 45:
            words = decoded_text.split()
            if len(words) > 1:
                decoded_text = '...' + ' '.join(words[-6:])  # Show last few words
            else:
                decoded_text = '...' + decoded_text[-42:]  # Show last chars
        cv2.putText(display_frame, f"📝 Text: {decoded_text}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Statistics
        stats = status['stats']
        stats_text = f"📊 {stats['letters']}L {stats['words']}W | {stats['dots']}● {stats['dashes']}━"
        cv2.putText(display_frame, stats_text, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Timing and model info
        if ml_detector and ml_detector.is_trained:
            model_info = ml_detector.get_model_info()
            timing_info = f"🧠 ML Model | Acc: {model_info['training_accuracy']:.1%} | Conf: {confidence:.2f}"
        else:
            timing_info = f"⏱️ Unit: {status['timing']['unit']:.2f}s | Conf: {confidence:.2f}"
        cv2.putText(display_frame, timing_info, (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Enhanced instructions
        instructions = "r=reset | s=stats | g=guide | ESC=quit"
        cv2.putText(display_frame, instructions, (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # ROI outline with enhanced visibility
        cv2.rectangle(display_frame, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 0), 3)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        display_frame[y1:y2, x1:x2] = roi_frame

        cv2.imshow(win, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Reset
            decoder.reset()
            print("🔄 Morse decoder reset!")
        elif key == ord('s'):  # Show statistics
            stats = status['stats']
            print(f"\n📊 === MORSE CODE STATISTICS ====")
            print(f"   Letters decoded: {stats['letters']}")
            print(f"   Words completed: {stats['words']}")
            print(f"   Total taps: {stats['total_taps']}")
            print(f"   Dots (●): {stats['dots']} | Dashes (━): {stats['dashes']}")
            if stats['total_taps'] > 0:
                accuracy = ((stats['dots'] + stats['dashes']) / stats['total_taps']) * 100
                print(f"   Recognition rate: {accuracy:.1f}%")
            print(f"   Current timing: {status['timing']['unit']:.2f}s unit")
            print(f"   Text so far: '{status['decoded_text']}'")
            print()
        elif key == ord('g'):  # Toggle guide
            current_guide = settings.get('show_guide')
            settings.set('show_guide', not current_guide)
            print(f"📋 Morse guide: {'ON' if not current_guide else 'OFF'}")
        elif key == ord('a'):  # Toggle audio
            current_audio = settings.get('audio_enabled')
            settings.set('audio_enabled', not current_audio)
            audio.enabled = not current_audio and AUDIO_AVAILABLE
            print(f"🔊 Audio feedback: {'ON' if audio.enabled else 'OFF'}")
        elif key == ord('c'):  # Toggle calibration
            timing.auto_calibrate = not timing.auto_calibrate
            settings.set('auto_calibrate', timing.auto_calibrate)
            print(f"🎯 Auto-calibration: {'ON' if timing.auto_calibrate else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()