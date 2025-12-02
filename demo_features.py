#!/usr/bin/env python3
"""
Virtual Morse Code System - Feature Demo & Test Suite
Demonstrates all the enhanced features of the advanced Morse system
"""

import json
import os
import time

def demo_intro():
    print("🎯 === VIRTUAL MORSE CODE SYSTEM v2.0 - FEATURE DEMO ===")
    print()
    print("🚀 NEW FEATURES ADDED:")
    print("   ✅ Audio feedback (dots, dashes, completion sounds)")
    print("   ✅ Auto-calibration (learns your tapping speed)")
    print("   ✅ Visual Morse guide (on-screen reference)")
    print("   ✅ Statistics tracking (accuracy, speed, etc.)")
    print("   ✅ Settings persistence (saves your preferences)")
    print("   ✅ Enhanced error handling & user experience")
    print("   ✅ Smart timing detection & word gap processing")
    print("   ✅ Improved visual feedback with emoji & symbols")
    print()
    
def demo_controls():
    print("🎮 === ENHANCED CONTROLS ===")
    controls = [
        ("ESC", "Quit program"),
        ("R", "Reset decoder & statistics"),
        ("S", "Show detailed statistics"),
        ("G", "Toggle Morse reference guide on/off"),
        ("A", "Toggle audio feedback on/off"),
        ("C", "Toggle auto-calibration on/off"),
        ("SPACE", "Capture reference states (during setup)")
    ]
    
    for key, desc in controls:
        print(f"   {key:6} → {desc}")
    print()

def demo_command_line():
    print("⚙️ === COMMAND LINE OPTIONS ===")
    options = [
        ("--camera 1", "Use different camera (0=default webcam)"),
        ("--confidence 0.05", "More sensitive click detection"),  
        ("--unit-time 0.15", "Faster Morse timing"),
        ("--no-audio", "Disable sound effects"),
        ("--no-guide", "Hide on-screen Morse reference"),
        ("--no-auto-calibrate", "Disable automatic timing adjustment"),
        ("--reset-settings", "Reset all saved preferences")
    ]
    
    for option, desc in options:
        print(f"   {option:20} → {desc}")
    print()
    
    print("💡 EXAMPLE USAGE:")
    print("   python table_click_detector.py --unit-time 0.15 --confidence 0.05")
    print("   python table_click_detector.py --no-audio --no-guide")
    print()

def demo_timing_guide():
    print("⏱️ === TIMING SYSTEM ===")
    print("📍 DEFAULT SETTINGS (auto-adjusts to your speed):")
    print("   • Dot (·):      Quick tap < 0.3 seconds")
    print("   • Dash (-):     Long tap 0.5-1.2 seconds") 
    print("   • Letter gap:   0.6 second pause")
    print("   • Word gap:     1.4 second pause")
    print("   • Auto-decode:  2.0 second timeout")
    print()
    print("🎯 AUTO-CALIBRATION:")
    print("   System learns from your tapping patterns!")
    print("   After 10+ taps, it automatically adjusts timing")
    print("   Faster tappers → shorter unit time")
    print("   Slower tappers → longer unit time")
    print()

def demo_practice_progression():
    print("📚 === LEARNING PROGRESSION ===")
    
    levels = [
        ("Beginner", ["E (·)", "T (-)", "A (·-)", "I (··)", "N (-·)"]),
        ("Basic Words", ["THE", "AND", "TO", "IT", "IS"]),
        ("Common Letters", ["S (···)", "H (····)", "R (·-·)", "D (-··)", "L (·-··)"]),
        ("Numbers", ["1 (·----)", "2 (··---)", "3 (···--)", "4 (····-)", "5 (·····)"]),
        ("Emergency", ["SOS (··· --- ···)", "HELP (···· · ·-·· ·--·)"]),
        ("Full Sentences", ["HELLO WORLD", "MORSE CODE IS FUN"]),
    ]
    
    for level, items in levels:
        print(f"🎓 {level:15} → {', '.join(items)}")
    print()

def demo_statistics():
    print("📊 === STATISTICS TRACKING ===")
    print("The system tracks your progress automatically:")
    print("   📈 Letters decoded correctly")
    print("   📝 Words completed")  
    print("   🎯 Total taps made")
    print("   ⚡ Dots vs dashes ratio")
    print("   🎪 Recognition accuracy %")
    print("   ⏰ Current timing calibration")
    print("   📋 Full decoded text history")
    print()
    print("💡 Press 'S' during operation to see live stats!")
    print()

def demo_audio_system():
    print("🔊 === AUDIO FEEDBACK ===")
    feedback = [
        ("Dot tap", "High beep (800Hz, 100ms)"),
        ("Dash tap", "Lower beep (600Hz, 300ms)"),  
        ("Letter complete", "Success tone (1000Hz, 150ms)"),
        ("Invalid tap", "Error buzz (300Hz, 200ms)")
    ]
    
    for event, sound in feedback:
        print(f"   {event:15} → {sound}")
    print()
    print("🎵 Audio helps you learn timing and confirms recognition!")
    print("🔇 Use --no-audio or press 'A' to toggle during use")
    print()

def show_settings_info():
    print("💾 === SETTINGS PERSISTENCE ===")
    print("Your preferences are automatically saved to 'morse_settings.json'")
    print()
    
    if os.path.exists("morse_settings.json"):
        try:
            with open("morse_settings.json", 'r') as f:
                settings = json.load(f)
            print("📋 CURRENT SAVED SETTINGS:")
            for key, value in settings.items():
                print(f"   {key:20} = {value}")
        except Exception:
            print("❌ Could not read settings file")
    else:
        print("📝 No settings file found - will create on first run")
    
    print()
    print("🔄 Use --reset-settings to restore defaults")
    print()

def run_quick_test():
    print("🧪 === QUICK SYSTEM TEST ===")
    print("Testing core functionality...")
    
    tests = [
        ("Import modules", lambda: __import__('cv2') and __import__('numpy')),
        ("Audio system", lambda: check_audio()),
        ("Settings system", lambda: test_settings()),
        ("Morse decoder", lambda: test_morse_decode()),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name:20} {status}")
        except Exception as e:
            print(f"   {test_name:20} ❌ ERROR: {str(e)[:30]}")
    
    print()

def check_audio():
    try:
        import winsound
        return True
    except ImportError:
        return False

def test_settings():
    # Quick test of settings system
    test_file = "test_settings.json"
    try:
        settings = {"test": True}
        with open(test_file, 'w') as f:
            json.dump(settings, f)
        
        with open(test_file, 'r') as f:
            loaded = json.load(f)
        
        os.remove(test_file)
        return loaded.get("test") == True
    except Exception:
        return False

def test_morse_decode():
    # Test basic Morse decoding
    morse_map = {
        '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E'
    }
    
    test_cases = [
        ('.-', 'A'),
        ('-...', 'B'), 
        ('...', 'S'),
        ('---', 'O')
    ]
    
    for morse, expected in test_cases:
        if morse_map.get(morse) != expected:
            return False
    return True

def main():
    demo_intro()
    demo_controls()
    demo_command_line()
    demo_timing_guide()
    demo_practice_progression()
    demo_statistics()
    demo_audio_system()
    show_settings_info()
    run_quick_test()
    
    print("🎯 === READY TO START! ===")
    print("Run the main system with:")
    print("   python table_click_detector.py")
    print()
    print("🎓 Start with simple letters like E (·) and T (-)")
    print("🚀 Work up to words like 'THE' and 'SOS'")
    print("📈 Watch your statistics improve over time!")
    print()
    print("Happy Morse coding! 📻✨")

if __name__ == "__main__":
    main()