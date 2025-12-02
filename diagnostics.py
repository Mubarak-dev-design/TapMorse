#!/usr/bin/env python3
"""
Virtual Morse Code System - Troubleshooting & Diagnostics
Quick diagnostic tool to identify and fix common issues
"""

import cv2
import numpy as np
import os
import json
import sys

def check_system_requirements():
    """Check if all required components are available"""
    print("🔍 === SYSTEM DIAGNOSTICS ===")
    print()
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 6):
        print("   ⚠️  WARNING: Python 3.6+ recommended")
    else:
        print("   ✅ Python version OK")
    
    # Check OpenCV
    try:
        print(f"📹 OpenCV Version: {cv2.__version__}")
        print("   ✅ OpenCV available")
    except Exception as e:
        print("   ❌ OpenCV not available:", str(e))
        return False
    
    # Check NumPy
    try:
        print(f"🔢 NumPy Version: {np.__version__}")
        print("   ✅ NumPy available")
    except Exception as e:
        print("   ❌ NumPy not available:", str(e))
        return False
    
    # Check audio system
    try:
        import winsound
        print("🔊 Audio System: Windows Sound available")
        print("   ✅ Audio feedback will work")
    except ImportError:
        print("🔇 Audio System: Not available on this platform")
        print("   ⚠️  Audio feedback disabled")
    
    return True

def test_camera_access():
    """Test camera connectivity and basic functionality"""
    print("\n📹 === CAMERA DIAGNOSTICS ===")
    
    # Test multiple camera indices
    working_cameras = []
    for i in range(5):  # Check cameras 0-4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"   Camera {i}: ✅ Working ({width}x{height})")
                working_cameras.append(i)
            else:
                print(f"   Camera {i}: ⚠️  Opens but no frame")
            cap.release()
        else:
            print(f"   Camera {i}: ❌ Not available")
    
    if not working_cameras:
        print("\n❌ NO WORKING CAMERAS FOUND!")
        print("💡 Troubleshooting tips:")
        print("   • Check camera is connected and not used by other apps")
        print("   • Try different USB port")
        print("   • Check camera permissions in Windows settings")
        print("   • Restart the application")
        return False
    else:
        print(f"\n✅ Found {len(working_cameras)} working camera(s): {working_cameras}")
        return True

def test_opencv_features():
    """Test OpenCV features used by the system"""
    print("\n🔧 === OPENCV FEATURES TEST ===")
    
    try:
        # Test basic operations
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test color conversion
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print("   ✅ Color conversion working")
        
        # Test blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        print("   ✅ Gaussian blur working")
        
        # Test contours
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("   ✅ Contour detection working")
        
        # Test drawing
        cv2.rectangle(test_img, (10, 10), (50, 50), (0, 255, 0), 2)
        cv2.putText(test_img, "Test", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        print("   ✅ Drawing functions working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ OpenCV feature test failed: {e}")
        return False

def check_settings_system():
    """Test settings persistence system"""
    print("\n💾 === SETTINGS SYSTEM TEST ===")
    
    test_settings = {
        "test_mode": True,
        "unit_time": 0.2,
        "confidence": 0.1
    }
    
    try:
        # Test write
        with open("test_morse_settings.json", 'w') as f:
            json.dump(test_settings, f, indent=2)
        print("   ✅ Settings write successful")
        
        # Test read
        with open("test_morse_settings.json", 'r') as f:
            loaded = json.load(f)
        
        if loaded == test_settings:
            print("   ✅ Settings read successful")
        else:
            print("   ⚠️  Settings data mismatch")
        
        # Cleanup
        os.remove("test_morse_settings.json")
        print("   ✅ Settings cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Settings test failed: {e}")
        return False

def performance_test():
    """Test system performance for real-time operation"""
    print("\n⚡ === PERFORMANCE TEST ===")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("   ❌ No camera available for performance test")
            return False
        
        frame_times = []
        print("   📊 Testing frame processing speed...")
        
        for i in range(30):  # Test 30 frames
            start_time = cv2.getTickCount()
            
            ret, frame = cap.read()
            if not ret:
                break
                
            # Simulate processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            end_time = cv2.getTickCount()
            frame_time = (end_time - start_time) / cv2.getTickFrequency()
            frame_times.append(frame_time)
        
        cap.release()
        
        if frame_times:
            avg_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_time
            print(f"   📈 Average frame time: {avg_time*1000:.1f}ms")
            print(f"   🎬 Estimated FPS: {fps:.1f}")
            
            if fps >= 15:
                print("   ✅ Performance: Excellent for real-time operation")
            elif fps >= 10:
                print("   ⚠️  Performance: Good, may have slight delays")
            else:
                print("   ❌ Performance: Too slow for real-time operation")
                print("   💡 Try reducing camera resolution or closing other apps")
            
            return fps >= 10
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def run_full_diagnostic():
    """Run complete system diagnostic"""
    print("🎯 === VIRTUAL MORSE CODE SYSTEM DIAGNOSTICS ===")
    print("Checking system readiness...\n")
    
    tests = [
        ("System Requirements", check_system_requirements),
        ("Camera Access", test_camera_access), 
        ("OpenCV Features", test_opencv_features),
        ("Settings System", check_settings_system),
        ("Performance", performance_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 === DIAGNOSTIC SUMMARY ===")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🚀 System ready! You can run the Morse code detector.")
    elif passed >= len(results) - 1:
        print("⚠️  System mostly ready. Minor issues detected.")
    else:
        print("❌ System has significant issues. Check error messages above.")
    
    return passed >= len(results) - 1

def show_common_fixes():
    """Show solutions for common problems"""
    print("\n🛠️ === COMMON ISSUES & FIXES ===")
    
    issues = [
        ("Camera not detected", [
            "Check camera is plugged in and recognized by Windows",
            "Close other apps using the camera (Skype, Teams, etc.)",
            "Try different USB port or camera",
            "Check Windows camera privacy settings"
        ]),
        ("Poor click detection", [
            "Ensure good lighting on table surface",
            "Use solid colored table (avoid patterns)",
            "Reduce --confidence value (try 0.05)",
            "Recapture reference states with better hand positioning",
            "Minimize background movement"
        ]),
        ("Wrong Morse timing", [
            "Let auto-calibration learn your speed (tap 10+ times)",
            "Manually adjust --unit-time (0.1=fast, 0.3=slow)",
            "Practice consistent dot/dash durations",
            "Use audio feedback to improve timing"
        ]),
        ("No audio feedback", [
            "Check if running on Windows (required for winsound)",
            "Verify system audio is working",
            "Use --no-audio if audio causes issues"
        ])
    ]
    
    for issue, fixes in issues:
        print(f"\n🔧 {issue}:")
        for fix in fixes:
            print(f"   • {fix}")

if __name__ == "__main__":
    try:
        success = run_full_diagnostic()
        show_common_fixes()
        
        if success:
            print("\n🎉 Ready to run: python table_click_detector.py")
        else:
            print("\n🔧 Please fix the issues above before running the system")
            
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")