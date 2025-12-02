#!/usr/bin/env python3
"""
Virtual Morse Code Practice Tool
Quick reference and practice phrases for the table-tap Morse system
"""

MORSE_CODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----', ' ': '/', '.': '.-.-.-', ',': '--..--',
    '?': '..--..', '!': '-.-.--', '-': '-....-', '/': '-..-.', '@': '.--.-.'
}

def text_to_morse(text):
    """Convert text to Morse code"""
    return ' '.join(MORSE_CODE.get(c.upper(), '?') for c in text)

def print_practice_words():
    """Print common practice words with their Morse patterns"""
    
    practice_words = [
        # Beginner (simple letters)
        "E", "T", "A", "I", "N", "S", "H", "R",
        
        # Easy words  
        "THE", "AND", "TO", "OF", "IN", "IT", "IS", "BE",
        
        # Common words
        "HELLO", "WORLD", "MORSE", "CODE", "TEST", "GOOD",
        
        # Numbers
        "123", "456", "789", "0",
        
        # Phrases
        "SOS", "HELP", "YES", "NO", "OK",
        
        # Sentences
        "HELLO WORLD", "TEST MESSAGE", "MORSE CODE"
    ]
    
    print("=== MORSE CODE PRACTICE GUIDE ===\\n")
    
    print("BASIC LETTERS (start here!):")
    for letter in "ETIANSHRDLU":
        morse = MORSE_CODE.get(letter, '?')
        print(f"  {letter}: {morse}")
    print()
    
    print("PRACTICE WORDS & PHRASES:")
    for word in practice_words:
        morse = text_to_morse(word)
        print(f"  {word:15} → {morse}")
    print()
    
    print("TIMING REMINDERS:")
    print("  · (dot)  = Quick tap (< 0.3 sec)")
    print("  - (dash) = Long tap (0.5-1.2 sec)")  
    print("  Letter gap = 0.6 sec pause")
    print("  Word gap = 1.4 sec pause")
    print()
    
    print("TIP: Start with single letters E and T, then try simple words!")

def interactive_practice():
    """Interactive practice mode"""
    print("\\n=== INTERACTIVE PRACTICE ===")
    print("Type a word/phrase and see its Morse code pattern:")
    print("(Type 'quit' to exit)\\n")
    
    while True:
        try:
            text = input("Enter text: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text:
                morse = text_to_morse(text)
                print(f"  Morse: {morse}")
                
                # Show timing breakdown for first few characters
                if len(text) <= 10:
                    print("  Timing breakdown:")
                    for char in text.upper():
                        if char == ' ':
                            print(f"    {char} → (1.4 sec pause for word gap)")
                        else:
                            pattern = MORSE_CODE.get(char, '?')
                            print(f"    {char} → {pattern}")
                print()
        except KeyboardInterrupt:
            break
    
    print("Happy tapping!")

if __name__ == "__main__":
    print_practice_words()
    
    try:
        interactive_practice()
    except KeyboardInterrupt:
        print("\\n\\nHappy Morse coding!")