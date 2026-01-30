"""
Quick validation script to check if setup is correct.
"""

import sys
import os
from dotenv import load_dotenv

def check_setup():
    """Check if the project is set up correctly"""
    print("Checking project setup...\n")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required (current: {}.{})".format(
            sys.version_info.major, sys.version_info.minor
        ))
    else:
        print("✓ Python version: {}.{}.{}".format(
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro
        ))
    
    # Check dependencies
    required_packages = [
        "openai", "pydantic", "tiktoken", "rich", "dotenv"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == "dotenv":
                __import__("dotenv")
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            missing_packages.append(package)
            issues.append(f"Missing package: {package}")
    
    # Check environment variables
    load_dotenv()
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    
    if has_openai:
        print("✓ OPENAI_API_KEY found")
    if has_anthropic:
        print("✓ ANTHROPIC_API_KEY found")
    if has_gemini:
        print("✓ GEMINI_API_KEY or GOOGLE_API_KEY found")
    
    if not has_openai and not has_anthropic and not has_gemini:
        issues.append("No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY in .env file")
    
    # Check test data
    test_files = [
        "test_data/conversation_1_long.jsonl",
        "test_data/conversation_2_ambiguous.jsonl",
        "test_data/conversation_3_mixed.jsonl",
        "test_data/conversation_4_technical.jsonl"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"✓ {test_file} exists")
        else:
            issues.append(f"Missing test file: {test_file}")
    
    # Summary
    print("\n" + "=" * 50)
    if issues:
        print("❌ Setup issues found:")
        for issue in issues:
            print(f"  • {issue}")
        print("\nTo fix:")
        if missing_packages:
            print(f"  Run: pip install -r requirements.txt")
        if not has_openai and not has_anthropic:
            print("  Create .env file with your API key")
        return False
    else:
        print("✓ Setup looks good! You're ready to run the demos.")
        print("\nTry running:")
        print("  python demo_cli.py --load-log test_data/conversation_1_long.jsonl")
        print("  python demo_flows.py")
        return True

if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)
