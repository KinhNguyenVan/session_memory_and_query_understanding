"""
Validate project setup: Python version, dependencies, GEMINI_API_KEY, test data.
"""

import os
import sys

from dotenv import load_dotenv


def check_setup():
    """Check if the project is set up correctly (Gemini-only)."""
    print("Checking project setup...\n")
    issues = []

    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required (current: {}.{})".format(
            sys.version_info.major, sys.version_info.minor))
    else:
        print("✓ Python {}.{}.{}".format(
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro))

    required = ["pydantic", "tiktoken", "rich", "dotenv", "langchain_google_genai", "langchain_core"]
    for name in required:
        try:
            if name == "dotenv":
                __import__("dotenv")
            else:
                __import__(name.replace("-", "_"))
            print("✓ {} installed".format(name))
        except ImportError:
            issues.append("Missing package: {}".format(name))

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("✓ GEMINI_API_KEY or GOOGLE_API_KEY found")
    else:
        issues.append("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")

    test_files = [
        "test_data/conversation_1_long.jsonl",
        "test_data/conversation_2_ambiguous.jsonl",
        "test_data/conversation_3_mixed.jsonl",
        "test_data/conversation_4_technical.jsonl",
    ]
    for path in test_files:
        if os.path.exists(path):
            print("✓ {} exists".format(path))
        else:
            issues.append("Missing: {}".format(path))

    print("\n" + "=" * 50)
    if issues:
        print("❌ Setup issues:")
        for i in issues:
            print("  • {}".format(i))
        print("\nFix: pip install -r requirements.txt ; set GEMINI_API_KEY in .env")
        return False
    print("✓ Setup OK. Run: python demos/demo_cli.py")
    return True


if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)
