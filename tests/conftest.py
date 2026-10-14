import os
import sys

# never call gemini from the tests, and make "src" importable when pytest runs from the repo root
os.environ["MOCK_LLM"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
