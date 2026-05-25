"""Hugging Face Spaces entry point.

HF Spaces looks for `app.py` at the repo root by default.
This file just builds and launches the Gradio UI defined in src/web_app.py.
"""
from src.web_app import build_ui

demo = build_ui()

if __name__ == "__main__":
    demo.launch()
