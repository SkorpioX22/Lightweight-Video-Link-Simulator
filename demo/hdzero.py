"""Run the LVLS demo GUI with the HDZero digital link engine.

Same video, same UI as demo/gui/main.py — only the engine changes.

Run: python demo/hdzero.py
"""
import importlib.util
import os
import sys

os.environ["LVLS_ENGINE"] = "hdzero"

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_PATH = os.path.join(HERE, "gui", "main.py")

_spec = importlib.util.spec_from_file_location("lvls_demo_main", MAIN_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main()
