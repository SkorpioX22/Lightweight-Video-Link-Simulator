"""Run the LVLS demo GUI with the HDZero digital link engine.

Same video, same UI as demo/gui/analog.py — only the engine changes.

Run: python demo/gui/hdzero.py
"""
import importlib.util
import os
import sys

os.environ["LVLS_ENGINE"] = "hdzero"

HERE = os.path.dirname(os.path.abspath(__file__))
ANALOG_PATH = os.path.join(HERE, "analog.py")

_spec = importlib.util.spec_from_file_location("lvls_demo_main", ANALOG_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main()
