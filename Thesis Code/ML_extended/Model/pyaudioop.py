# Re-export the workspace-root compatibility shim so ManimGL/PyDub imports
# succeed on Python 3.14 when running from this folder.

from pathlib import Path
import runpy

shim_path = Path(__file__).resolve().parents[2] / "pyaudioop.py"
runpy.run_path(str(shim_path), run_name=__name__)
