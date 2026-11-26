"""Keyword arguments for starting helper processes.

On Windows, a console program started by a process that has no console
(for example a server launched in the background) can fail at startup with
0xC0000142 (3221225794, "DLL initialization failed"), because Windows can't
give it a console. CREATE_NO_WINDOW starts it without one. Our helpers only
talk through pipes, so they never need a console.
"""

from __future__ import annotations

import os
import subprocess

NO_CONSOLE: dict = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
