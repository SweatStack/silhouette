"""
PyScript Worker

Loads analysis scripts, detects dependencies, installs them via micropip,
and provides a run() function called from JavaScript.

Configuration (set in HTML before loading):
    window.ANALYSIS_SCRIPT = "main.py"  # Script to run
"""
from pyscript import display, window, document
from pyodide.ffi import create_proxy
import asyncio
import builtins
import re
import runtime

# Get script name from JS config, default to main.py
SCRIPT_NAME = getattr(window, 'ANALYSIS_SCRIPT', None) or 'main.py'

# ============================================
# Output Functions
# ============================================

def log(msg):
    """Log to browser console with prefix."""
    window.console.log(f"[PyScript] {msg}")


def set_status(state, text=""):
    """Update the status indicator in the UI."""
    window.setStatus(state, text)


def status(text):
    """Show a status message in the output panel (cleared on next output)."""
    document.getElementById("output-status").textContent = text


def output(*values):
    """Display values in the output panel. Handles text, DataFrames, figures, etc."""
    document.getElementById("output-status").textContent = ""
    for value in values:
        display(value, target="output", append=True)


def clear():
    """Clear the output panel."""
    document.getElementById("output-status").textContent = ""
    document.getElementById("output").innerHTML = ""


# ============================================
# Dependency Detection
# ============================================

def parse_pep723_deps(code):
    """Extract dependencies from PEP 723 script metadata block."""
    try:
        import tomllib
    except ImportError:
        return None

    pattern = r'(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$'
    matches = [m for m in re.finditer(pattern, code) if m.group('type') == 'script']
    if not matches:
        return None

    content = ''.join(
        line[2:] if line.startswith('# ') else line[1:]
        for line in matches[0].group('content').splitlines(keepends=True)
    )

    try:
        return tomllib.loads(content).get('dependencies')
    except Exception:
        return None


def detect_imports(code):
    """Detect required packages from import statements."""
    from pyodide.code import find_imports
    import pyodide_js

    try:
        imports = find_imports(code)
    except SyntaxError:
        return []

    pkg_map = pyodide_js._api._import_name_to_package_name.to_py()
    return [
        pkg_map.get(mod) or mod
        for mod in imports
        if not is_installed(mod) and '.' not in mod
    ]


def is_installed(module):
    """Check if a module is already available."""
    try:
        __import__(module)
        return True
    except ModuleNotFoundError:
        return False


async def install_deps(code):
    """Install dependencies declared in PEP 723 metadata or detected from imports."""
    import micropip

    deps = parse_pep723_deps(code) or detect_imports(code)
    if deps:
        log(f"Installing: {', '.join(deps)}")
        await micropip.install(deps, keep_going=True)
        log("Installed")

    return deps


def warmup_matplotlib():
    """Pre-initialize matplotlib to speed up first plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plt.close(fig)


# ============================================
# Script Execution
# ============================================

_script_code = None


def run(params):
    """Execute the analysis script with the given context.

    params: Any value from JavaScript - passed directly to ctx.params
    """
    set_status('running', 'analyzing')
    clear()

    if not _script_code:
        return output("Error: Script not loaded")

    # Populate runtime context
    runtime.ctx.output = output
    runtime.ctx.status = status
    runtime.ctx.api_key = window.localStorage.getItem("access_token")
    runtime.ctx.params = params

    try:
        exec(_script_code, {"__builtins__": builtins})
    except Exception as e:
        import traceback
        output(f"Error: {e}")
        output(traceback.format_exc())

    set_status('ready', 'ready')


# ============================================
# Initialization
# ============================================

async def init():
    global _script_code

    set_status('loading', 'loading 1/3')
    log(f"Loading {SCRIPT_NAME}...")
    try:
        with open(SCRIPT_NAME) as f:
            _script_code = f.read()
    except Exception as e:
        log(f"Error loading script: {e}")
        return set_status('loading', f'Error: {e}')

    set_status('loading', 'loading 2/3')
    try:
        await install_deps(_script_code)
    except Exception as e:
        log(f"Install error: {e}")
        return set_status('loading', f'Error: {e}')

    set_status('loading', 'loading 3/3')
    log("Warming up matplotlib...")
    try:
        warmup_matplotlib()
    except Exception as e:
        log(f"Warmup error: {e}")

    log("Ready")
    if hasattr(window, 'onPythonReady'):
        window.onPythonReady()


# ============================================
# Start
# ============================================

window.runAnalysis = create_proxy(run)
asyncio.ensure_future(init())
