import sys
from pathlib import Path

ROOT = Path(__file__).parent
DASHBOARD = ROOT / "dashboard"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DASHBOARD))

from waitress import serve
from dashboard.app import server

if __name__ == "__main__":
    print("Starting IPL Analytics Platform...")
    print("Open http://0.0.0.0:8050 in your browser")
    serve(server, host="0.0.0.0", port=8050, threads=4)