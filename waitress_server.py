import sys, os
from pathlib import Path

ROOT      = Path(__file__).parent
DASHBOARD = ROOT / "dashboard"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DASHBOARD))

from waitress import serve
from dashboard.app import server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Starting IPL Analytics Platform on port {port}...")
    serve(server, host="0.0.0.0", port=port, threads=4)