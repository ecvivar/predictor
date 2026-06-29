import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  FORECAST - Football Prediction MVP")
    print("  Modelo Predictivo de 16 Etapas")
    print(f"  Servidor en http://{HOST}:{PORT}")
    print("=" * 60)
    reload_enabled = os.environ.get("RENDER") != "true"
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=reload_enabled)
