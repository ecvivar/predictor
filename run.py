import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "127.0.0.1")

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  FORECAST - Football Prediction MVP")
    print("  Modelo Predictivo de 16 Etapas")
    print(f"  Servidor en http://{HOST}:{PORT}")
    print("=" * 60)
    is_reload = os.environ.get("RENDER") != "true"
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=is_reload)
