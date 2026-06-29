import os
import sys
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend.predictor import Predictor

app = FastAPI(title="Forecast - Football Prediction MVP", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = Predictor()

class PredictParams(BaseModel):
    team_a: str
    team_b: str
    neutral: Optional[bool] = False
    home_team: Optional[str] = None
    altitude: Optional[int] = 0
    temperature: Optional[float] = 20.0
    humidity: Optional[int] = 50
    rest_days_a: Optional[int] = 7
    rest_days_b: Optional[int] = 7
    injuries_a: Optional[int] = 0
    injuries_b: Optional[int] = 0
    suspensions_a: Optional[int] = 0
    suspensions_b: Optional[int] = 0
    travel_a: Optional[int] = 0
    travel_b: Optional[int] = 0
    importance: Optional[str] = "normal"

@app.get("/api/teams")
def get_teams():
    return {"teams": predictor.teams, "count": len(predictor.teams)}

@app.get("/api/elo")
def get_elo():
    sorted_elo = sorted(predictor.elo_ratings.items(), key=lambda x: x[1], reverse=True)
    return {"ratings": [{"team": t, "elo": round(r, 1)} for t, r in sorted_elo[:100]]}

@app.get("/api/elo/{team}")
def get_team_elo(team: str):
    elo = predictor.elo_ratings.get(team, None)
    history = predictor.elo_history.get(team, [])
    if elo is None:
        return {"error": "Team not found"}
    return {"team": team, "elo": round(elo, 1), "history": history[-50:]}

@app.get("/api/predict")
def predict_get(
    team_a: str = Query(...),
    team_b: str = Query(...),
    neutral: bool = False,
    home_team: Optional[str] = None,
    altitude: int = 0,
    temperature: float = 20.0,
    humidity: int = 50,
    rest_days_a: int = 7,
    rest_days_b: int = 7,
    injuries_a: int = 0,
    injuries_b: int = 0,
    suspensions_a: int = 0,
    suspensions_b: int = 0,
    travel_a: int = 0,
    travel_b: int = 0,
    importance: str = "normal"
):
    params = {
        "team_a": team_a, "team_b": team_b, "neutral": neutral,
        "home_team": home_team, "altitude": altitude, "temperature": temperature,
        "humidity": humidity,
        "rest_days_a": rest_days_a, "rest_days_b": rest_days_b,
        "injuries_a": injuries_a, "injuries_b": injuries_b,
        "suspensions_a": suspensions_a, "suspensions_b": suspensions_b,
        "travel_a": travel_a, "travel_b": travel_b,
        "importance": importance
    }
    try:
        result = predictor.predict(team_a, team_b, params)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/predict")
def predict_post(params: PredictParams):
    try:
        result = predictor.predict(params.team_a, params.team_b, params.dict())
        return result
    except Exception as e:
        return {"error": str(e)}

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/{filename:path}")
def serve_static(filename: str):
    filepath = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
