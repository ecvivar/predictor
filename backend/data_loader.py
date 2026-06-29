import csv
import os
import urllib.request
from datetime import datetime, timedelta
from collections import defaultdict

from .utils import elo_update, tournament_k_factor

DATA_URL = "https://github.com/martj42/international_results/raw/master/results.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "results.csv")

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_PATH):
        return True
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        return True
    except Exception:
        return False

def load_matches():
    matches = []
    if not os.path.exists(DATA_PATH):
        if not download_data():
            return matches
    with open(DATA_PATH, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                date = row.get("date", "").strip()
                home = row.get("home_team", "").strip()
                away = row.get("away_team", "").strip()
                home_score = row.get("home_score", "").strip()
                away_score = row.get("away_score", "").strip()
                tournament = row.get("tournament", "").strip()
                country = row.get("country", "").strip()
                neutral = row.get("neutral", "").strip().lower() == "true"
                if not all([date, home, away, home_score, away_score]):
                    continue
                matches.append({
                    "date": date,
                    "home_team": home,
                    "away_team": away,
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                    "tournament": tournament,
                    "country": country,
                    "neutral": neutral
                })
            except (ValueError, KeyError):
                continue
    matches.sort(key=lambda m: m["date"])
    return matches

def compute_elo_history(matches):
    elo_ratings = {}
    elo_history = defaultdict(list)
    initial_elo = 1500
    for m in matches:
        home = m["home_team"]
        away = m["away_team"]
        if home not in elo_ratings:
            elo_ratings[home] = initial_elo
        if away not in elo_ratings:
            elo_ratings[away] = initial_elo
        k = tournament_k_factor(m["tournament"])
        new_home, new_away = elo_update(
            elo_ratings[home], elo_ratings[away],
            m["home_score"], m["away_score"],
            k=k
        )
        elo_ratings[home] = new_home
        elo_ratings[away] = new_away
        elo_history[home].append({"date": m["date"], "elo": new_home})
        elo_history[away].append({"date": m["date"], "elo": new_away})
    return elo_ratings, elo_history

def get_team_list(matches):
    teams = set()
    for m in matches:
        teams.add(m["home_team"])
        teams.add(m["away_team"])
    return sorted(teams)

def get_team_matches(matches, team):
    team_matches = []
    for m in matches:
        if m["home_team"] == team or m["away_team"] == team:
            team_matches.append(m)
    return team_matches

def get_recent_matches(matches, team, n=20):
    team_matches = get_team_matches(matches, team)
    return team_matches[-n:] if len(team_matches) >= n else team_matches

def get_recent_matches_by_months(matches, team, months=24):
    team_matches = get_team_matches(matches, team)
    if not team_matches:
        return []
    cutoff = datetime.now() - timedelta(days=months * 30)
    recent = []
    for m in reversed(team_matches):
        try:
            m_date = datetime.strptime(m["date"], "%Y-%m-%d")
            if m_date >= cutoff:
                recent.append(m)
        except ValueError:
            continue
    return list(reversed(recent))

def get_head_to_head(matches, team_a, team_b, n=20):
    h2h = []
    for m in matches:
        if (m["home_team"] == team_a and m["away_team"] == team_b) or \
           (m["home_team"] == team_b and m["away_team"] == team_a):
            h2h.append(m)
    return h2h[-n:] if len(h2h) > n else h2h

def get_elo_at_date(elo_history, team, date_str):
    if team not in elo_history:
        return 1500
    best = 1500
    for entry in elo_history[team]:
        if entry["date"] <= date_str:
            best = entry["elo"]
        else:
            break
    return best
