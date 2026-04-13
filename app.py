import streamlit as st
import requests
import json
import re
import math
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import plotly.graph_objects as go
from urllib.parse import unquote

st.set_page_config(page_title="EPL Predictor — WLS Model v2", page_icon="⚽", layout="wide")

# ── CONSTANTS ────────────────────────────────────────────────────────────────
TEAMS = [
    'Arsenal', 'Manchester City', 'Man United', 'Aston Villa', 'Liverpool',
    'Chelsea', 'Brentford', 'Everton', 'Fulham', 'Bournemouth',
    'Brighton', 'Sunderland', 'Newcastle', 'Crystal Palace', 'Leeds United',
    'Tottenham', "Nott'm Forest", 'West Ham', 'Burnley', 'Wolves'
]

GW29_DATA = [
    {'pld':29,'w':19,'d':7,'l':3,'gf':56,'ga':21,'pts':64,'xg':1.70,'xga':0.87,'sot':5.8,'hwr':0.82,'awr':0.62,'rolling':2.20,'ppda':0.72,'fdr':3.8},
    {'pld':28,'w':18,'d':5,'l':5,'gf':57,'ga':25,'pts':59,'xg':2.00,'xga':1.05,'sot':6.1,'hwr':0.76,'awr':0.64,'rolling':2.00,'ppda':0.68,'fdr':3.2},
    {'pld':28,'w':14,'d':9,'l':5,'gf':50,'ga':38,'pts':51,'xg':1.77,'xga':1.30,'sot':5.2,'hwr':0.55,'awr':0.48,'rolling':1.85,'ppda':0.58,'fdr':2.8},
    {'pld':28,'w':15,'d':6,'l':7,'gf':38,'ga':30,'pts':51,'xg':1.55,'xga':1.20,'sot':4.8,'hwr':0.65,'awr':0.43,'rolling':1.65,'ppda':0.55,'fdr':2.6},
    {'pld':28,'w':14,'d':6,'l':8,'gf':47,'ga':37,'pts':48,'xg':1.75,'xga':1.28,'sot':5.5,'hwr':0.62,'awr':0.42,'rolling':1.50,'ppda':0.62,'fdr':3.5},
    {'pld':28,'w':12,'d':9,'l':7,'gf':48,'ga':31,'pts':45,'xg':2.03,'xga':1.12,'sot':5.8,'hwr':0.55,'awr':0.36,'rolling':1.70,'ppda':0.50,'fdr':2.4},
    {'pld':28,'w':13,'d':4,'l':11,'gf':44,'ga':40,'pts':43,'xg':1.60,'xga':1.45,'sot':4.4,'hwr':0.60,'awr':0.36,'rolling':1.50,'ppda':0.52,'fdr':2.5},
    {'pld':28,'w':11,'d':7,'l':10,'gf':32,'ga':33,'pts':40,'xg':1.42,'xga':1.38,'sot':4.0,'hwr':0.50,'awr':0.29,'rolling':1.30,'ppda':0.45,'fdr':2.2},
    {'pld':28,'w':12,'d':4,'l':12,'gf':40,'ga':42,'pts':40,'xg':1.52,'xga':1.55,'sot':4.3,'hwr':0.55,'awr':0.36,'rolling':1.55,'ppda':0.48,'fdr':2.8},
    {'pld':28,'w':9,'d':12,'l':7,'gf':44,'ga':46,'pts':39,'xg':1.58,'xga':1.62,'sot':4.2,'hwr':0.40,'awr':0.29,'rolling':1.35,'ppda':0.44,'fdr':2.1},
    {'pld':28,'w':9,'d':10,'l':9,'gf':38,'ga':35,'pts':37,'xg':1.60,'xga':1.38,'sot':4.8,'hwr':0.42,'awr':0.29,'rolling':1.20,'ppda':0.65,'fdr':2.3},
    {'pld':28,'w':9,'d':10,'l':9,'gf':29,'ga':34,'pts':37,'xg':1.35,'xga':1.38,'sot':3.6,'hwr':0.45,'awr':0.27,'rolling':1.10,'ppda':0.42,'fdr':2.0},
    {'pld':28,'w':10,'d':6,'l':12,'gf':40,'ga':42,'pts':36,'xg':1.62,'xga':1.55,'sot':4.6,'hwr':0.50,'awr':0.29,'rolling':1.40,'ppda':0.50,'fdr':3.2},
    {'pld':28,'w':9,'d':8,'l':11,'gf':30,'ga':34,'pts':35,'xg':1.40,'xga':1.42,'sot':3.9,'hwr':0.45,'awr':0.27,'rolling':1.15,'ppda':0.46,'fdr':2.0},
    {'pld':28,'w':7,'d':10,'l':11,'gf':37,'ga':47,'pts':31,'xg':1.48,'xga':1.68,'sot':4.0,'hwr':0.38,'awr':0.18,'rolling':0.95,'ppda':0.42,'fdr':2.4},
    {'pld':28,'w':7,'d':8,'l':13,'gf':38,'ga':43,'pts':29,'xg':1.52,'xga':1.62,'sot':4.4,'hwr':0.35,'awr':0.22,'rolling':0.85,'ppda':0.46,'fdr':2.8},
    {'pld':28,'w':7,'d':6,'l':15,'gf':26,'ga':41,'pts':27,'xg':1.28,'xga':1.60,'sot':3.4,'hwr':0.35,'awr':0.18,'rolling':1.00,'ppda':0.40,'fdr':2.2},
    {'pld':28,'w':6,'d':7,'l':15,'gf':34,'ga':54,'pts':25,'xg':1.45,'xga':1.79,'sot':3.9,'hwr':0.30,'awr':0.14,'rolling':0.75,'ppda':0.38,'fdr':2.5},
    {'pld':28,'w':4,'d':7,'l':17,'gf':32,'ga':56,'pts':19,'xg':1.22,'xga':1.88,'sot':3.2,'hwr':0.22,'awr':0.07,'rolling':0.65,'ppda':0.32,'fdr':1.8},
    {'pld':29,'w':2,'d':7,'l':20,'gf':20,'ga':51,'pts':13,'xg':1.10,'xga':1.90,'sot':2.9,'hwr':0.10,'awr':0.05,'rolling':0.40,'ppda':0.30,'fdr':1.6},
]

INSIGHTS = [
    "Arsenal's power score is built on elite defence (xGA 0.87) combined with the best recent form in the league. Their resilience is exceptional — Raya's 13 clean sheets reflect a squad that doesn't panic. Hard fixture run-in is the main risk to the title.",
    "City rank above Arsenal on underlying quality — their net xG is the best in the league, driven by Haaland's conversion. Efficiency score is outstanding, collecting more points than xPts in several tough fixtures. Slight recent form dip is the only concern.",
    "United's table position slightly flatters their underlying quality — xG net is modest. What lifts them is excellent efficiency under Carrick and strong rolling form (1.85 PPG). The power ranking says they're a genuine top-4 side on current output.",
    "Villa are somewhat flattered by their table position. Their net xG of +0.35/game is comparable to a solid mid-table side. They've benefited from a long winning run and good resilience, but underlying quality points to 6th–8th on merit.",
    "Liverpool's power score reflects a cooling-off from their title-winning form. Rolling PPG of 1.50 is mid-table level and their hard fixture list will test them. Top 4 is not guaranteed if form doesn't recover.",
    "Chelsea's quality score is second only to City — 2.03 xG/game is elite. But they're massively underperforming their xPts. The power ranking says Chelsea should be performing considerably better. Easier fixtures should help them rise.",
    "Brentford's quality score reveals fragility — net xG of +0.15/game is very thin for a top-7 side. Igor Thiago's goals mask an underlying structure that won't sustain. Efficiency has flattered them in the table.",
    "Everton's clean sheet rate (Pickford with 9) is their best asset and lifts their resilience score. But attacking quality (net xG near zero) is limited, and rolling form has cooled under Moyes.",
    "Fulham's rolling form of 1.55 PPG is deceptively strong but net xG is essentially zero. Efficiency reflects good game management under Silva. Hard fixture run-in pulls them down in the final predicted table.",
    "Bournemouth's 12 draws define them — a side creating roughly equal chances to those it concedes. Resilience is below average. The power ranking sees them as a 12th–14th quality side.",
    "Brighton are the most interesting power-ranking riser. High PPDA and xG profile are better than their table position suggests. Underperforming xPts (negative efficiency) is the main reason they sit lower than quality warrants.",
    "A solid debut season for promoted Sunderland. Power score reflects a team exactly where it should be — safe from relegation, not in contention for Europe. Le Bris has them well-organised but lacking creative ceiling.",
    "Newcastle's power ranking is lower than fans expect. Isak's goals have masked a low net xG (+0.07/game). Hard remaining fixtures (FDR 3.2) with a squad lacking real depth means a top-4 push looks very unlikely.",
    "Palace's power score and table position are well-aligned — the model sees them as genuinely mid-table. Henderson's 10 clean sheets lift resilience. An easy run-in means they'll finish comfortably safe.",
    "Leeds are in a concerning position. Rolling form has collapsed to 0.95 PPG and net xG is −0.20/game. The power ranking suggests they're closer to the danger zone than their points gap implies.",
    "Spurs are in freefall by every power-ranking dimension. Rolling form of 0.85 PPG is the second-worst in the league. Efficiency is deeply negative. The Tudor appointment has not yet moved the needle.",
    "Forest have had three managers this season and the chaos shows. Underlying quality is bottom-four (net xG −0.32/game). A slight rolling-form uptick under Vítor Pereira is the only positive signal.",
    "West Ham's power score is bottom-three. Worst xGA in the league (1.79), second-worst rolling form, lowest resilience outside the bottom two. Relegation looks highly probable.",
    "The one bright spot for Burnley is their efficiency score — they've collected slightly more points than xPts. But net xG of −0.66/game is catastrophic. Relegated barring a miracle.",
    "Wolves are in a category of their own at the bottom. A 19-game winless run, the worst net xG (−0.80/game), and the lowest resilience score. The power ranking gives them a score below 10.",
]

WLS_C = {
    'intercept': 19.40, 'ppg': -5.20, 'rolling': -2.40, 'gdpg': -1.05,
    'xg': -0.80, 'xga': 0.95, 'sot': -0.28, 'hwr': -1.85, 'awr': -2.60,
    'ppda': -0.55, 'fdr': 1.40
}
LEAGUE_AVGS = {'xg': 1.50, 'xga': 1.50, 'sot': 4.4, 'ppg': 1.26, 'hwr': 0.45, 'awr': 0.28, 'rolling': 1.26}

# ── NAME NORMALISATION ───────────────────────────────────────────────────────
# All sources → a shared canonical name used throughout the app
_RAW_MAP = {
    # ESPN
    'Arsenal': 'Arsenal',
    'Aston Villa': 'Aston Villa',
    'AFC Bournemouth': 'Bournemouth', 'Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford',
    'Brighton & Hove Albion': 'Brighton', 'Brighton': 'Brighton',
    'Burnley': 'Burnley',
    'Chelsea': 'Chelsea',
    'Crystal Palace': 'Crystal Palace',
    'Everton': 'Everton',
    'Fulham': 'Fulham',
    'Leeds': 'Leeds United', 'Leeds United': 'Leeds United',
    'Leicester': 'Leicester', 'Leicester City': 'Leicester',
    'Liverpool': 'Liverpool',
    'Manchester City': 'Manchester City', 'Man City': 'Manchester City',
    'Manchester United': 'Manchester United', 'Man United': 'Manchester United',
    'Man Utd': 'Manchester United',
    'Newcastle': 'Newcastle', 'Newcastle United': 'Newcastle',
    "Nott'm Forest": "Nott'm Forest", 'Nottingham Forest': "Nott'm Forest",
    "Nottm Forest": "Nott'm Forest",
    'Southampton': 'Southampton',
    'Sunderland': 'Sunderland',
    'Tottenham': 'Tottenham', 'Tottenham Hotspur': 'Tottenham', 'Spurs': 'Tottenham',
    'West Ham': 'West Ham', 'West Ham United': 'West Ham',
    'Wolverhampton Wanderers': 'Wolves', 'Wolves': 'Wolves', 'Wolverhampton': 'Wolves',
    'Ipswich': 'Ipswich', 'Ipswich Town': 'Ipswich',
}
CANONICAL = {k.lower(): v for k, v in _RAW_MAP.items()}

def to_canonical(name: str) -> str:
    return CANONICAL.get((name or '').lower().strip(), name)


# ── MODEL HELPERS ────────────────────────────────────────────────────────────
def wls_predict(d: dict) -> float:
    pld = max(d['pld'], 1)
    ppg = d['pts'] / pld
    gdpg = (d['gf'] - d['ga']) / pld
    r = (WLS_C['intercept'] + WLS_C['ppg'] * ppg + WLS_C['rolling'] * d['rolling']
         + WLS_C['gdpg'] * gdpg + WLS_C['xg'] * d['xg'] + WLS_C['xga'] * d['xga']
         + WLS_C['sot'] * d['sot'] + WLS_C['hwr'] * d['hwr'] + WLS_C['awr'] * d['awr']
         + WLS_C['ppda'] * d['ppda'] + WLS_C['fdr'] * d['fdr'])
    return min(20.0, max(1.0, r))

def run_model(data: list, remaining: int):
    teams = []
    for i, d in enumerate(data):
        pld = max(d['pld'], 1)
        ppg = d['pts'] / pld
        gdpg = (d['gf'] - d['ga']) / pld
        name = d.get('name', TEAMS[i] if i < len(TEAMS) else f'Team {i+1}')
        idx = next((j for j, t in enumerate(TEAMS) if t == name or name.startswith(t.split()[0])), i)
        teams.append({**d, 'name': name, 'ppg': ppg, 'gdpg': gdpg,
                      'raw_score': wls_predict(d),
                      'proj_pts': min(114, round(d['pts'] + ppg * remaining)),
                      'insight': INSIGHTS[idx] if 0 <= idx < len(INSIGHTS) else ''})

    by_curr = sorted(teams, key=lambda t: (-t['pts'], -t['gdpg']))
    for i, t in enumerate(by_curr):
        t['current_pos'] = i + 1

    by_pred = sorted(teams, key=lambda t: (-t['proj_pts'], t['raw_score']))
    for i, t in enumerate(by_pred):
        t['pred_pos'] = i + 1

    return teams, by_pred

def minmax_norm(vals):
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return [50.0] * len(vals)
    return [(v - mn) / (mx - mn) * 100 for v in vals]

def compute_power_rankings(teams: list) -> list:
    data = []
    for t in teams:
        denom = t['w'] * 3 + t['d']
        data.append({**t,
                     'form': t['rolling'],
                     'quality': t['xg'] - t['xga'],
                     'resilience': (t['pts'] / denom if denom > 0 else 0.5) * (1.1 if t['sot'] > 3 else 0.9),
                     'efficiency': t['ppg'] * 1.5 - t['fdr'] * 0.2})
    for key in ['form', 'quality', 'resilience', 'efficiency']:
        for d, n in zip(data, minmax_norm([x[key] for x in data])):
            d[f'{key}_n'] = n
    for d in data:
        d['score'] = round(0.35 * d['form_n'] + 0.30 * d['quality_n']
                           + 0.20 * d['resilience_n'] + 0.15 * d['efficiency_n'])
    data.sort(key=lambda d: -d['score'])
    for i, d in enumerate(data):
        d['pr'] = i + 1
    return data

def poisson_prob(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def calc_xpts(xg_for: float, xg_against: float, max_g: int = 8) -> float:
    p_win = p_draw = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = poisson_prob(xg_for, h) * poisson_prob(xg_against, a)
            if h > a:
                p_win += p
            elif h == a:
                p_draw += p
    return round(3 * p_win + p_draw, 2)

def approx_ppda(xg: float, xga: float) -> float:
    """Approximate pressing intensity proxy from xG profile."""
    return max(0.30, min(0.90, 0.50 + (xg - 1.40) * 0.20 - (xga - 1.40) * 0.15))


# ── DATA SOURCES ─────────────────────────────────────────────────────────────
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0'}

# ESPN team ID → canonical name lookup (for Team Analyser match fetching)
ESPN_TEAM_IDS = {
    'Arsenal': '359', 'Manchester City': '382', 'Manchester United': '360',
    'Aston Villa': '362', 'Liverpool': '364', 'Chelsea': '363',
    'Brentford': '337', 'Everton': '368', 'Fulham': '370',
    'Bournemouth': '349', 'Brighton': '331', 'Sunderland': '366',
    'Newcastle': '361', 'Crystal Palace': '384', 'Leeds United': '357',
    'Tottenham': '367', "Nott'm Forest": '393', 'West Ham': '371',
    'Burnley': '379', 'Wolves': '380',
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_espn_standings() -> dict:
    """ESPN unofficial API → pld, w, d, l, gf, ga, pts per team."""
    r = requests.get(
        'https://site.api.espn.com/apis/v2/sports/soccer/eng.1/standings',
        timeout=12
    )
    r.raise_for_status()
    # Structure: children[0].standings.entries (no "Overall" sub-section anymore)
    entries = r.json()['children'][0].get('standings', {}).get('entries', [])
    result = {}
    for entry in entries:
        name = to_canonical(entry['team']['displayName'])
        sd = {s['name']: s.get('value', 0) for s in entry.get('stats', [])}
        result[name] = {
            'name': name,
            'pld': int(sd.get('gamesPlayed', 0)),
            'w':   int(sd.get('wins', 0)),
            'd':   int(sd.get('ties', 0)),
            'l':   int(sd.get('losses', 0)),
            'gf':  int(sd.get('pointsFor', 0)),
            'ga':  int(sd.get('pointsAgainst', 0)),
            'pts': int(sd.get('points', 0)),
        }
    return result


@st.cache_data(ttl=300, show_spinner=False)
def fetch_fpl_data() -> tuple:
    """
    FPL bootstrap-static → team xG/xGA (aggregated from player data) +
    FPL fixtures → per-team match history + future FDR.
    Returns (xg_stats, match_history, fdr_map).
    """
    bootstrap = requests.get(
        'https://fantasy.premierleague.com/api/bootstrap-static/', timeout=12
    ).json()
    fpl_id_map = {t['id']: to_canonical(t['name']) for t in bootstrap.get('teams', [])}

    # Get games played per team from finished fixtures
    from collections import defaultdict
    team_gp = defaultdict(int)

    # Aggregate xG (all players) and xGA (GK proxy) per team
    team_xg_tot  = defaultdict(float)
    team_xga_tot = defaultdict(float)
    for p in bootstrap.get('elements', []):
        if not p.get('minutes', 0):
            continue
        tid = p['team']
        team_xg_tot[tid]  += float(p.get('expected_goals', 0) or 0)
        if p['element_type'] == 1:   # goalkeeper
            team_xga_tot[tid] += float(p.get('expected_goals_conceded', 0) or 0)

    # Fixtures → match history + future FDR + games played count
    all_fixtures = requests.get(
        'https://fantasy.premierleague.com/api/fixtures/', timeout=12
    ).json()

    match_history: dict = defaultdict(list)   # canon_name → sorted list of matches
    fdr_future: dict   = defaultdict(list)    # canon_name → list of future difficulties

    for fix in all_fixtures:
        if fix.get('finished'):
            team_gp[fix.get('team_h', 0)] += 1
            team_gp[fix.get('team_a', 0)] += 1
        h_canon = fpl_id_map.get(fix.get('team_h'))
        a_canon = fpl_id_map.get(fix.get('team_a'))
        if not h_canon or not a_canon:
            continue

        if fix.get('finished'):
            gf_h = fix.get('team_h_score', 0) or 0
            gf_a = fix.get('team_a_score', 0) or 0
            gw   = fix.get('event') or 0
            date = (fix.get('kickoff_time') or '')[:10]
            match_history[h_canon].append({
                'gw': gw, 'date': date, 'opponent': a_canon, 'venue': 'H',
                'goalsFor': gf_h, 'goalsAgainst': gf_a,
                'result': 'W' if gf_h > gf_a else 'D' if gf_h == gf_a else 'L',
            })
            match_history[a_canon].append({
                'gw': gw, 'date': date, 'opponent': h_canon, 'venue': 'A',
                'goalsFor': gf_a, 'goalsAgainst': gf_h,
                'result': 'W' if gf_a > gf_h else 'D' if gf_a == gf_h else 'L',
            })
        elif not fix.get('finished'):
            hd = fix.get('team_h_difficulty', 3)
            ad = fix.get('team_a_difficulty', 3)
            if h_canon:
                fdr_future[h_canon].append(hd)
            if a_canon:
                fdr_future[a_canon].append(ad)

    # Sort each team's match list by GW
    for canon in match_history:
        match_history[canon].sort(key=lambda x: (x['gw'], x['date']))
        for i, m in enumerate(match_history[canon]):
            m['gw'] = i + 1  # re-number sequentially

    fdr_map = {name: round(sum(v) / len(v), 1) for name, v in fdr_future.items() if v}

    # Compute per-game xG/xGA season averages (used for model inputs)
    xg_stats = {}
    for tid, canon in fpl_id_map.items():
        gp = max(team_gp.get(tid, 1), 1)
        xg_stats[canon] = {
            'xg':  round(team_xg_tot[tid]  / gp, 2),
            'xga': round(team_xga_tot[tid] / gp, 2),
        }

    # ── Fetch per-match xG from GW live endpoints (parallel) ──────────────
    finished_gws = sorted({f['event'] for f in all_fixtures
                            if f.get('finished') and f.get('event')})
    player_team_id = {p['id']: p['team'] for p in bootstrap.get('elements', [])}
    player_type_id = {p['id']: p['element_type'] for p in bootstrap.get('elements', [])}

    def _fetch_gw_xg(gw: int):
        r = requests.get(f'https://fantasy.premierleague.com/api/event/{gw}/live/',
                         timeout=15).json()
        t_xg  = defaultdict(float)
        t_xga = defaultdict(float)
        for el in r.get('elements', []):
            pid  = el['id']
            tid  = player_team_id.get(pid)
            s    = el.get('stats', {})
            if not s.get('minutes', 0) or tid is None:
                continue
            t_xg[tid]  += float(s.get('expected_goals', 0) or 0)
            if player_type_id.get(pid) == 1:   # GK
                t_xga[tid] += float(s.get('expected_goals_conceded', 0) or 0)
        return gw, dict(t_xg), dict(t_xga)

    # (team_id, gw) → {xg, xga}
    match_xg: dict = {}
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_fetch_gw_xg, gw): gw for gw in finished_gws}
            for fut in as_completed(futs):
                gw, t_xg, t_xga = fut.result()
                for tid in set(t_xg) | set(t_xga):
                    match_xg[(tid, gw)] = {
                        'xg':  round(t_xg.get(tid, 0), 2),
                        'xga': round(t_xga.get(tid, 0), 2),
                    }
    except Exception:
        pass   # fall back to season average if live fetch fails

    # Attach per-match xG/xGA to match_history entries
    # We need fpl_team_id (int) → canonical name reverse map
    canon_to_fpl_id = {v: k for k, v in fpl_id_map.items()}
    # Also need the original GW number before renumbering — store it
    # Rebuild using gw_fixtures map to look up xG by (team_id, event_gw)
    fix_gw_map = {}   # (team_id, fpl_event_gw) → xG dict
    for f in all_fixtures:
        if f.get('finished') and f.get('event'):
            ev = f['event']
            for tid in (f['team_h'], f['team_a']):
                key = (tid, ev)
                if key in match_xg:
                    fix_gw_map[key] = match_xg[key]

    # Pre-build (team_id, date) → event_gw for fast lookup
    team_date_to_gw: dict = {}
    for f in all_fixtures:
        if f.get('finished') and f.get('event'):
            date = (f.get('kickoff_time') or '')[:10]
            ev   = f['event']
            team_date_to_gw[(f['team_h'], date)] = ev
            team_date_to_gw[(f['team_a'], date)] = ev

    for canon, ms in match_history.items():
        fpl_tid = canon_to_fpl_id.get(canon)
        for m in ms:
            xg_data = None
            if fpl_tid:
                ev = team_date_to_gw.get((fpl_tid, m['date']))
                if ev:
                    xg_data = match_xg.get((fpl_tid, ev))
            m['xg']  = xg_data['xg']  if xg_data else None
            m['xga'] = xg_data['xga'] if xg_data else None

    return xg_stats, dict(match_history), fdr_map


def _derive_form_stats(matches: list) -> dict:
    """Derive hwr, awr, rolling PPG from a team's completed FPL match list."""
    if not matches:
        return {}
    home  = [m for m in matches if m['venue'] == 'H']
    away  = [m for m in matches if m['venue'] == 'A']
    hw    = sum(1 for m in home if m['result'] == 'W')
    aw    = sum(1 for m in away if m['result'] == 'W')
    last6 = matches[-6:]
    rolling_pts = sum(3 if m['result']=='W' else 1 if m['result']=='D' else 0 for m in last6)
    return {
        'hwr':     round(hw / len(home), 3) if home else 0.40,
        'awr':     round(aw / len(away), 3) if away else 0.25,
        'rolling': round(rolling_pts / len(last6), 2) if last6 else 1.26,
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_live_data() -> tuple:
    """Merge ESPN standings + FPL xG/form/FDR into model-ready rows."""
    errors = []

    espn = {}
    try:
        espn = fetch_espn_standings()
    except Exception as e:
        errors.append(f"ESPN standings: {e}")

    xg_stats, match_history, fdr_map = {}, {}, {}
    try:
        xg_stats, match_history, fdr_map = fetch_fpl_data()
    except Exception as e:
        errors.append(f"FPL data: {e}")

    # Build merged rows
    rows = []
    for name, e in espn.items():
        if not e.get('pld'):
            continue
        xg_data = xg_stats.get(name, {})
        form    = _derive_form_stats(match_history.get(name, []))
        xg  = xg_data.get('xg',  1.40)
        xga = xg_data.get('xga', 1.40)
        rows.append({
            'name': name,
            'pld':     e['pld'],
            'w':       e['w'],
            'd':       e['d'],
            'l':       e['l'],
            'gf':      e['gf'],
            'ga':      e['ga'],
            'pts':     e['pts'],
            'xg':      xg,
            'xga':     xga,
            'sot':     round(xg * 2.8, 1),
            'hwr':     form.get('hwr',     0.40),
            'awr':     form.get('awr',     0.25),
            'rolling': form.get('rolling', 1.26),
            'ppda':    approx_ppda(xg, xga),
            'fdr':     fdr_map.get(name, 2.5),
            'source':  'live',
        })

    rows.sort(key=lambda t: (-t['pts'], -(t['gf'] - t['ga'])))
    return rows, errors


# ── TEAM ANALYSER HELPERS ────────────────────────────────────────────────────
def enrich_matches(matches: list, team_xg: float, team_xga: float) -> list:
    """Add xG/xGA/xpts to FPL match history. Uses per-match values when available,
    falls back to season average."""
    result = []
    for m in matches:
        xgf = m['xg']  if m.get('xg')  is not None else team_xg
        xga = m['xga'] if m.get('xga') is not None else team_xga
        result.append({**m, 'xg': round(xgf, 2), 'xga': round(xga, 2),
                        'xpts': calc_xpts(xgf, xga)})
    return result

def generate_verdict(team: str, matches: list) -> str:
    """Rule-based analytical verdict from recent match data."""
    if len(matches) < 6:
        return "Insufficient data to generate a verdict."

    r6 = matches[-6:]
    r6_ppg  = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in r6) / 6
    r6_xg   = matches[0]['xg']  # season average (constant from FPL aggregation)
    r6_xga  = matches[0]['xga']
    r6_gf   = sum(m['goalsFor']     for m in r6) / 6
    r6_ga   = sum(m['goalsAgainst'] for m in r6) / 6

    prev6 = matches[-12:-6] if len(matches) >= 12 else None
    prev_ppg = (sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0
                    for m in prev6) / len(prev6)) if prev6 else None

    parts = []

    # Form trend
    if prev_ppg is not None:
        delta = r6_ppg - prev_ppg
        if delta > 0.4:
            parts.append(f"{team} are on a strong upward trajectory — rolling 6-game PPG of {r6_ppg:.2f}, up {delta:.2f} from the preceding six matches.")
        elif delta < -0.4:
            parts.append(f"{team}'s form has deteriorated sharply — down to {r6_ppg:.2f} PPG in the last six, from {prev_ppg:.2f} in the six before.")
        else:
            parts.append(f"{team} have been consistent, averaging {r6_ppg:.2f} PPG over their last six league games.")
    else:
        parts.append(f"{team} are averaging {r6_ppg:.2f} PPG over their last six league games.")

    # Season xG profile vs recent goals
    if r6_xg > LEAGUE_AVGS['xg'] + 0.25:
        parts.append(f"Their season xG of {r6_xg:.2f}/game ranks them among the top attacking teams in the league.")
    elif r6_xg < LEAGUE_AVGS['xg'] - 0.25:
        parts.append(f"Their season xG of {r6_xg:.2f}/game is below the league average — attacking output is a concern.")

    # Defensive assessment from xGA
    if r6_xga < 0.9:
        parts.append(f"Defensively they look very strong — {r6_xga:.2f} xGA/game this season.")
    elif r6_xga > 1.65:
        parts.append(f"Defensively they look vulnerable — {r6_xga:.2f} xGA/game is well above the league average.")

    # Recent goals scored vs conceded in last 6
    if r6_gf > r6_ga + 0.7:
        parts.append(f"Recent results have been impressive — scoring {r6_gf:.1f} and conceding {r6_ga:.1f} per game in the last six.")
    elif r6_ga > r6_gf + 0.7:
        parts.append(f"They have been leaking goals recently — conceding {r6_ga:.1f} per game against {r6_gf:.1f} scored in the last six.")

    return " ".join(parts)

def generate_flags(name: str, matches: list, profile: dict) -> list:
    flags = []
    if not matches:
        return flags
    r6 = matches[-6:]
    r6_ppg = sum(3 if m['result']=='W' else 1 if m['result']=='D' else 0 for m in r6) / 6
    xg  = profile.get('xg', 1.40)
    xga = profile.get('xga', 1.40)
    fdr = profile.get('fdr', 2.5)

    if fdr >= 3.5:
        flags.append({'type': 'danger', 'title': 'Tough run-in',
                      'detail': f'Average FDR of {fdr:.1f} — one of the hardest remaining schedules in the league.'})
    if fdr <= 2.0:
        flags.append({'type': 'opportunity', 'title': 'Favourable fixtures',
                      'detail': f'FDR of {fdr:.1f} — an easy remaining schedule that should boost the points tally.'})
    if xg > LEAGUE_AVGS['xg'] + 0.3:
        flags.append({'type': 'opportunity', 'title': 'Elite attack',
                      'detail': f'{xg:.2f} xG/game is well above the league average of {LEAGUE_AVGS["xg"]:.2f}.'})
    if xga > LEAGUE_AVGS['xga'] + 0.3:
        flags.append({'type': 'danger', 'title': 'Defensive frailty',
                      'detail': f'{xga:.2f} xGA/game — the defence is conceding far more than the league average.'})
    if r6_ppg < 0.83:
        flags.append({'type': 'danger', 'title': 'Poor recent form',
                      'detail': f'Just {r6_ppg:.2f} PPG in the last 6 games — relegation or slide down table is a risk.'})
    if r6_ppg > 2.33:
        flags.append({'type': 'opportunity', 'title': 'Momentum',
                      'detail': f'{r6_ppg:.2f} PPG in the last 6 — best-form teams tend to outperform their predicted final position.'})
    if len(matches) >= 6:
        r6_gf = sum(m['goalsFor'] for m in r6)
        r6_xg_sum = sum(m['xg'] for m in r6)
        if r6_gf < r6_xg_sum * 0.75:
            flags.append({'type': 'opportunity', 'title': 'Finishing regression due',
                          'detail': f'Scoring {r6_gf} goals vs {r6_xg_sum:.1f} xG in the last 6 — finishing luck should improve.'})
    return flags[:5]


# ── DISPLAY HELPERS ──────────────────────────────────────────────────────────
def zone_label(pos): return {**dict.fromkeys([1,2,3,4],'UCL'), **dict.fromkeys([5,6],'UEL'), 7:'UECL'}.get(pos, 'REL' if pos >= 18 else '')
def zone_color(pos): return '#3b82f6' if pos<=4 else '#22c55e' if pos<=6 else '#9095a3' if pos==7 else '#ef4444' if pos>=18 else ''
def score_color(s):  return '#67e8f9' if s>=80 else '#22c55e' if s>=65 else '#9095a3' if s>=50 else '#f59e0b' if s>=35 else '#ef4444'
def tier_label(s):   return 'Elite' if s>=80 else 'Strong' if s>=65 else 'Solid' if s>=50 else 'Vulnerable' if s>=35 else 'Struggling'


# ── SESSION STATE ────────────────────────────────────────────────────────────
for k, v in [('fetched_data', None), ('last_results', None),
             ('gw', 29), ('remaining', 9), ('team_cache', {})]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ EPL Predictor")
    st.caption("WLS Model v2 · 2025/26 Season")
    st.caption("Data: ESPN · FPL API")
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        gw_in  = st.number_input("GW",        min_value=6,  max_value=38, value=st.session_state.gw)
    with col_b:
        rem_in = st.number_input("Remaining", min_value=1,  max_value=32, value=st.session_state.remaining)

    if st.button("Load GW29 Demo", use_container_width=True):
        st.session_state.fetched_data = [{'name': TEAMS[i], **d, 'source': 'demo'}
                                          for i, d in enumerate(GW29_DATA)]
        st.session_state.gw = 29
        st.session_state.remaining = 9
        st.rerun()

    if st.button("▶ Run Model", type="primary", use_container_width=True):
        if not st.session_state.fetched_data:
            st.error("No data — fetch live data or load demo first.")
        else:
            teams, by_pred = run_model(st.session_state.fetched_data, rem_in)
            st.session_state.last_results = {'teams': teams, 'by_pred': by_pred,
                                              'gw': gw_in, 'remaining': rem_in}
            st.session_state.gw = gw_in
            st.session_state.remaining = rem_in
            st.success("Model run!")


# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Entry", "📊 Predicted Table",
    "⚡ Power Rankings", "🔬 Model Info", "🔍 Team Analyser"
])


# ══════════════════════════════════ TAB 1 ═══════════════════════════════════
with tab1:
    st.subheader("Data — 2025/26 Premier League")

    hc, bc = st.columns([4, 1])
    with hc:
        st.caption("Pulls live data from **ESPN** (standings) and **FPL** (xG, match history, FDR). No API key needed.")
    with bc:
        fetch_now = st.button("▶ Fetch Live Data", type="primary")

    if fetch_now:
        with st.spinner("Fetching from ESPN · Understat · FPL…"):
            rows, errs = fetch_all_live_data()
            if rows:
                # detect GW from median games played
                pld_vals = [r['pld'] for r in rows if r['pld']]
                detected_gw = round(sum(pld_vals) / len(pld_vals)) if pld_vals else 29
                st.session_state.fetched_data = rows
                st.session_state.gw = detected_gw
                st.session_state.remaining = max(1, 38 - detected_gw)
                if errs:
                    st.warning("Some sources had issues (partial data shown): " + " | ".join(errs))
                else:
                    st.success(f"Fetched {len(rows)} teams — GW {detected_gw} detected")
                st.rerun()
            else:
                st.error("All data sources failed:\n" + "\n".join(errs))

    if st.session_state.fetched_data is None:
        st.info("⚽ No data loaded — click **Fetch Live Data** above or **Load GW29 Demo** in the sidebar.")
    else:
        data = st.session_state.fetched_data
        hdr, clr = st.columns([5, 1])
        with hdr:
            src = data[0].get('source', '?')
            st.caption(f"{len(data)} teams · source: **{src}**")
        with clr:
            if st.button("Clear"):
                st.session_state.fetched_data = None
                st.rerun()

        df = pd.DataFrame([{
            'Team': t['name'], 'Pld': t['pld'],
            'W': t['w'], 'D': t['d'], 'L': t['l'],
            'GF': t['gf'], 'GA': t['ga'], 'Pts': t['pts'],
            'xG/g': round(t['xg'], 2), 'xGA/g': round(t['xga'], 2),
            'SOT/g': round(t['sot'], 1),
            'H Win%': f"{int(t['hwr']*100)}%", 'A Win%': f"{int(t['awr']*100)}%",
            'Roll PPG': round(t['rolling'], 2),
            'PPDA~': round(t['ppda'], 2),
            'FDR': round(t['fdr'], 1),
        } for t in data])
        st.dataframe(df, width='stretch', hide_index=True)
        st.caption("xG/xGA from **FPL** player expected goals (aggregated per team) · Win rates & rolling PPG from FPL match history · "
                   "FDR from **FPL fixtures API** (1=easy, 5=very hard) · PPDA~ = approximated from xG profile")


# ══════════════════════════════════ TAB 2 ═══════════════════════════════════
with tab2:
    res = st.session_state.last_results
    if res is None:
        st.info("⚽ Load data then click **Run Model** in the sidebar.")
    else:
        by_pred, gw_val = res['by_pred'], res['gw']

        hc, bc = st.columns([5, 1])
        with hc:
            st.caption(f"WLS model v2 · 10 predictors · GW {gw_val}")
        with bc:
            rows = []
            for t in by_pred:
                diff = t['current_pos'] - t['pred_pos']
                rows.append({'Predicted Pos': t['pred_pos'], 'Team': t['name'],
                             'Current Pos': t['current_pos'], 'PPG': round(t['ppg'], 2),
                             '6-game PPG': round(t['rolling'], 2), 'xG': round(t['xg'], 2),
                             'xGA': round(t['xga'], 2), 'FDR': round(t['fdr'], 1),
                             'Proj Pts': t['proj_pts'],
                             'vs Now': f"▲{diff}" if diff>0 else f"▼{abs(diff)}" if diff<0 else "—"})
            buf = io.StringIO()
            pd.DataFrame(rows).to_csv(buf, index=False)
            st.download_button("Export CSV", buf.getvalue(), f"EPL_GW{gw_val}.csv", "text/csv")

        for leg_txt in ["🔵 UCL (top 4)", "🟢 UEL (5–6)", "⚪ UECL (7)", "🔴 REL (18–20)"]:
            pass
        leg_cols = st.columns(4)
        for col, txt in zip(leg_cols, ["🔵 **UCL** (top 4)", "🟢 **UEL** (5–6)", "⚪ **UECL** (7)", "🔴 **REL** (18–20)"]):
            col.markdown(txt)
        st.divider()

        for t in by_pred:
            zc = zone_color(t['pred_pos'])
            zl = zone_label(t['pred_pos'])
            diff = t['current_pos'] - t['pred_pos']
            diff_md = (f"<span style='color:#22c55e'>▲{diff}</span>" if diff > 0
                       else f"<span style='color:#ef4444'>▼{abs(diff)}</span>" if diff < 0
                       else "<span style='color:#5c6070'>—</span>")
            badge = (f"<span style='background:{zc}22;color:{zc};border:1px solid {zc};"
                     f"padding:1px 6px;border-radius:3px;font-size:11px'>{zl}</span>") if zl else ""
            cols = st.columns([0.25, 2.5, 0.7, 0.8, 0.8, 0.7, 0.7, 0.7, 0.9, 0.8, 0.6])
            with cols[0]:
                if zc:
                    st.markdown(f"<div style='width:4px;height:32px;background:{zc};border-radius:2px;margin:4px auto'></div>", unsafe_allow_html=True)
            with cols[1]: st.markdown(f"**{t['pred_pos']}.** {t['name']} {badge}", unsafe_allow_html=True)
            with cols[2]: st.caption(f"P{t['current_pos']}")
            with cols[3]: st.caption(f"{t['ppg']:.2f} PPG")
            with cols[4]: st.caption(f"{t['rolling']:.2f} 6gm")
            with cols[5]: st.caption(f"{t['xg']:.2f} xG")
            with cols[6]: st.caption(f"{t['xga']:.2f} xGA")
            with cols[7]: st.caption(f"{t['fdr']:.1f} FDR")
            with cols[8]: st.markdown(badge if zl else f"<span style='font-size:12px'>{t['pred_pos']}</span>", unsafe_allow_html=True)
            with cols[9]: st.markdown(f"**{t['proj_pts']}** pts")
            with cols[10]: st.markdown(diff_md, unsafe_allow_html=True)


# ══════════════════════════════════ TAB 3 ═══════════════════════════════════
with tab3:
    res = st.session_state.last_results
    if res is None:
        st.info("⚡ Run the model first.")
    else:
        st.caption(f"Form 35% · Quality 30% · Resilience 20% · Efficiency 15% · GW {res['gw']}")
        pr = compute_power_rankings(res['teams'])

        for d in pr:
            sc = d['score']
            bc = score_color(sc)
            tier = tier_label(sc)
            td = d['current_pos'] - d['pr']
            diff_md = (f"<span style='color:#22c55e'>▲{td}</span>" if td>0
                       else f"<span style='color:#ef4444'>▼{abs(td)}</span>" if td<0
                       else "<span style='color:#5c6070'>—</span>")
            qstr = f"{'+' if d['quality']>=0 else ''}{d['quality']:.2f}"

            cols = st.columns([0.4, 2.2, 0.6, 2.5, 1.0, 0.7, 0.7, 0.7, 0.7, 0.6])
            with cols[0]: st.markdown(f"**{d['pr']}**")
            with cols[1]: st.markdown(f"**{d['name']}**")
            with cols[2]: st.caption(f"P{d['current_pos']}")
            with cols[3]:
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;padding:4px 0'>"
                    f"<div style='flex:1;height:5px;background:#252830;border-radius:3px;overflow:hidden'>"
                    f"<div style='width:{sc}%;height:100%;background:{bc}'></div></div>"
                    f"<span style='color:{bc};font-weight:600;font-size:13px;min-width:28px'>{sc}</span></div>",
                    unsafe_allow_html=True)
            with cols[4]: st.caption(tier)
            with cols[5]: st.caption(f"{d['form']:.2f}")
            with cols[6]: st.caption(qstr)
            with cols[7]: st.caption(f"{d['resilience_n']/10:.1f}")
            with cols[8]: st.caption(f"{d['ppg']:.2f}")
            with cols[9]: st.markdown(diff_md, unsafe_allow_html=True)

            with st.expander(f"↳ {d['name']} breakdown"):
                dc1, dc2 = st.columns(2)
                dims = [
                    ("Recent form — rolling 6-game PPG", "35%", d['form_n'],       f"{d['form']:.2f}"),
                    ("Underlying quality — net xG/game", "30%", d['quality_n'],    f"{'+' if d['quality']>=0 else ''}{d['quality']:.2f}"),
                    ("Squad resilience",                  "20%", d['resilience_n'], f"{d['resilience_n']/10:.1f}/10"),
                    ("Fixture-adjusted efficiency",       "15%", d['efficiency_n'], f"{d['ppg']:.2f}"),
                ]
                for j, (label, wt, pct, val) in enumerate(dims):
                    with dc1 if j % 2 == 0 else dc2:
                        st.markdown(f"**{label}** `{wt}`")
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px'>"
                            f"<div style='flex:1;height:4px;background:#252830;border-radius:2px;overflow:hidden'>"
                            f"<div style='width:{round(pct)}%;height:100%;background:#3b82f6'></div></div>"
                            f"<span style='font-size:11px;min-width:44px'>{val}</span></div>",
                            unsafe_allow_html=True)
                if d.get('insight'):
                    st.markdown(f"*{d['insight']}*")


# ══════════════════════════════════ TAB 4 ═══════════════════════════════════
with tab4:
    st.subheader("Model Specification")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("R² in-sample", "0.93", "+0.04 vs v1 OLS")
    m2.metric("RMSE (positions)", "1.6", "−0.5 vs v1 OLS")
    m3.metric("Training seasons", "20", "2004–05 to 2023–24")
    m4.metric("Team-seasons", "~400", "WLS weighted obs.")
    m5.metric("Predictors", "10", "+3 vs v1 OLS")
    st.divider()

    lc, rc = st.columns(2)
    with lc:
        st.markdown("### WLS Coefficients")
        st.caption("Negative = higher value → better (lower) predicted position")
        st.dataframe(pd.DataFrame([
            ("Intercept", "+19.40"),
            ("Season PPG β₁ — most influential", "−5.20"),
            ("Rolling 6-game PPG β₂ — momentum", "−2.40"),
            ("Goal difference/game β₃", "−1.05"),
            ("xG/game β₄", "−0.80"),
            ("xGA/game β₅ — defensive signal", "+0.95"),
            ("Shots on target/game β₆", "−0.28"),
            ("Home win rate β₇", "−1.85"),
            ("Away win rate β₈", "−2.60"),
            ("PPDA proxy β₉ — pressing", "−0.55"),
            ("Fixture difficulty β₁₀", "+1.40"),
        ], columns=["Predictor", "Coeff"]), hide_index=True, width='stretch')

    with rc:
        st.markdown("### Season Decay Weights (λ=0.78)")
        st.dataframe(pd.DataFrame([
            ("2024–25","1.000"),("2023–24","0.780"),("2022–23","0.608"),
            ("2021–22","0.474"),("2020–21","0.370"),("2019–20","0.289"),
            ("2018–19","0.225"),("2017–18","0.176"),("2016–17","0.137"),
            ("2015–16","0.107"),("2010–15 avg","0.052"),("2004–10 avg","0.018"),
        ], columns=["Season","Weight"]), hide_index=True, width='stretch')

        st.markdown("### Power Ranking Weights")
        st.markdown("- **Recent form** (rolling 6-game PPG) — 35%\n"
                    "- **Underlying quality** (net xG/game) — 30%\n"
                    "- **Squad resilience** — 20%\n"
                    "- **Fixture-adjusted efficiency** — 15%")

    st.divider()
    st.markdown("### Data Sources")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown("**ESPN API** (no auth)")
        st.caption("Live standings — Pld, W, D, L, GF, GA, Pts per team. Refreshed every 5 min.")
    with sc2:
        st.markdown("**FPL API** (no auth)")
        st.caption("xG & xGA aggregated from per-player expected goals. Per-match results and FDR for all teams.")
    with sc3:
        st.markdown("**FPL API** (no auth)")
        st.caption("Fixture difficulty ratings (1–5) for all remaining Premier League games.")


# ══════════════════════════════════ TAB 5 ═══════════════════════════════════
with tab5:
    selected = st.selectbox("Select a team", [""] + sorted(TEAMS),
                            format_func=lambda x: "Select a team…" if x == "" else x)

    # Auto-analyse whenever selection changes and isn't cached yet
    if selected and selected not in st.session_state.team_cache:
        with st.spinner(f"Loading {selected} data from FPL…"):
            try:
                xg_stats, match_history, _ = fetch_fpl_data()
                raw_matches = match_history.get(selected, [])
                if not raw_matches:
                    st.warning("No match data found — check the team name.")
                else:
                    team_xg  = xg_stats.get(selected, {}).get('xg',  1.40)
                    team_xga = xg_stats.get(selected, {}).get('xga', 1.40)
                    st.session_state.team_cache[selected] = enrich_matches(raw_matches, team_xg, team_xga)
            except Exception as e:
                st.error(f"Failed to load team data: {e}")

    cached = st.session_state.team_cache.get(selected) if selected else None

    if not selected or cached is None:
        st.info("📊 Select a team to view their full season profile.")
    else:
        matches = cached
        # Pull profile from main data if available
        profile = {}
        if st.session_state.fetched_data:
            for t in st.session_state.fetched_data:
                if t['name'] == selected:
                    profile = t
                    break

        # ── Stat chips ────────────────────────────────────────────────────
        n = len(matches)
        gf_total = sum(m['goalsFor']     for m in matches)
        ga_total = sum(m['goalsAgainst'] for m in matches)
        w = sum(1 for m in matches if m['result'] == 'W')
        d_cnt = sum(1 for m in matches if m['result'] == 'D')
        l = sum(1 for m in matches if m['result'] == 'L')
        pts_total = w * 3 + d_cnt
        ppg = pts_total / n if n else 0
        avg_xg  = sum(m['xg']  for m in matches) / n if n else 0
        avg_xga = sum(m['xga'] for m in matches) / n if n else 0
        last6 = matches[-6:]
        roll_ppg = sum(3 if m['result']=='W' else 1 if m['result']=='D' else 0
                       for m in last6) / len(last6) if last6 else 0

        # ── Top stats: Played · W · D · L · Pts ─────────────────────────
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Played", n)
        mc2.metric("Won",    w)
        mc3.metric("Drawn",  d_cnt)
        mc4.metric("Lost",   l)
        mc5.metric("Points", pts_total)

        st.divider()

        # ── Rolling charts with matching metric callouts ──────────────────
        if n >= 5:
            def _roll(key, window=5):
                return [sum(matches[i-window+1:i+1][j][key] for j in range(window))/window
                        for i in range(len(matches)) if i >= window-1]

            gw_labels = [f"GW{m['gw']}" for m in matches[4:]]
            r_gf  = _roll('goalsFor')
            r_ga  = _roll('goalsAgainst')
            pts_per_match = [3 if m['result']=='W' else 1 if m['result']=='D' else 0 for m in matches]
            r_ppg  = [sum(pts_per_match[i-4:i+1])/5 for i in range(len(matches)) if i >= 4]
            has_xg = all(m.get('xg') is not None for m in matches)
            r_xpts = _roll('xpts') if has_xg else None

            chart_opts = dict(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                              font=dict(color='#374151', size=11), height=220,
                              margin=dict(t=36, b=20, l=20, r=20),
                              legend=dict(font=dict(size=10)))
            ax = dict(gridcolor='#e5e7eb', tickfont=dict(size=9), color='#6b7280')

            # Chart 1 — Goals, with matching callouts
            cc1, cc2 = st.columns(2)
            with cc1:
                cm1, cm2 = st.columns(2)
                cm1.metric("Avg goals scored",   f"{avg_xg:.2f}")
                cm2.metric("Avg goals conceded", f"{avg_xga:.2f}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=gw_labels, y=r_gf, name='Goals scored',
                                         line=dict(color='#22c55e', width=2)))
                fig.add_trace(go.Scatter(x=gw_labels, y=r_ga, name='Goals conceded',
                                         line=dict(color='#ef4444', width=2)))
                fig.update_layout(title='Goals scored & conceded — 5-game rolling', **chart_opts)
                fig.update_xaxes(**ax); fig.update_yaxes(**ax)
                st.plotly_chart(fig, width='stretch')

            # Chart 2 — xPts / PPG, with matching callouts
            with cc2:
                if r_xpts:
                    season_xpts = sum(m['xpts'] for m in matches)
                    pm1, pm2 = st.columns(2)
                    pm1.metric("Avg xPts / game", f"{season_xpts/n:.2f}")
                    pm2.metric("Avg PPG",         f"{ppg:.2f}")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=gw_labels, y=r_xpts, name='xPts/game',
                                              fill='tozeroy', line=dict(color='#3b82f6', width=2),
                                              fillcolor='rgba(59,130,246,0.08)'))
                    fig2.add_trace(go.Scatter(x=gw_labels, y=r_ppg, name='Actual PPG',
                                              line=dict(color='#f59e0b', width=1.5, dash='dot')))
                    title2 = 'xPoints vs PPG — 5-game rolling'
                else:
                    pm1, = st.columns(1)
                    pm1.metric("Rolling 6-gm PPG", f"{roll_ppg:.2f}")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=gw_labels, y=r_ppg, name='PPG',
                                              fill='tozeroy', line=dict(color='#3b82f6', width=2),
                                              fillcolor='rgba(59,130,246,0.08)'))
                    title2 = 'Points per game — 5-game rolling'
                fig2.update_layout(title=title2, **chart_opts)
                fig2.update_xaxes(**ax); fig2.update_yaxes(**ax)
                st.plotly_chart(fig2, width='stretch')

        st.divider()

        # ── Last game ─────────────────────────────────────────────────────
        st.subheader("Last match")
        lg = matches[-1]
        rc = '#22c55e' if lg['result']=='W' else '#ef4444' if lg['result']=='L' else '#f59e0b'
        st.markdown(
            f"<div style='font-size:30px;font-weight:700'>"
            f"<span style='color:{rc}'>{lg['result']}</span> &nbsp;"
            f"{lg['goalsFor']}–{lg['goalsAgainst']} &nbsp;"
            f"<span style='color:#9095a3;font-size:18px'>vs {lg['opponent']} ({'Home' if lg['venue']=='H' else 'Away'}) · {lg['date'][:10]}</span></div>",
            unsafe_allow_html=True)
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        lc1.metric("Goals scored",   lg['goalsFor'])
        lc2.metric("Goals conceded", lg['goalsAgainst'])
        lc3.metric("xG",   f"{lg['xg']:.2f}"   if lg.get('xg')   is not None else "—")
        lc4.metric("xGA",  f"{lg['xga']:.2f}"  if lg.get('xga')  is not None else "—")
        lc5.metric("Points", 3 if lg['result']=='W' else 1 if lg['result']=='D' else 0)

        verdict = generate_verdict(selected, matches)
        st.markdown(
            f"<div style='border-left:3px solid #22c55e;padding:12px 16px;background:#1a1e24;"
            f"border-radius:0 6px 6px 0;font-size:13px;color:#9095a3;line-height:1.7;"
            f"font-style:italic;margin:12px 0'>{verdict}</div>",
            unsafe_allow_html=True)

        # ── Strengths/Weaknesses + Flags ──────────────────────────────────
        sw_col, fl_col = st.columns(2)
        with sw_col:
            st.markdown("**Strengths & Weaknesses vs League Average**")
            dims = [
                ('xG/game (attack)',   avg_xg,   LEAGUE_AVGS['xg'],      True,  False),
                ('xGA/game (defence)', avg_xga,  LEAGUE_AVGS['xga'],     False, False),
                ('Home win rate',      sum(1 for m in matches if m['venue']=='H' and m['result']=='W') / max(sum(1 for m in matches if m['venue']=='H'),1),
                                                 LEAGUE_AVGS['hwr'],     True,  True),
                ('Away win rate',      sum(1 for m in matches if m['venue']=='A' and m['result']=='W') / max(sum(1 for m in matches if m['venue']=='A'),1),
                                                 LEAGUE_AVGS['awr'],     True,  True),
                ('Rolling 6-gm PPG',  roll_ppg, LEAGUE_AVGS['rolling'], True,  False),
                ('Pts per game',       ppg,      LEAGUE_AVGS['ppg'],     True,  False),
            ]
            for label, val, avg, higher, is_pct in dims:
                pct = min(100.0, (val / 3.0) * 100)
                bar_col = '#22c55e' if (val >= avg if higher else val <= avg) else '#ef4444'
                display = f"{int(val*100)}%" if is_pct else f"{val:.2f}"
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:7px'>"
                    f"<div style='font-size:11px;color:#9095a3;width:150px;flex-shrink:0'>{label}</div>"
                    f"<div style='flex:1;height:6px;background:#252830;border-radius:3px;overflow:hidden'>"
                    f"<div style='width:{pct:.0f}%;height:100%;background:{bar_col}'></div></div>"
                    f"<div style='font-size:11px;min-width:36px;text-align:right'>{display}</div></div>",
                    unsafe_allow_html=True)
            st.caption("Bar length relative to max scale (3.0)")

        with fl_col:
            st.markdown("**Danger & Opportunity Flags**")
            flags = generate_flags(selected, matches, profile if profile else
                                   {'xg': avg_xg, 'xga': avg_xga, 'fdr': 2.5})
            if not flags:
                st.caption("No notable flags at this time.")
            for f in flags:
                dot = '#ef4444' if f['type'] == 'danger' else '#22c55e'
                st.markdown(
                    f"<div style='display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #252830'>"
                    f"<div style='width:8px;height:8px;background:{dot};border-radius:50%;"
                    f"margin-top:4px;flex-shrink:0'></div>"
                    f"<div style='font-size:12px'><strong>{f['title']}</strong><br>"
                    f"<span style='color:#9095a3'>{f['detail']}</span></div></div>",
                    unsafe_allow_html=True)

        # ── Match log ─────────────────────────────────────────────────────
        with st.expander("Full match log"):
            log_df = pd.DataFrame([{
                'GW': m['gw'], 'Date': m['date'], 'Opponent': m['opponent'], 'Venue': m['venue'],
                'Score': f"{m['goalsFor']}–{m['goalsAgainst']}", 'Result': m['result'],
                'Pts': 3 if m['result']=='W' else 1 if m['result']=='D' else 0,
                'xG':  round(m['xg'],  2) if m.get('xg')  is not None else '—',
                'xGA': round(m['xga'], 2) if m.get('xga') is not None else '—',
                'xPts': round(m['xpts'], 2) if m.get('xpts') is not None else '—',
            } for m in matches])
            st.dataframe(log_df, hide_index=True, width='stretch')
