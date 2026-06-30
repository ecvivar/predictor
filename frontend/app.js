// ─── STATE ───────────────────────────────────────────────────────────────────
const state = {
  teams: [], teamA: '', teamB: '', teamA_api: '', teamB_api: '',
  teamAMap: {}, result: null,
  charts: {}, loadingStep: 0, loadTimer: null
};

// ─── INIT ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadTeams();
  setupEvents();
  updateControls();
});

function setupEvents() {
  document.getElementById('predictBtn').addEventListener('click', runPrediction);
  document.getElementById('teamA').addEventListener('change', e => {
    state.teamA_api = e.target.value;
    state.teamA = state.teamAMap[state.teamA_api] || state.teamA_api;
  });
  document.getElementById('teamB').addEventListener('change', e => {
    state.teamB_api = e.target.value;
    state.teamB = state.teamAMap[state.teamB_api] || state.teamB_api;
  });
  document.querySelectorAll('.ctrl').forEach(el => el.addEventListener('input', updateControls));
}

// ─── CONTROL DISPLAY ────────────────────────────────────────────────────────
function updateControls() {
  const map = [
    ['altitude','altV', v => `${v} m`],
    ['temperature','tempV', v => `${v}°C`],
    ['humidity','humV', v => `${v}%`],
    ['restA','restAV', v => `${v}d`],
    ['restB','restBV', v => `${v}d`],
    ['injA','injAV', v => v],
    ['injB','injBV', v => v],
    ['susA','susAV', v => v],
    ['susB','susBV', v => v],
    ['travelA','travAV', v => `${Number(v).toLocaleString()} km`],
    ['travelB','travBV', v => `${Number(v).toLocaleString()} km`],
  ];
  map.forEach(([inp, out, fmt]) => {
    const i = document.getElementById(inp), o = document.getElementById(out);
    if (i && o) o.textContent = fmt(i.value);
  });
}

// ─── LOAD TEAMS ──────────────────────────────────────────────────────────────
async function loadTeams() {
  try {
    const r = await fetch('/api/teams');
    const d = await r.json();
    state.teamAMap = {};
    d.teams.forEach(t => { state.teamAMap[t.en] = t.es; });
    state.teams = d.teams.map(t => t.en);
    populateSelects();
  } catch(e) {
    showError('No se pudo cargar equipos. Verifica que el servidor esté corriendo.');
  }
}

function populateSelects() {
  const a = document.getElementById('teamA'), b = document.getElementById('teamB');
  const popular = [
    'Argentina','Brazil','France','England','Spain','Germany','Italy',
    'Netherlands','Portugal','Belgium','Uruguay','Croatia','Colombia',
    'Morocco','Mexico','USA','Japan','Senegal','Korea Republic',
    'Australia','Chile','Ecuador','Peru','Switzerland','Denmark',
    'Sweden','Poland','Turkey','Serbia','Algeria','Nigeria'
  ];
  [a, b].forEach(sel => {
    popular.forEach(t => {
      if (state.teams.includes(t)) {
        const o = document.createElement('option');
        o.value = t; o.textContent = state.teamAMap[t] || t; sel.appendChild(o);
      }
    });
    const sep = document.createElement('option');
    sep.disabled = true; sep.textContent = '──────────';
    sel.appendChild(sep);
    state.teams.forEach(t => {
      const o = document.createElement('option');
      o.value = t; o.textContent = state.teamAMap[t] || t; sel.appendChild(o);
    });
  });
  if (state.teams.includes('Argentina')) a.value = 'Argentina';
  if (state.teams.includes('Brazil')) b.value = 'Brazil';
  const ae = a.value, be = b.value;
  state.teamA_api = ae; state.teamB_api = be;
  state.teamA = state.teamAMap[ae] || ae;
  state.teamB = state.teamAMap[be] || be;
}

// ─── PREDICTION ──────────────────────────────────────────────────────────────
async function runPrediction() {
  state.teamA_api = document.getElementById('teamA').value;
  state.teamB_api = document.getElementById('teamB').value;
  state.teamA = state.teamAMap[state.teamA_api] || state.teamA_api;
  state.teamB = state.teamAMap[state.teamB_api] || state.teamB_api;
  if (!state.teamA_api || !state.teamB_api) { showError('Selecciona ambos equipos'); return; }
  if (state.teamA_api === state.teamB_api) { showError('Los equipos deben ser diferentes'); return; }
  hideError();
  showLoading(true);
  startLoadingAnimation();

  const p = new URLSearchParams({
    team_a: state.teamA_api, team_b: state.teamB_api,
    neutral: document.getElementById('neutral').checked,
    altitude: document.getElementById('altitude').value,
    temperature: document.getElementById('temperature').value,
    humidity: document.getElementById('humidity').value,
    rest_days_a: document.getElementById('restA').value,
    rest_days_b: document.getElementById('restB').value,
    injuries_a: document.getElementById('injA').value,
    injuries_b: document.getElementById('injB').value,
    suspensions_a: document.getElementById('susA').value,
    suspensions_b: document.getElementById('susB').value,
    travel_a: document.getElementById('travelA').value,
    travel_b: document.getElementById('travelB').value,
    importance: document.getElementById('importance').value
  });

  try {
    const r = await fetch(`/api/predict?${p}`);
    if (!r.ok) { showError('Error del servidor: ' + r.status); showLoading(false); return; }
    const d = await r.json();
    if (d.error) { showError(d.error); showLoading(false); return; }
    state.result = d;
    renderAll();
    document.getElementById('results').style.display = 'flex';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch(e) {
    showError('Error de conexión. Asegúrate de ejecutar "python run.py" en la terminal.');
  } finally {
    showLoading(false);
    stopLoadingAnimation();
  }
}

function startLoadingAnimation() {
  const steps = ['ls1','ls2','ls3','ls4','ls5'];
  let i = 0;
  steps.forEach(s => document.getElementById(s)?.classList.remove('active'));
  document.getElementById(steps[0])?.classList.add('active');
  state.loadTimer = setInterval(() => {
    document.getElementById(steps[i])?.classList.remove('active');
    i = Math.min(i + 1, steps.length - 1);
    document.getElementById(steps[i])?.classList.add('active');
  }, 1200);
}

function stopLoadingAnimation() {
  if (state.loadTimer) clearInterval(state.loadTimer);
}

// ─── RENDER ALL ───────────────────────────────────────────────────────────────
function renderAll() {
  if (!state.result) return;
  renderHero();
  renderStage16();
  renderStage1();
  renderStage2();
  renderStage3();
  renderStage4();
  renderStage5();
  renderStage6();
  renderStage7();
  renderStage8();
  renderStage9();
  renderStage10();
  renderStage11();
  renderStage12();
  renderStage13();
  renderStage14();
  renderStage15();
}

// ─── HELPERS ─────────────────────────────────────────────────────────────────
const pct = (v, d=1) => `${(v*100).toFixed(d)}%`;
const fmt = (v, d=2) => typeof v === 'number' ? v.toFixed(d) : v;
const sign = v => v >= 0 ? `+${v}` : `${v}`;
const mono = v => `<span style="font-family:'JetBrains Mono',monospace">${v}</span>`;

function statRow(label, value, cls='') {
  return `<div class="stat-row"><span class="stat-label">${label}</span><span class="stat-value ${cls}">${value}</span></div>`;
}

function teamTitle(name, side) {
  return `<div class="team-section-title ${side}">${name}</div>`;
}

function ifrBar(score, color) {
  const c = color === 'blue' ? 'var(--blue)' : 'var(--red)';
  return `
  <div class="ifr-bar-wrap">
    <div class="ifr-bar-label"><span>IFR</span><span style="color:${c};font-weight:700">${score}/100</span></div>
    <div class="ifr-bar-track"><div class="ifr-bar-fill" style="width:${score}%;background:${c}"></div></div>
  </div>`;
}

// ─── HERO ─────────────────────────────────────────────────────────────────────
function renderHero() {
  const mc = state.result.stage_11;
  const s2 = state.result.stage_2;
  if (!mc) return;
  const p = mc.probabilities;
  const wa = (p.win_a*100).toFixed(1), wd = (p.draw*100).toFixed(1), wb = (p.win_b*100).toFixed(1);
  const eloA = s2?.team_a?.elo || '?';
  const eloB = s2?.team_b?.elo || '?';
  const tierA = s2?.team_a?.tier || '';
  const tierB = s2?.team_b?.tier || '';

  document.getElementById('heroTeams').innerHTML = `
    <div class="hero-team">
      <div class="hero-team-name team-a">${state.teamA}</div>
      <div class="hero-team-elo">Elo ${eloA} · ${tierA}</div>
      <div class="hero-team-prob team-a">${wa}%</div>
    </div>
    <div class="hero-center">
      <div style="font-size:12px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px">Empate</div>
      <div class="hero-draw-pct">${wd}%</div>
    </div>
    <div class="hero-team" style="text-align:right">
      <div class="hero-team-name team-b">${state.teamB}</div>
      <div class="hero-team-elo">Elo ${eloB} · ${tierB}</div>
      <div class="hero-team-prob team-b">${wb}%</div>
    </div>
  `;

  document.getElementById('probTrack').innerHTML = `
    <div class="prob-seg prob-seg-a" style="width:${wa}%">${wa}%</div>
    <div class="prob-seg prob-seg-draw" style="width:${wd}%">${wd}%</div>
    <div class="prob-seg prob-seg-b" style="width:${wb}%">${wb}%</div>
  `;

  const oddA = (1/p.win_a).toFixed(2);
  const oddD = (1/p.draw).toFixed(2);
  const oddB = (1/p.win_b).toFixed(2);
  document.getElementById('probLabels').innerHTML = `
    <span style="color:var(--blue)">${state.teamA}</span>
    <span>Empate</span>
    <span style="color:var(--red)">${state.teamB}</span>
  `;
  document.getElementById('probOdds').innerHTML = `
    <span class="odds-chip" style="color:var(--blue)">1&nbsp;&nbsp;@${oddA}</span>
    <span class="odds-chip" style="color:var(--gold)">X&nbsp;&nbsp;@${oddD}</span>
    <span class="odds-chip" style="color:var(--red)">2&nbsp;&nbsp;@${oddB}</span>
  `;

  const s16 = state.result.stage_16;
  const lam = state.result.stage_8;
  document.getElementById('heroVerdict').innerHTML = mc.interpretacion || 
    `${state.teamA}: ${wa}% | Empate: ${wd}% | ${state.teamB}: ${wb}% · Goles esperados: ${fmt(lam?.lambda_a)} – ${fmt(lam?.lambda_b)}`;
}

// ─── STAGE 1 ─────────────────────────────────────────────────────────────────
function renderStage1() {
  const s = state.result.stage_1;
  if (!s) return;
  const a = s.team_a, b = s.team_b, h = s.head_to_head;

  function generalHtml(t, data, cls) {
    return `
      ${teamTitle(t, cls)}
      ${statRow('Partidos jugados', mono(data.general.total))}
      ${statRow('V / E / D', `<span class="sv-green">${data.general.wins}</span> / ${data.general.draws} / <span class="sv-red">${data.general.losses}</span>`)}
      ${statRow('GF / GA / DG', mono(`${data.general.gf} / ${data.general.ga} / ${sign(data.general.gd)}`))}
      ${statRow('Win %', mono(data.general.win_pct + '%'), 'sv-blue')}
      ${statRow('Puntos/partido', mono(data.general.points_per_game), cls === 'a' ? 'sv-blue' : 'sv-red')}
    `;
  }

  function recentHtml(recent) {
    return `
      <div style="margin-top:12px">
        <div class="methodology" style="margin-top:0;padding:0;background:none;border:none;margin-bottom:6px;font-style:normal;font-size:11px;color:var(--text-muted)">FORMA RECIENTE</div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:6px">
          ${['last_5','last_10','last_20','last_24m'].map((k,i) => {
            const d = recent[k], lbl = ['Últ.5','Últ.10','Últ.20','24M'][i];
            return `<div style="background:var(--bg-secondary);border-radius:var(--r-sm);padding:8px;text-align:center;border:1px solid var(--border)">
              <div style="font-size:9px;color:var(--text-muted);text-transform:uppercase;margin-bottom:4px">${lbl}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600"><span style="color:var(--green)">${d.wins}</span>-${d.draws}-<span style="color:var(--red)">${d.losses}</span></div>
              <div style="font-size:9px;color:var(--text-muted);margin-top:2px">${fmt(d.points_per_game)}PPG</div>
            </div>`;
          }).join('')}
        </div>
      </div>
    `;
  }

  const advantageText = h.advantage === 'A' ? `<span class="sv-blue">${a.name}</span>` :
                        h.advantage === 'B' ? `<span class="sv-red">${b.name}</span>` : 'Equilibrado';

  document.getElementById('s1').innerHTML = `
    <div class="glass-card">
      ${generalHtml(a.name, a, 'a')}
      ${recentHtml(a.recent)}
    </div>
    <div class="glass-card">
      ${generalHtml(b.name, b, 'b')}
      ${recentHtml(b.recent)}
      <div style="margin-top:16px">
        <div class="team-section-title" style="color:var(--gold)">HEAD TO HEAD (${h.matches} partidos)</div>
        ${statRow(`${a.name}`, mono(`${h.wins_a}V · ${h.draws}E · ${h.wins_b}D`), 'sv-blue')}
        ${statRow('Goles', mono(`${h.gf_a} – ${h.gf_b}`))}
        ${statRow('GD A', mono(sign(h.gd_a)), h.gd_a >= 0 ? 'sv-green' : 'sv-red')}
        ${statRow('Ventaja histórica', advantageText)}
        ${h.recent_trend ? statRow('Tendencia reciente', h.recent_trend) : ''}
      </div>
    </div>
  `;
}

// ─── STAGE 2 ─────────────────────────────────────────────────────────────────
function renderStage2() {
  const s = state.result.stage_2;
  if (!s) return;
  const diffColor = s.elo_diff > 30 ? 'sv-blue' : s.elo_diff < -30 ? 'sv-red' : 'sv-gold';

  const kHtml = Object.entries(s.k_factor_desglose).map(([k,v]) =>
    `<div style="display:flex;justify-content:space-between;padding:3px 0;font-size:11px">
      <span style="color:var(--text-muted)">${k.replace(/_/g,' ')}</span>
      <span style="font-family:'JetBrains Mono',monospace;color:var(--text-secondary)">K=${v}</span>
    </div>`).join('');

  document.getElementById('s2').innerHTML = `
    <div class="glass-card">
      ${teamTitle(state.teamA, 'a')}
      ${statRow('Rating Elo', `<span style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:800;color:var(--blue)">${s.team_a.elo}</span>`)}
      ${statRow('Nivel', s.team_a.tier)}
      ${statRow('Tendencia 12m', `<span class="${s.team_a.trend_12m > 5 ? 'sv-green' : s.team_a.trend_12m < -5 ? 'sv-red' : ''}">${sign(s.team_a.trend_12m)} pts (${s.team_a.trend_direction})</span>`)}
      ${statRow('Prob. teórica Elo', mono(s.probabilidad_teorica_elo.team_a + '%'), 'sv-blue')}
      <div style="margin-top:16px">
        ${teamTitle(state.teamB, 'b')}
        ${statRow('Rating Elo', `<span style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:800;color:var(--red)">${s.team_b.elo}</span>`)}
        ${statRow('Nivel', s.team_b.tier)}
        ${statRow('Tendencia 12m', `<span class="${s.team_b.trend_12m > 5 ? 'sv-green' : s.team_b.trend_12m < -5 ? 'sv-red' : ''}">${sign(s.team_b.trend_12m)} pts (${s.team_b.trend_direction})</span>`)}
        ${statRow('Prob. teórica Elo', mono(s.probabilidad_teorica_elo.team_b + '%'), 'sv-red')}
      </div>
      ${statRow('Diferencial Elo', mono(sign(s.elo_diff)), diffColor)}
      <div class="methodology">${s.interpretacion}</div>
    </div>
    <div class="glass-card">
      <div class="card-header-sm">
        <h3>K-Factors por Competición</h3>
        <span class="tag-blue">Elo</span>
      </div>
      <div style="margin-bottom:12px">${kHtml}</div>
      <div class="methodology">${s.metodologia}</div>
    </div>
  `;
}

// ─── STAGE 3 ─────────────────────────────────────────────────────────────────
function renderStage3() {
  const s = state.result.stage_3;
  if (!s) return;
  const iA = s.team_a.ifr_score, iB = s.team_b.ifr_score;

  function recentMatches(desglose) {
    const matches = desglose[0]?.details || [];
    return matches.slice(0, 5).map(m => {
      const isW = m.base_score === 100, isD = m.base_score === 50;
      const dot = isW ? '🟢' : isD ? '🟡' : '🔴';
      return `<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;font-size:12px;border-bottom:1px solid rgba(255,255,255,0.03)">
        <span>${dot} ${m.opponent}</span>
        <span style="font-family:'JetBrains Mono',monospace;color:var(--text-muted)">${m.result}</span>
      </div>`;
    }).join('');
  }

  document.getElementById('s3').innerHTML = `
    <div class="glass-card">
      ${teamTitle(state.teamA, 'a')}
      ${ifrBar(iA, 'blue')}
      <div style="font-size:12px;color:var(--text-muted);margin-bottom:10px">${s.team_a.interpretacion}</div>
      <div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">Últimos 5 partidos</div>
      ${recentMatches(s.team_a.desglose_ponderado || [])}
      <div class="methodology">${s.metodologia}</div>
    </div>
    <div class="glass-card">
      ${teamTitle(state.teamB, 'b')}
      ${ifrBar(iB, 'red')}
      <div style="font-size:12px;color:var(--text-muted);margin-bottom:10px">${s.team_b.interpretacion}</div>
      <div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">Últimos 5 partidos</div>
      ${recentMatches(s.team_b.desglose_ponderado || [])}
    </div>
  `;
}

// ─── STAGE 4 ─────────────────────────────────────────────────────────────────
function renderStage4() {
  const s = state.result.stage_4;
  if (!s) return;

  function tierTable(data) {
    return Object.entries(data.tiers).map(([k,v]) => {
      const ppg = v.ppg || 0;
      const color = ppg >= 2 ? 'sv-green' : ppg >= 1.2 ? 'sv-gold' : 'sv-red';
      return `<div style="display:grid;grid-template-columns:110px 1fr auto;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.03);font-size:12px">
        <span style="color:var(--text-muted)">${v.label}</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px">${v.count}j · <span style="color:var(--green)">${v.wins}</span>V ${v.draws}E <span style="color:var(--red)">${v.losses}</span>D</span>
        <span class="stat-value ${color}">${ppg}PPG</span>
      </div>`;
    }).join('');
  }

  document.getElementById('s4').innerHTML = `
    <div class="glass-card">
      ${teamTitle(state.teamA, 'a')}
      ${tierTable(s.team_a)}
      ${statRow('PPG Global', mono(s.team_a.overall_ppg), 'sv-blue')}
      ${statRow('% Invicto vs Elite/Fuertes', mono(s.team_a.pct_unbeaten_vs_elite_strong + '%'))}
      <div style="margin-top:8px;padding:8px 12px;background:rgba(245,197,24,0.07);border-radius:var(--r-xs);border-left:2px solid var(--gold);font-size:12px;color:var(--gold)">${s.team_a.inflated_warning}</div>
      <div class="methodology">${s.metodologia}</div>
    </div>
    <div class="glass-card">
      ${teamTitle(state.teamB, 'b')}
      ${tierTable(s.team_b)}
      ${statRow('PPG Global', mono(s.team_b.overall_ppg), 'sv-red')}
      ${statRow('% Invicto vs Elite/Fuertes', mono(s.team_b.pct_unbeaten_vs_elite_strong + '%'))}
      <div style="margin-top:8px;padding:8px 12px;background:rgba(245,197,24,0.07);border-radius:var(--r-xs);border-left:2px solid var(--gold);font-size:12px;color:var(--gold)">${s.team_b.inflated_warning}</div>
    </div>
  `;
}

// ─── STAGE 5 ─────────────────────────────────────────────────────────────────
function renderStage5() {
  const s = state.result.stage_5;
  if (!s) return;

  function xgPanel(data, cls) {
    const venueRows = Object.entries(data.by_venue || {})
      .filter(([,v]) => v.count > 0)
      .map(([loc, v]) => statRow(loc, mono(`xG ${v.xg} / xGA ${v.xga} / xGD ${v.xgd >= 0 ? '+'+v.xgd : v.xgd}`))).join('');

    return `
      ${statRow('xG promedio', mono(data.xg), 'sv-green')}
      ${statRow('xGA promedio', mono(data.xga), 'sv-red')}
      ${statRow('xGD', mono(data.xgd >= 0 ? '+'+data.xgd : data.xgd), data.xgd >= 0 ? 'sv-green' : 'sv-red')}
      ${statRow('Efic. Ofensiva', data.offensive_efficiency, 'sv-blue')}
      ${statRow('Efic. Defensiva', data.defensive_efficiency, 'sv-purple')}
      <div style="margin-top:6px;font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px">Por condición</div>
      ${venueRows}
    `;
  }

  document.getElementById('s5').innerHTML = `
    <div class="glass-card">
      ${teamTitle(state.teamA, 'a')}
      ${xgPanel(s.team_a)}
      <div class="methodology">${s.metodologia}</div>
    </div>
    <div class="glass-card">
      ${teamTitle(state.teamB, 'b')}
      ${xgPanel(s.team_b)}
    </div>
  `;
}

// ─── STAGE 6 ─────────────────────────────────────────────────────────────────
function renderStage6() {
  const s = state.result.stage_6;
  if (!s) return;
  const rows = (s.factores || []).map(f => {
    const iaColor = f.impact_a >= 0 ? 'sv-green' : 'sv-red';
    const ibColor = f.impact_b >= 0 ? 'sv-green' : 'sv-red';
    return `<div class="stat-row">
      <span class="stat-label">${f.factor}</span>
      <span class="stat-value" style="display:flex;gap:16px;align-items:center">
        <span class="${iaColor}">${sign((f.impact_a*100).toFixed(0))}%</span>
        <span style="color:var(--border-active)">|</span>
        <span class="${ibColor}">${sign((f.impact_b*100).toFixed(0))}%</span>
        <span style="font-size:10px;color:var(--text-muted);max-width:200px;text-align:right">${f.detail}</span>
      </span>
    </div>`;
  }).join('');

  document.getElementById('s6').innerHTML = `
    <div class="card-header-sm">
      <h3>Impacto por Factor — <span style="color:var(--blue)">${state.teamA}</span> vs <span style="color:var(--red)">${state.teamB}</span></h3>
      <span class="tag-gold">Ajuste contextual</span>
    </div>
    <div style="font-size:11px;color:var(--text-muted);margin-bottom:10px">Porcentaje de ajuste aplicado al λ de cada equipo</div>
    ${rows}
    <div class="stat-row" style="font-weight:700;background:rgba(255,255,255,0.02);padding:10px;border-radius:var(--r-xs)">
      <span class="stat-label">AJUSTE TOTAL</span>
      <span style="display:flex;gap:24px">
        <span class="${s.ajuste_total_a >= 0 ? 'sv-green' : 'sv-red'}" style="font-family:'JetBrains Mono',monospace">${sign((s.ajuste_total_a*100).toFixed(1))}% (${state.teamA})</span>
        <span class="${s.ajuste_total_b >= 0 ? 'sv-green' : 'sv-red'}" style="font-family:'JetBrains Mono',monospace">${sign((s.ajuste_total_b*100).toFixed(1))}% (${state.teamB})</span>
      </span>
    </div>
  `;
}

// ─── STAGE 7 ─────────────────────────────────────────────────────────────────
function renderStage7() {
  const s = state.result.stage_7;
  if (!s) return;
  const diffColor = s.diferencia > 5 ? 'sv-blue' : s.diferencia < -5 ? 'sv-red' : 'sv-gold';

  const compRows = Object.entries(s.componentes).map(([k,v]) => {
    const max = Math.max(v.a, v.b, 1);
    const wA = (v.a/100*100).toFixed(0), wB = (v.b/100*100).toFixed(0);
    return `<div style="margin:6px 0">
      <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px">
        <span style="color:var(--text-muted)">${k.replace(/_/g,' ')} <span style="color:var(--text-muted);font-size:10px">(${v.peso})</span></span>
        <span><span style="color:var(--blue);font-family:'JetBrains Mono',monospace">${v.a.toFixed(1)}</span> vs <span style="color:var(--red);font-family:'JetBrains Mono',monospace">${v.b.toFixed(1)}</span></span>
      </div>
      <div style="display:flex;gap:2px;height:6px">
        <div style="flex:1;background:var(--bg-secondary);border-radius:3px;overflow:hidden">
          <div style="width:${wA}%;height:100%;background:var(--blue);border-radius:3px"></div>
        </div>
        <div style="flex:1;background:var(--bg-secondary);border-radius:3px;overflow:hidden">
          <div style="width:${wB}%;height:100%;background:var(--red);border-radius:3px"></div>
        </div>
      </div>
    </div>`;
  }).join('');

  document.getElementById('s7').innerHTML = `
    <div style="display:flex;gap:20px;margin-bottom:18px;flex-wrap:wrap">
      <div style="flex:1;min-width:120px">
        <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">IFF ${state.teamA}</div>
        <div style="font-family:'Outfit',sans-serif;font-size:42px;font-weight:900;color:var(--blue);line-height:1">${s.iff_a.toFixed(1)}</div>
      </div>
      <div style="flex:1;min-width:120px;text-align:right">
        <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">IFF ${state.teamB}</div>
        <div style="font-family:'Outfit',sans-serif;font-size:42px;font-weight:900;color:var(--red);line-height:1">${s.iff_b.toFixed(1)}</div>
      </div>
    </div>
    ${statRow('Diferencia IFF', mono(sign(s.diferencia.toFixed(1))), diffColor)}
    <div style="font-size:12px;color:var(--text-secondary);margin:8px 0 14px;padding:8px 12px;background:var(--bg-secondary);border-radius:var(--r-xs)">${s.interpretacion}</div>
    <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px">Componentes (A vs B)</div>
    ${compRows}
  `;

  renderRadar(s.componentes);
}

function renderRadar(comp) {
  const ctx = document.getElementById('radarChart').getContext('2d');
  if (state.charts.radar) state.charts.radar.destroy();
  const labels = Object.keys(comp).map(k => k.replace(/_/g,' '));
  const aVals = Object.values(comp).map(c => c.a);
  const bVals = Object.values(comp).map(c => c.b);
  state.charts.radar = new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [
        { label: state.teamA, data: aVals, backgroundColor: 'rgba(76,157,255,0.12)', borderColor: '#4c9dff', borderWidth: 2, pointBackgroundColor: '#4c9dff', pointRadius: 4 },
        { label: state.teamB, data: bVals, backgroundColor: 'rgba(255,76,106,0.12)', borderColor: '#ff4c6a', borderWidth: 2, pointBackgroundColor: '#ff4c6a', pointRadius: 4 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#8d97b0', font: { family: 'Inter', size: 12 } } } },
      scales: {
        r: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          angleLines: { color: 'rgba(255,255,255,0.04)' },
          pointLabels: { color: '#4d5670', font: { family: 'Inter', size: 10 } },
          ticks: { display: false }, suggestedMin: 0, suggestedMax: 100
        }
      }
    }
  });
}

// ─── STAGE 8 ─────────────────────────────────────────────────────────────────
function renderStage8() {
  const s = state.result.stage_8;
  if (!s) return;
  document.getElementById('s8').innerHTML = `
    <div class="card-header-sm">
      <h3>Goles Esperados por Equipo</h3>
      <span class="tag-green">Distribución Poisson</span>
    </div>
    <div style="display:flex;gap:32px;margin-bottom:20px;flex-wrap:wrap">
      <div style="text-align:center">
        <div style="font-size:11px;color:var(--blue);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">λ ${state.teamA}</div>
        <div style="font-family:'Outfit',sans-serif;font-size:52px;font-weight:900;color:var(--blue);line-height:1">${s.lambda_a}</div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:4px">goles esperados</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:11px;color:var(--red);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">λ ${state.teamB}</div>
        <div style="font-family:'Outfit',sans-serif;font-size:52px;font-weight:900;color:var(--red);line-height:1">${s.lambda_b}</div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:4px">goles esperados</div>
      </div>
    </div>
    ${statRow(`Coef. Ataque ${state.teamA}`, mono(s.ataque_a), 'sv-blue')}
    ${statRow(`Coef. Defensa ${state.teamB}`, mono(s.defensa_b), 'sv-red')}
    ${statRow(`Coef. Ataque ${state.teamB}`, mono(s.ataque_b), 'sv-red')}
    ${statRow(`Coef. Defensa ${state.teamA}`, mono(s.defensa_a), 'sv-blue')}
    ${statRow(`Factor Contexto ${state.teamA}`, mono(s.factor_contexto_a))}
    ${statRow(`Factor Contexto ${state.teamB}`, mono(s.factor_contexto_b))}
    ${statRow('G_avg global (histórico)', mono(s.goles_promedio_global))}
    <div class="methodology">Fórmula: ${s.formula} · G_avg histórico calculado sobre ${49478} partidos desde 1872</div>
  `;
}

// ─── STAGE 9 ─────────────────────────────────────────────────────────────────
function renderStage9() {
  const s = state.result.stage_9;
  if (!s) return;
  const ctx = document.getElementById('poissonChart').getContext('2d');
  if (state.charts.poisson) state.charts.poisson.destroy();
  state.charts.poisson = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: s.labels.map((l,i) => i === 5 ? '5+' : `${l} goles`),
      datasets: [
        { label: state.teamA, data: s.distribucion_a, backgroundColor: 'rgba(76,157,255,0.7)', borderColor: '#4c9dff', borderWidth: 1, borderRadius: 4 },
        { label: state.teamB, data: s.distribucion_b, backgroundColor: 'rgba(255,76,106,0.7)', borderColor: '#ff4c6a', borderWidth: 1, borderRadius: 4 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#8d97b0', font: { family: 'Inter', size: 12 } } },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${(ctx.parsed.y*100).toFixed(2)}%` } }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#4d5670', font: { family: 'Inter' } } },
        y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#4d5670', callback: v => `${(v*100).toFixed(0)}%`, font: { family: 'Inter' } } }
      }
    }
  });
}

// ─── STAGE 10 ────────────────────────────────────────────────────────────────
function renderStage10() {
  const s = state.result.stage_10;
  if (!s) return;
  document.getElementById('s10Info').textContent = `Corrección Dixon-Coles · ρ = ${s.rho} · ${s.explicacion?.substring(0,80)}...`;

  // Axis labels
  const axX = document.getElementById('heatmapAxisX');
  const axY = document.getElementById('heatmapAxisY');
  if (axX) axX.innerHTML = [0,1,2,3,4,'5+'].map(n => `<div style="width:42px;text-align:center;font-size:9px;color:var(--text-muted);font-family:'JetBrains Mono',monospace">${n}</div>`).join('');
  if (axY) axY.innerHTML = [0,1,2,3,4,'5+'].map(n => `<div class="heatmap-axis-label">${n}</div>`).join('');

  const matrix = s.after;
  const maxVal = Math.max(...matrix.flat());
  const grid = document.getElementById('heatmapGrid');
  grid.innerHTML = '';
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 6; j++) {
      const v = matrix[i][j];
      const t = v / maxVal;
      const cell = document.createElement('div');
      cell.className = 'hm-cell';
      cell.style.background = `rgba(0,232,122,${0.06 + 0.65*t})`;
      cell.style.borderColor = `rgba(0,232,122,${0.1 + 0.5*t})`;
      cell.title = `${i}-${j}: ${(v*100).toFixed(3)}%`;
      cell.innerHTML = `<span class="hm-pct" style="color:rgba(255,255,255,${0.5 + 0.5*t})">${(v*100).toFixed(1)}</span>`;
      grid.appendChild(cell);
    }
  }

  const cc = s.cambios_clave;
  document.getElementById('s10Changes').innerHTML = `
    <div style="margin-bottom:12px">
      ${[['0-0','00'],['1-0','10'],['0-1','01'],['1-1','11']].map(([label, key]) => {
        const val = cc[key];
        const color = val > 0 ? 'sv-green' : 'sv-red';
        return statRow(`Cambio ${label}`, `${sign((val*100).toFixed(4))}%`, color);
      }).join('')}
    </div>
    <div class="methodology">${s.explicacion}</div>
  `;
}

// ─── STAGE 11 ────────────────────────────────────────────────────────────────
function renderStage11() {
  const s = state.result.stage_11;
  if (!s) return;
  const p = s.probabilities;

  document.getElementById('s11').innerHTML = `
    <div class="card-header-sm">
      <h3>Resultados de 100,000 Simulaciones</h3>
      <span class="tag-green">Monte Carlo</span>
    </div>
    <div class="mc-stats">
      <div class="mc-stat-box">
        <div class="label">Victoria ${state.teamA}</div>
        <div class="value" style="color:var(--blue)">${pct(p.win_a)}</div>
      </div>
      <div class="mc-stat-box">
        <div class="label">Empate</div>
        <div class="value" style="color:var(--gold)">${pct(p.draw)}</div>
      </div>
      <div class="mc-stat-box">
        <div class="label">Victoria ${state.teamB}</div>
        <div class="value" style="color:var(--red)">${pct(p.win_b)}</div>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div>
        <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px">${state.teamA}</div>
        ${statRow('Goles prom.', mono(s.avg_goals_a?.toFixed(3) || '?'), 'sv-blue')}
        ${statRow('Varianza', mono(s.variance_a?.toFixed(4) || '?'))}
        ${statRow('Desv. estándar', mono(s.std_a?.toFixed(4) || '?'))}
        ${statRow('IC 95%', mono(`[${s.ci_a?.[0]}, ${s.ci_a?.[1]}]`))}
      </div>
      <div>
        <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px">${state.teamB}</div>
        ${statRow('Goles prom.', mono(s.avg_goals_b?.toFixed(3) || '?'), 'sv-red')}
        ${statRow('Varianza', mono(s.variance_b?.toFixed(4) || '?'))}
        ${statRow('Desv. estándar', mono(s.std_b?.toFixed(4) || '?'))}
        ${statRow('IC 95%', mono(`[${s.ci_b?.[0]}, ${s.ci_b?.[1]}]`))}
      </div>
    </div>
  `;
}

// ─── STAGE 12 ────────────────────────────────────────────────────────────────
function renderStage12() {
  const s = state.result.stage_12;
  if (!s) return;
  const base = s.base;

  const html = (s.escenarios || []).map(sc => {
    const wa = (sc.result.win_a*100).toFixed(1);
    const d = sc.delta_a;
    const dColor = d > 0 ? 'delta-pos' : d < 0 ? 'delta-neg' : 'delta-neu';
    return `<div class="sens-card">
      <div class="sens-name">${sc.name}</div>
      <div class="sens-prob" style="color:var(--blue)">${wa}%</div>
      <div class="sens-delta ${dColor}">${sign(d)}% vs base</div>
    </div>`;
  }).join('');

  // Add base card
  const basePct = (base.win_a*100).toFixed(1);
  document.getElementById('s12').innerHTML = `
    <div class="sens-card" style="border-color:var(--green);background:var(--green-glow)">
      <div class="sens-name" style="color:var(--green)">BASE</div>
      <div class="sens-prob" style="color:var(--green)">${basePct}%</div>
      <div class="sens-delta" style="color:var(--text-muted)">Escenario actual</div>
    </div>
    ${html}
  `;
}

// ─── STAGE 13 ────────────────────────────────────────────────────────────────
function renderStage13() {
  const s = state.result.stage_13;
  if (!s) return;
  const scDef = [
    { key:'base', name:'Base', color:'var(--green)' },
    { key:'conservador', name:'Conservador', color:'var(--purple)' },
    { key:'optimista_a', name:`Opt. ${state.teamA}`, color:'var(--blue)' },
    { key:'optimista_b', name:`Opt. ${state.teamB}`, color:'var(--red)' },
    { key:'sorpresa', name:'Sorpresa', color:'var(--gold)' }
  ];
  const maxH = 140;

  const barsHtml = scDef.map(sc => {
    const data = s[sc.key];
    if (!data) return '';
    const wa = (data.win_a*100).toFixed(1), wd = (data.draw*100).toFixed(1), wb = (data.win_b*100).toFixed(1);
    return `<div class="scenario-col">
      <div class="scenario-name" style="color:${sc.color}">${sc.name}</div>
      <div class="scenario-bars">
        <div class="sc-bar" style="height:${data.win_a*maxH}px;background:var(--blue)" title="${state.teamA}: ${wa}%"></div>
        <div class="sc-bar" style="height:${data.draw*maxH}px;background:var(--gold)" title="Empate: ${wd}%"></div>
        <div class="sc-bar" style="height:${data.win_b*maxH}px;background:var(--red)" title="${state.teamB}: ${wb}%"></div>
      </div>
      <div class="sc-vals">
        <span style="color:var(--blue)">${wa}</span>
        <span style="color:var(--gold)">${wd}</span>
        <span style="color:var(--red)">${wb}</span>
      </div>
    </div>`;
  }).join('');

  const descHtml = Object.entries(s.descripcion_escenarios || {}).map(([k,v]) =>
    `<div style="padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.03);font-size:12px">
      <span style="color:var(--text-secondary);font-weight:600">${k.charAt(0).toUpperCase()+k.slice(1)}:</span>
      <span style="color:var(--text-muted);margin-left:6px">${v}</span>
    </div>`).join('');

  document.getElementById('s13').innerHTML = `
    <div class="card-header-sm">
      <h3>Comparativa de Escenarios</h3>
      <div style="display:flex;gap:6px">
        <span style="display:flex;align-items:center;gap:3px;font-size:11px;color:var(--text-muted)"><div style="width:10px;height:10px;background:var(--blue);border-radius:2px"></div> A</span>
        <span style="display:flex;align-items:center;gap:3px;font-size:11px;color:var(--text-muted)"><div style="width:10px;height:10px;background:var(--gold);border-radius:2px"></div> X</span>
        <span style="display:flex;align-items:center;gap:3px;font-size:11px;color:var(--text-muted)"><div style="width:10px;height:10px;background:var(--red);border-radius:2px"></div> B</span>
      </div>
    </div>
    <div class="scenarios-wrap">${barsHtml}</div>
    <div style="margin-top:12px">${descHtml}</div>
  `;
}

// ─── STAGE 14 ────────────────────────────────────────────────────────────────
function renderStage14() {
  const s = state.result.stage_14;
  if (!s) return;

  const mainMarkets = [
    { label: 'Ambos Marcan', pct: s.btts_pct, odds: s.btts_odds },
    { label: 'No Ambos', pct: s.no_btts_pct, odds: s.no_btts_odds },
    { label: `CS ${state.teamA}`, pct: s.clean_sheet_a_pct, odds: s.clean_sheet_a_odds },
    { label: `CS ${state.teamB}`, pct: s.clean_sheet_b_pct, odds: s.clean_sheet_b_odds },
  ];

  const mainHtml = mainMarkets.map(m =>
    `<div class="market-card">
      <div class="market-label">${m.label}</div>
      <div class="market-pct">${m.pct || '-'}</div>
      ${m.odds ? `<div class="market-odds">@ ${m.odds}</div>` : ''}
    </div>`).join('');

  const overTargets = [0.5, 1.5, 2.5, 3.5, 4.5];
  const ovHtml = overTargets.map(t => {
    const k = `over_${t}`;
    return `<div class="market-card">
      <div class="market-label">Over ${t}</div>
      <div class="market-pct" style="color:var(--green)">${s.overs_pct?.[k] || '-'}</div>
      <div class="market-odds">@ ${s.overs_odds?.[k] || '-'}</div>
    </div>`;
  }).join('');

  const unHtml = overTargets.map(t => {
    const k = `under_${t}`;
    return `<div class="market-card">
      <div class="market-label">Under ${t}</div>
      <div class="market-pct" style="color:var(--red)">${s.unders_pct?.[k] || '-'}</div>
    </div>`;
  }).join('');

  document.getElementById('s14').innerHTML = `
    <div class="markets-section-title">Mercados Principales</div>
    <div class="markets-grid">${mainHtml}</div>
    <div class="markets-section-title">Over / Más de X Goles Totales</div>
    <div class="markets-grid">${ovHtml}</div>
    <div class="markets-section-title">Under / Menos de X Goles Totales</div>
    <div class="markets-grid">${unHtml}</div>
  `;
}

// ─── STAGE 15 ────────────────────────────────────────────────────────────────
function renderStage15() {
  const s = state.result.stage_15;
  if (!s) return;

  document.getElementById('s15').innerHTML = `
    <div class="card-header-sm">
      <h3>Comparativa con Rankings Externos</h3>
      <span class="tag-gold">Validación</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:14px">
      <div>
        <div class="team-section-title a">${state.teamA}</div>
        ${statRow('Elo Rating', mono(s.elo_rating.team_a), 'sv-blue')}
        ${statRow('Posición Elo Global', mono(s.elo_rating.posicion_global_a), 'sv-blue')}
        ${statRow('FIFA Ranking Est.', mono(s.fifa_ranking_estimado.team_a))}
      </div>
      <div>
        <div class="team-section-title b">${state.teamB}</div>
        ${statRow('Elo Rating', mono(s.elo_rating.team_b), 'sv-red')}
        ${statRow('Posición Elo Global', mono(s.elo_rating.posicion_global_b), 'sv-red')}
        ${statRow('FIFA Ranking Est.', mono(s.fifa_ranking_estimado.team_b))}
      </div>
    </div>
    <div style="padding:12px 16px;background:rgba(245,197,24,0.06);border:1px solid rgba(245,197,24,0.15);border-radius:var(--r-sm)">
      <div style="font-size:12px;color:var(--gold);font-weight:600;margin-bottom:4px">Análisis de Divergencias</div>
      <div style="font-size:12px;color:var(--text-secondary)">${s.analisis_divergencias.divergencia_elo_fifa}</div>
    </div>
    <div class="methodology">${s.analisis_divergencias.nota} ${s.analisis_divergencias.limitacion}</div>
  `;
}

// ─── STAGE 16 ────────────────────────────────────────────────────────────────
function renderStage16() {
  const s = state.result.stage_16;
  if (!s) return;
  const pf = s.probabilidades_finales;
  const conf = s.nivel_confianza;
  const lam = state.result.stage_8;

  const riskHtml = (s.principales_riesgos || []).map(r =>
    `<li class="risk-item"><span class="risk-icon">⚠</span><span>${r}</span></li>`).join('');

  const topScoresHtml = (s.top_10_marcadores || []).map(t =>
    `<div class="score-chip">
      <span class="sc-score">${t.score}</span>
      <span class="sc-pct">${t.probability}%</span>
    </div>`).join('');

  document.getElementById('s16').innerHTML = `
    <div class="exec-wrapper">
      <div class="exec-hero">
        <div class="exec-score-label">Marcador Más Probable</div>
        <div class="exec-score">${s.marcador_mas_probable?.score || '?'}</div>
        <div style="font-size:13px;color:var(--text-muted);margin-top:4px">${s.marcador_mas_probable?.probability || 0}% de probabilidad</div>
        <div class="exec-finprobs">
          <div class="exec-prob-item">
            <div class="exec-prob-val" style="color:var(--blue)">${pf.victoria_a}</div>
            <div class="exec-prob-label">${state.teamA}</div>
          </div>
          <div class="exec-prob-item">
            <div class="exec-prob-val" style="color:var(--gold)">${pf.empate}</div>
            <div class="exec-prob-label">Empate</div>
          </div>
          <div class="exec-prob-item">
            <div class="exec-prob-val" style="color:var(--red)">${pf.victoria_b}</div>
            <div class="exec-prob-label">${state.teamB}</div>
          </div>
        </div>
        <div style="margin-top:16px">
          <div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">Goles Esperados</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:16px">
            <span style="color:var(--blue)">${lam?.lambda_a?.toFixed(2) || '?'}</span>
            <span style="color:var(--text-muted)"> – </span>
            <span style="color:var(--red)">${lam?.lambda_b?.toFixed(2) || '?'}</span>
          </div>
        </div>
      </div>

      <div class="exec-right">
        <div class="exec-confidence">
          <div class="conf-label">Nivel de Confianza del Modelo</div>
          <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width:${conf}%"></div>
          </div>
          <div class="conf-value">${conf}%</div>
        </div>

        <div class="glass-card" style="padding:16px">
          <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px">Top 10 Marcadores Más Probables</div>
          <div class="top-scores">${topScoresHtml}</div>
        </div>

        <div class="glass-card" style="padding:16px">
          <div style="font-size:11px;color:var(--red);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px">Principales Riesgos</div>
          <ul class="risks-list">${riskHtml}</ul>
        </div>
      </div>

      <div class="exec-summary">
        <div style="font-size:11px;color:var(--green);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;font-weight:700">📋 Resumen Ejecutivo</div>
        ${s.resumen_ejecutivo}
      </div>
    </div>
  `;
}

// ─── UTILS ───────────────────────────────────────────────────────────────────
function showLoading(on) {
  document.getElementById('loading').classList.toggle('active', on);
  document.getElementById('predictBtn').disabled = on;
}

function showError(msg) {
  const el = document.getElementById('errorMsg');
  el.textContent = '⚠ ' + msg;
  el.classList.add('active');
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideError() {
  document.getElementById('errorMsg').classList.remove('active');
}
