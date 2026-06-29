import math
import numpy as np

def poisson_prob(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def poisson_distribution(lam, max_goals=6):
    return [poisson_prob(i, lam) for i in range(max_goals)]

def dixon_coles_adjustment(lam_a, lam_b, rho=-0.06, max_goals=6):
    base = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            base[i][j] = poisson_prob(i, lam_a) * poisson_prob(j, lam_b)
    tau = np.ones((max_goals, max_goals))
    for i in range(2):
        for j in range(2):
            if i == 0 and j == 0:
                tau[i][j] = 1 - lam_a * lam_b * rho
            elif i == 0 and j == 1:
                tau[i][j] = 1 + lam_a * rho
            elif i == 1 and j == 0:
                tau[i][j] = 1 + lam_b * rho
            elif i == 1 and j == 1:
                tau[i][j] = 1 - rho
    adjusted = base * tau
    adjusted = adjusted / adjusted.sum()
    return adjusted.tolist()

def monte_carlo_simulation(matrix, iterations=100000):
    max_goals = len(matrix)
    flat_probs = np.array(matrix).flatten()
    flat_probs = flat_probs / flat_probs.sum()
    indices = np.random.choice(len(flat_probs), size=iterations, p=flat_probs)
    i_indices = indices // max_goals
    j_indices = indices % max_goals
    wins_a = int(np.sum(i_indices > j_indices))
    draws = int(np.sum(i_indices == j_indices))
    wins_b = iterations - wins_a - draws
    probs = {'win_a': wins_a / iterations, 'draw': draws / iterations, 'win_b': wins_b / iterations}
    avg_a = float(np.mean(i_indices))
    avg_b = float(np.mean(j_indices))
    var_a = float(np.var(i_indices))
    var_b = float(np.var(j_indices))
    std_a = float(np.std(i_indices))
    std_b = float(np.std(j_indices))
    sorted_a = np.sort(i_indices)
    sorted_b = np.sort(j_indices)
    ci_a = (int(sorted_a[int(iterations * 0.025)]), int(sorted_a[int(iterations * 0.975)]))
    ci_b = (int(sorted_b[int(iterations * 0.025)]), int(sorted_b[int(iterations * 0.975)]))
    return {
        'probabilities': probs,
        'avg_goals_a': avg_a,
        'avg_goals_b': avg_b,
        'variance_a': var_a,
        'variance_b': var_b,
        'std_a': std_a,
        'std_b': std_b,
        'ci_a': ci_a,
        'ci_b': ci_b,
        'total_simulations': iterations
    }

def expected_goals_from_matrix(matrix, max_goals=6):
    exp_a = 0.0
    exp_b = 0.0
    for i in range(max_goals):
        for j in range(max_goals):
            exp_a += i * matrix[i][j]
            exp_b += j * matrix[i][j]
    return exp_a, exp_b

def elo_expected(ra, rb):
    return 1.0 / (1.0 + math.pow(10, (rb - ra) / 400.0))

def elo_update(ra, rb, score_a, score_b, k=30, goal_diff=None):
    expected = elo_expected(ra, rb)
    actual = 1.0 if score_a > score_b else (0.5 if score_a == score_b else 0.0)
    if goal_diff is None:
        goal_diff = abs(score_a - score_b)
    if goal_diff <= 1:
        g_mult = 1.0
    elif goal_diff == 2:
        g_mult = 1.5
    else:
        g_mult = (11.0 + goal_diff) / 8.0
    adjustment = k * g_mult * (actual - expected)
    return ra + adjustment, rb - adjustment

def tournament_k_factor(tournament):
    t = tournament.lower()
    if any(w in t for w in ['world cup', 'worldcup', 'fifa world']):
        return 60
    if any(w in t for w in ['qualif', 'qualifier', 'world cup qualification',
                            'euro qualification', 'afcon qualification',
                            'copa america qualification', 'asian cup qualification']):
        return 54
    if any(w in t for w in ['euro', 'copa america', 'africa cup', 'afcon',
                            'asian cup', 'gold cup', 'african cup',
                            'copa américa', 'concacaf']):
        return 48
    if any(w in t for w in ['nations league', 'nationsleague', 'nations']):
        return 42
    if any(w in t for w in ['friendly']):
        return 21
    return 30

def btts_probability(matrix, max_goals=6):
    prob = 0.0
    for i in range(1, max_goals):
        for j in range(1, max_goals):
            prob += matrix[i][j]
    return prob

def over_probability(matrix, threshold, max_goals=6):
    prob = 0.0
    for i in range(max_goals):
        for j in range(max_goals):
            if i + j > threshold:
                prob += matrix[i][j]
    return prob

def under_probability(matrix, threshold, max_goals=6):
    prob = 0.0
    for i in range(max_goals):
        for j in range(max_goals):
            if i + j <= threshold:
                prob += matrix[i][j]
    return prob

def clean_sheet_probability(matrix, team='a', max_goals=6):
    prob = 0.0
    if team.lower() == 'a':
        for j in range(max_goals):
            prob += matrix[0][j]
    else:
        for i in range(max_goals):
            prob += matrix[i][0]
    return prob

def implied_odds(prob):
    if prob <= 0:
        return 999.0
    return 1.0 / prob
