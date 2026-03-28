import numpy as np
import statsmodels.api as sm

# -------------------------------------------------------------------------
# HISTORICAL DATA (5 YEARS: 2021-2025)
# -------------------------------------------------------------------------
company_a = {
    'name': 'Company A',
    'years': [2021, 2022, 2023, 2024, 2025],
    'cell_towers':          [21_000, 22_200, 23_500, 24_300, 25_000],
    'fiber_route_km':       [58_000, 63_000, 69_000, 75_000, 80_000],
    'employees':            [48_000, 47_200, 46_500, 45_800, 45_000],
    'annual_opex_millions':  [10_200, 10_800, 11_300, 11_700, 12_000],
    'annual_capex_millions': [3_000,  3_100,  3_300,  3_400,  3_500],
    'subscribers_millions':  [28.0,   30.1,   32.0,   33.8,   35.0],
}

company_b = {
    'name': 'Company B',
    'years': [2021, 2022, 2023, 2024, 2025],
    'cell_towers':          [14_500, 15_300, 16_200, 17_100, 18_000],
    'fiber_route_km':       [38_000, 42_000, 46_000, 50_500, 55_000],
    'employees':            [35_000, 34_200, 33_500, 32_800, 32_000],
    'annual_opex_millions':  [7_000,  7_400,  7_800,  8_100,  8_500],
    'annual_capex_millions': [1_800,  1_900,  2_000,  2_100,  2_200],
    'subscribers_millions':  [17.0,   18.5,   19.8,   21.0,   22.0],
}


# -------------------------------------------------------------------------
# FIT LINEAR TREND
# -------------------------------------------------------------------------

def fit_trend(years, values):
    """
    Fit a linear trend via OLS.
    Returns (slope_per_year, intercept, r_squared, p_value).
    """
    X = sm.add_constant(years)
    model = sm.OLS(values, X).fit()

    intercept, slope = model.params
    r_squared = model.rsquared
    p_value = model.pvalues[1]

    return slope, intercept, r_squared, p_value


# -------------------------------------------------------------------------
# FORECAST VALUE
# -------------------------------------------------------------------------

def project_value(years, values, target_year):
    """Project a metric to target_year using the linear trend."""
    slope, intercept, r_sq, p_val = fit_trend(years, values)
    projected = slope * target_year + intercept
    return projected, slope, r_sq, p_val


# -------------------------------------------------------------------------
# BUILT FORECAST
# -------------------------------------------------------------------------

def build_projections(company, target_year=2026):
    """
    For each metric in a company's history, compute:
      - latest actual value
      - projected value at target_year
      - annual growth (slope)
      - trend reliability (R-squared)
    """
    metrics = [k for k in company if k not in ('name', 'years')]
    years = company['years']
    projections = {'name': company['name']}

    for metric in metrics:
        values = company[metric]
        projected, slope, r_sq, p_val = project_value(years, values, target_year)
        projections[metric] = {
            'history': dict(zip(years, values)),
            'latest': values[-1],
            'projected': round(projected, 2),
            'r_squared': round(r_sq, 4),
        }

    return projections


# -------------------------------------------------------------------------
# CALCULATE SYNERGIES
# -------------------------------------------------------------------------

def calc_synergy(proj_a, proj_b, scenario=None):
    """
    Compute merger synergies using projected (or latest) values.
    Uses projected values when the trend is reliable (R^2 > 0.85),
    falls back to latest actual otherwise.
    """
    if scenario is None:
        scenario = {
            'tower_overlap_pct': 0.30,
            'fiber_overlap_pct': 0.20,
            'workforce_reduction_pct': 0.15,
            'procurement_discount_pct': 0.08,
            'execution_risk': 0.75,
        }

    def eval_trend(projection_entry):
        """Use projected value if trend is strong, otherwise latest actual."""
        if projection_entry['r_squared'] > 0.85:
            return projection_entry['projected']
        return projection_entry['latest']

    # Pull values with trend-aware selection
    towers_a = eval_trend(proj_a['cell_towers'])
    towers_b = eval_trend(proj_b['cell_towers'])
    fiber_a = eval_trend(proj_a['fiber_route_km'])
    fiber_b = eval_trend(proj_b['fiber_route_km'])
    emp_a = eval_trend(proj_a['employees'])
    emp_b = eval_trend(proj_b['employees'])
    opex_a = eval_trend(proj_a['annual_opex_millions'])
    opex_b = eval_trend(proj_b['annual_opex_millions'])
    capex_a = eval_trend(proj_a['annual_capex_millions'])
    capex_b = eval_trend(proj_b['annual_capex_millions'])

    s = scenario

    # Savings (annual, in $M)
    savings = {}

    # Decommission towers (~$100K/tower/year)
    redundant_towers = min(towers_a, towers_b) * s['tower_overlap_pct']
    savings['network_towers'] = redundant_towers * 0.1

    # Duplicate fiber (~$2K/km/year maintenance)
    redundant_fiber = min(fiber_a, fiber_b) * s['fiber_overlap_pct']
    savings['network_fiber'] = redundant_fiber * 0.002

    # Redundant roles (~$85K/employee/year)
    combined_employees = emp_a + emp_b
    headcount_cut = combined_employees * s['workforce_reduction_pct']
    savings['workforce'] = headcount_cut * 0.085

    # Procurement discount on combined opex
    combined_opex = opex_a + opex_b
    savings['procurement'] = combined_opex * s['procurement_discount_pct']

    total_annual = sum(savings.values())

    # Integration costs (one-time, in $M)
    costs = {}
    costs['severance'] = headcount_cut * 0.08             # ~$80K/person
    costs['tower_decommission'] = redundant_towers * 0.15  # ~$150K/tower
    costs['it_migration'] = 360.0                          # flat estimate for BSS/OSS
    costs['advisory_regulatory'] = combined_opex * 0.01

    total_costs = sum(costs.values())

    # Compute present value of synergies over 5 years.
    # Savings ramp linearly from 0% to 100% over the integration period,
    # then get discounted at WACC and reduced by execution risk.
    wacc = 0.09
    integration_years = 3.0  # years until synergies are fully realized
    npv_savings = 0.0
    for yr in range(1, 6):
         # Linear ramp: yr1 = 33%, yr2 = 67%, yr3+ = 100%
        ramp = min(yr / integration_years, 1.0)
        # Apply execution risk: only 75% of projected synergies typically materialize
        realized = total_annual * ramp * s['execution_risk']
        # Discount back to present value: $1 in year 3 is worth $1/(1.09)^3 today
        npv_savings += realized / (1 + wacc) ** yr
    
    # Integration costs are front-loaded: 60% hits in year 1, 40% in year 2
    npv_costs = 0.0
    for yr in range(1, 3):
        share = 0.6 if yr == 1 else 0.4
        npv_costs += (total_costs * share) / (1 + wacc) ** yr

    return {
        'annual_savings_breakdown': {k: round(v, 1) for k, v in savings.items()},
        'total_annual_savings_M': round(total_annual, 1),
        'integration_costs_breakdown': {k: round(v, 1) for k, v in costs.items()},
        'total_integration_costs_M': round(total_costs, 1),
        'npv_synergies_5yr_M': round(npv_savings, 1),
        'npv_integration_costs_M': round(npv_costs, 1),
        'net_synergy_npv_M': round(npv_savings - npv_costs, 1),
        'synergy_pct_combined_opex': round(total_annual / (opex_a + opex_b) * 100, 2),
    }


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

proj_a = build_projections(company_a, target_year=2026)
proj_b = build_projections(company_b, target_year=2026)

print('=' * 65)
print(f"  TREND ANALYSIS: {company_a['name']}")
print('=' * 65)
for metric, data in proj_a.items():
    if metric == 'name':
        continue
    print(f"\n  {metric}:")
    print(f"    Latest (2025):  {data['latest']}")
    print(f"    Projected 2026: {data['projected']}")
    print(f"    Trend fit (R2): {data['r_squared']}")

print(f"\n{'=' * 65}")
print(f"  TREND ANALYSIS: {company_b['name']}")
print('=' * 65)
for metric, data in proj_b.items():
    if metric == 'name':
        continue
    print(f"\n  {metric}:")
    print(f"    Latest (2025):  {data['latest']}")
    print(f"    Projected 2026: {data['projected']}")
    print(f"    Trend fit (R2): {data['r_squared']}")

print(f"\n{'=' * 65}")
print(f"  MERGER SYNERGY (using 2026 projections)")
print('=' * 65)

result = calc_synergy(proj_a, proj_b)

print("\n  Annual Savings ($M):")
for k, v in result['annual_savings_breakdown'].items():
    print(f"    {k:.<35s} ${v:>10.1f}M")
print(f"    {'TOTAL':.<35s} ${result['total_annual_savings_M']:>10.1f}M")
print(f"\n    Synergies as % of combined opex: {result['synergy_pct_combined_opex']}%")

print("\n  One-Time Integration Costs ($M):")
for k, v in result['integration_costs_breakdown'].items():
    print(f"    {k:.<35s} ${v:>10.1f}M")
print(f"    {'TOTAL':.<35s} ${result['total_integration_costs_M']:>10.1f}M")

print(f"\n  5-Year NPV Summary ($M):")
print(f"    NPV synergies:         ${result['npv_synergies_5yr_M']:>10.1f}M")
print(f"    NPV integration costs: ${result['npv_integration_costs_M']:>10.1f}M")
print(f"    NET SYNERGY VALUE:     ${result['net_synergy_npv_M']:>10.1f}M")
print('=' * 65)