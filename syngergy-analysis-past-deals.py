import pandas as pd
import statsmodels.api as sm

# -------------------------------------------------------------------------
# PAST DEALS HISTORICAL ASSUMPTIONS
# -------------------------------------------------------------------------

deals = pd.DataFrame({
    'deal_name': [
        'Verizon-Alltel', 'AT&T-Cingular', 'Sprint-Nextel',
        'T-Mobile-MetroPCS', 'CenturyLink-Level3', 'Charter-TWC',
        'Altice-Cablevision', 'Frontier-Verizon'
    ],
    # % of service areas where both companies operated before the merger
    # Higher = more redundant infrastructure to consolidate = more savings
    'geo_overlap_pct': [30, 60, 15, 70, 40, 55, 20, 75],
    # Target company size / acquirer company size (by revenue or assets)
    # 0.3 = target is much smaller, 0.9 = nearly equal
    'size_ratio': [0.5, 0.8, 0.3, 0.9, 0.6, 0.7, 0.4, 0.85],
    # % of physical network assets (towers, fiber, data centers) that overlap
    'network_overlap_pct': [20, 45, 10, 55, 30, 40, 15, 60],
    # Actual cost savings achieved after merger, as % of combined operating costs
    'realized_synergy_pct': [8, 12, 5, 14, 9, 11, 6, 15]
})

print(deals.to_string(index=False))
print(f"\n{len(deals)} deals, {deals.shape[1] - 2} features\n")


# -------------------------------------------------------------------------
# REGRESSION
# -------------------------------------------------------------------------

feature_cols = ['geo_overlap_pct', 'size_ratio', 'network_overlap_pct']
X = deals[feature_cols]
y = deals['realized_synergy_pct']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# -------------------------------------------------------------------------
# FORECASTING
# -------------------------------------------------------------------------

your_deal = pd.DataFrame({
    'const': [1],
    'geo_overlap_pct': [45],
    'size_ratio': [0.65],
    'network_overlap_pct': [35]
})

prediction = model.get_prediction(your_deal)
print(f"\nYour deal prediction:")
print(prediction.summary_frame(alpha=0.05))

# -------------------------------------------------------------------------
# PRESENT DEAL
# -------------------------------------------------------------------------

combined_costs = 50  # $50M
pred_pct = prediction.predicted_mean[0]
print(f"\nPredicted synergy: {pred_pct:.1f}% of ${combined_costs}M = ${pred_pct/100 * combined_costs:.1f}M")


# -------------------------------------------------------------------------
# CONCLUSIONS
# -------------------------------------------------------------------------

# P-values are very bad, which implies that mergers produce roughly
# 2.9% synergies on average, but it can't tell  whether 
# geographic overlap, size ratio, or network overlap actually predict how much.
# The confidence intervals on every variable cross zero, which means 
# any of them could have a positive, negative, or zero effect and you can't tell which.

# 8 observations, 3 variables.
# Only 4 degrees of freedom. Almost no data.

# Multicollinearity.
# Geographic overlap and network overlap are probably highly correlated.
# Companies that overlap geographically tend to overlap in infrastructure too.