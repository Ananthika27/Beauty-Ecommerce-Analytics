"""Measured campaign uplift from the Dunnhumby "Complete Journey" data.

The cosmetics dataset has no marketing-treatment data, so campaign ROI there rests on
assumed uplift. Dunnhumby records which households were TARGETED by each of 30
campaigns, so we can measure what targeting actually did to spend and to the
probability of buying.

Method (difference-in-differences with matching on past behaviour):
  * For each campaign, window = [START_DAY, END_DAY]; pre-window = same length just before.
  * Outcome per household: spend in window, and whether it bought at all in window.
  * Change = outcome(window) - outcome(pre-window).
  * Uplift = mean change of targeted households - mean change of non-targeted ones,
    computed within quintiles of pre-window spend (so like is compared with like)
    and averaged with weights equal to the number of targeted households.
  * 95% confidence interval by bootstrapping households.
Checks: a pre-trend test (same calculation on two PRE windows, should be ~0) and a
placebo test (random fake targeting, should be ~0).

Targeting was NOT randomised, so this is observational evidence, not proof.

Usage:  python -I src/campaign_uplift.py [data_dir] [out_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 300
MIN_TREATED = 10


def build_matrices(tx):
    """Households x days arrays of spend and basket counts."""
    days = int(tx["DAY"].max())
    hh = np.sort(tx["household_key"].unique())
    idx = {h: i for i, h in enumerate(hh)}
    spend = np.zeros((len(hh), days + 1))
    baskets = np.zeros((len(hh), days + 1))
    daily = tx.groupby(["household_key", "DAY"]).agg(s=("SALES_VALUE", "sum"), b=("BASKET_ID", "nunique")).reset_index()
    rows = daily["household_key"].map(idx).to_numpy()
    spend[rows, daily["DAY"].to_numpy()] = daily["s"].to_numpy()
    baskets[rows, daily["DAY"].to_numpy()] = daily["b"].to_numpy()
    return hh, spend, baskets


def window(mat, start, end):
    return mat[:, start:end + 1].sum(axis=1)


def stratified_did(delta, treated, strata, weights_by="treated"):
    """Weighted mean over strata of (treated mean change - control mean change)."""
    effects, weights = [], []
    for s in np.unique(strata):
        t = delta[(strata == s) & treated]
        c = delta[(strata == s) & ~treated]
        if len(t) >= 3 and len(c) >= 3:
            effects.append(t.mean() - c.mean())
            weights.append(len(t))
    return np.average(effects, weights=weights) if effects else np.nan


def make_strata(pre_spend, q=5):
    ranks = pd.Series(pre_spend).rank(method="first")
    return pd.qcut(ranks, q, labels=False).to_numpy()


def estimate(spend_delta, buy_delta, treated, pre_spend, rng, n_boot=N_BOOT):
    strata = make_strata(pre_spend)
    est = {
        "spend": stratified_did(spend_delta, treated, strata),
        "buy": stratified_did(buy_delta, treated, strata),
    }
    boots = {"spend": [], "buy": []}
    n = len(treated)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        st = make_strata(pre_spend[i])
        boots["spend"].append(stratified_did(spend_delta[i], treated[i], st))
        boots["buy"].append(stratified_did(buy_delta[i], treated[i], st))
    for k in boots:
        lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
        est[f"{k}_ci"] = (lo, hi)
    return est


def campaign_effect(spend, baskets, active_before, treated_ids, hh, start, end, rng, shift=0, n_boot=N_BOOT):
    """Effect of targeting for a window (shift moves the whole comparison earlier for pre-trend tests)."""
    length = end - start + 1
    s, e = start - shift, end - shift
    pre_s, pre_e = s - length, s - 1
    if pre_s < 1:
        return None
    post_spend, pre_spend = window(spend, s, e), window(spend, pre_s, pre_e)
    post_buy, pre_buy = (window(baskets, s, e) > 0).astype(float), (window(baskets, pre_s, pre_e) > 0).astype(float)
    eligible = active_before(pre_s)
    treated = np.isin(hh, treated_ids)
    mask = eligible
    if treated[mask].sum() < MIN_TREATED:
        return None
    out = estimate((post_spend - pre_spend)[mask], (post_buy - pre_buy)[mask], treated[mask], pre_spend[mask], rng, n_boot)
    out.update(
        treated=int(treated[mask].sum()), controls=int((~treated[mask]).sum()),
        weeks=round(length / 7, 1), baseline_spend=float(post_spend[mask][~treated[mask]].mean()),
        baseline_buy_rate=float(post_buy[mask][~treated[mask]].mean()),
    )
    return out


def main(data_dir, out_dir):
    rng = np.random.default_rng(42)
    tx = pd.read_csv(data_dir / "transaction_data.csv", usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    desc = pd.read_csv(data_dir / "campaign_desc.csv")
    table = pd.read_csv(data_dir / "campaign_table.csv")
    redemptions = pd.read_csv(data_dir / "coupon_redempt.csv")

    hh, spend, baskets = build_matrices(tx)
    last_day = int(tx["DAY"].max())
    first_seen = tx.groupby("household_key")["DAY"].min().reindex(hh).to_numpy()

    def active_before(day):
        return first_seen < day            # household already a customer before the pre-window

    rows, pretrend, placebo = [], [], []
    for c in desc.sort_values("START_DAY").itertuples():
        if c.END_DAY > last_day:
            continue                        # window extends past the data
        targeted = table.loc[table["CAMPAIGN"] == c.CAMPAIGN, "household_key"].to_numpy()
        res = campaign_effect(spend, baskets, active_before, targeted, hh, c.START_DAY, c.END_DAY, rng)
        if res is None:
            continue
        rows.append({
            "campaign": c.CAMPAIGN, "type": c.DESCRIPTION, "start_day": c.START_DAY, "end_day": c.END_DAY, **{
                "targeted_households": res["treated"], "control_households": res["controls"], "weeks": res["weeks"],
                "uplift_spend_per_household": res["spend"], "spend_ci_low": res["spend_ci"][0], "spend_ci_high": res["spend_ci"][1],
                "uplift_buy_prob_pp": res["buy"] * 100, "buy_ci_low_pp": res["buy_ci"][0] * 100, "buy_ci_high_pp": res["buy_ci"][1] * 100,
                "control_spend_per_household": res["baseline_spend"], "control_buy_rate": res["baseline_buy_rate"],
            }})
        # pre-trend: same calculation shifted one window earlier (no campaign was running)
        length = c.END_DAY - c.START_DAY + 1
        pt = campaign_effect(spend, baskets, active_before, targeted, hh, c.START_DAY, c.END_DAY, rng, shift=length, n_boot=100)
        if pt is not None:
            pretrend.append({"campaign": c.CAMPAIGN, "pretrend_spend": pt["spend"], "pretrend_buy_pp": pt["buy"] * 100})
        # placebo: same number of random households "targeted"
        fake = rng.choice(hh, size=len(targeted), replace=False)
        pl = campaign_effect(spend, baskets, active_before, fake, hh, c.START_DAY, c.END_DAY, rng, n_boot=100)
        if pl is not None:
            placebo.append({"campaign": c.CAMPAIGN, "placebo_spend": pl["spend"], "placebo_buy_pp": pl["buy"] * 100})

    per = pd.DataFrame(rows)
    pt_df = pd.DataFrame(pretrend)
    per = per.merge(pt_df, on="campaign", how="left")
    # Trend-adjusted: remove the growth targeted households already showed BEFORE the campaign
    per["adj_uplift_spend_per_household"] = per["uplift_spend_per_household"] - per["pretrend_spend"]
    per["adj_uplift_buy_prob_pp"] = per["uplift_buy_prob_pp"] - per["pretrend_buy_pp"]
    per["week_normalised_uplift_spend"] = per["uplift_spend_per_household"] / per["weeks"]
    per["significant"] = (per["spend_ci_low"] > 0) | (per["spend_ci_high"] < 0)

    def pooled(g):
        w = g["targeted_households"]
        a = g.dropna(subset=["adj_uplift_spend_per_household"])
        return pd.Series({
            "campaigns": len(g),
            "campaigns_with_pretrend": len(a),
            "adj_uplift_spend_per_household": np.average(a["adj_uplift_spend_per_household"], weights=a["targeted_households"]),
            "adj_uplift_buy_prob_pp": np.average(a["adj_uplift_buy_prob_pp"], weights=a["targeted_households"]),
            "avg_uplift_spend_per_household": np.average(g["uplift_spend_per_household"], weights=w),
            "avg_uplift_buy_prob_pp": np.average(g["uplift_buy_prob_pp"], weights=w),
            "avg_control_spend": np.average(g["control_spend_per_household"], weights=w),
            "avg_control_buy_rate": np.average(g["control_buy_rate"], weights=w),
            "campaigns_with_significant_uplift": int((g["significant"] & (g["uplift_spend_per_household"] > 0)).sum()),
        })

    summary = pd.concat([
        pooled(per).to_frame().T.assign(group="All campaigns"),
        *[pooled(g).to_frame().T.assign(group=f"{t}") for t, g in per.groupby("type")],
    ])
    summary["relative_uplift_pct"] = summary["avg_uplift_spend_per_household"] / summary["avg_control_spend"] * 100
    summary["adj_relative_uplift_pct"] = summary["adj_uplift_spend_per_household"] / summary["avg_control_spend"] * 100

    checks = pd.DataFrame({
        "check": ["Pre-trend (should be ~0)", "Placebo / random targeting (should be ~0)"],
        "mean_spend_effect": [pd.DataFrame(pretrend)["pretrend_spend"].mean(), pd.DataFrame(placebo)["placebo_spend"].mean()],
        "mean_buy_prob_effect_pp": [pd.DataFrame(pretrend)["pretrend_buy_pp"].mean(), pd.DataFrame(placebo)["placebo_buy_pp"].mean()],
        "campaigns": [len(pretrend), len(placebo)],
    })

    # coupon redemptions: share of targeted households that redeemed anything in the campaign
    red = redemptions.groupby("CAMPAIGN")["household_key"].nunique()
    tgt = table.groupby("CAMPAIGN")["household_key"].nunique()
    per["redemption_rate_pct"] = per["campaign"].map((red / tgt * 100).fillna(0))

    out_dir.mkdir(parents=True, exist_ok=True)
    per.round(4).to_csv(out_dir / "dunnhumby_campaign_uplift.csv", index=False)
    summary.round(4).to_csv(out_dir / "dunnhumby_uplift_summary.csv", index=False)
    checks.round(4).to_csv(out_dir / "dunnhumby_uplift_checks.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print(checks.round(3).to_string(index=False))
    print(f"{int(per['significant'].sum())}/{len(per)} campaigns have a CI excluding zero; "
          f"{(per['uplift_spend_per_household'] > 0).sum()} positive")


if __name__ == "__main__":
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data_external" / "dunnhumby"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs"
    main(data, out)
