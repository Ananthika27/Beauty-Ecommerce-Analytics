import pandas as pd
import pytest

import basket
import churn_model as cm
import pipeline


def toy_purchases():
    rows = [
        # user, time, session, brand, category, price
        (1, "2019-12-05", "a", "x", 10, 10.0),
        (1, "2019-12-05", "a", "y", 11, 5.0),      # same order, 2 items
        (1, "2020-01-10", "b", "x", 10, 10.0),
        (2, "2019-12-20", "c", "x", 10, 20.0),
        (3, "2020-02-10", "d", "y", 11, 7.0),      # only buys after the snapshot
    ]
    df = pd.DataFrame(rows, columns=["user_id", "event_time", "user_session", "brand", "category_id", "price"])
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


@pytest.mark.parametrize("r,f,m,expected", [
    (5, 5, 5, "Champions"),
    (3, 3, 1, "Loyal Customers"),
    (5, 1, 1, "New Customers"),
    (4, 2, 1, "Potential Loyalists"),
    (1, 4, 5, "At Risk High Value"),
    (2, 2, 1, "At Risk"),
    (1, 1, 4, "High Value Lost"),
    (3, 1, 1, "Regular Customers"),
])
def test_segment_rules(r, f, m, expected):
    assert pipeline.assign_segment(r, f, m) == expected


def test_every_score_combination_gets_a_segment():
    segments = {pipeline.assign_segment(r, f, m) for r in range(1, 6) for f in range(1, 6) for m in range(1, 6)}
    assert len(segments) == 8


def test_frequency_counts_orders_not_items():
    p = toy_purchases()
    p["event_time"] = pd.to_datetime(p["event_time"], utc=True)
    orders, summary = pipeline.repeat_purchase(p)
    user1 = orders.set_index("user_id").loc[1]
    assert user1["items_purchased"] == 3 and user1["orders"] == 2


def test_features_use_only_past_data():
    p = toy_purchases()
    base = cm.build_features(p, "2020-02-01")
    # Adding future purchases must not change features for the snapshot
    future = pd.DataFrame([(1, pd.Timestamp("2020-02-15"), "z", "x", 10, 999.0)], columns=p.columns)
    changed = cm.build_features(pd.concat([p, future]), "2020-02-01")
    pd.testing.assert_frame_equal(base, changed)
    assert 3 not in base.index            # first purchase is after the snapshot


def test_labels_are_next_month_purchases():
    p = toy_purchases()
    X = cm.build_features(p, "2020-02-01")
    y = cm.build_labels(p, "2020-02-01", X.index)
    assert y.to_dict() == {1: 0, 2: 0}    # neither bought in February
    p2 = pd.concat([p, pd.DataFrame([(2, pd.Timestamp("2020-02-03"), "e", "x", 10, 5.0)], columns=p.columns)])
    assert cm.build_labels(p2, "2020-02-01", X.index).to_dict() == {1: 0, 2: 1}


def test_top_decile_lift_perfect_ranking():
    y = pd.Series([1] * 10 + [0] * 90).values
    scores = pd.Series(range(100, 0, -1)).values
    assert cm.top_decile_lift(y, scores) == pytest.approx(10.0)


def test_basket_lift():
    rows = []
    for i in range(200):                       # x and y always together
        rows += [(f"s{i}", "x"), (f"s{i}", "y")]
    for i in range(200, 400):                  # z and w together elsewhere
        rows += [(f"s{i}", "z"), (f"s{i}", "w")]
    df = pd.DataFrame(rows, columns=["user_session", "brand"])
    pairs, n = basket.brand_pairs(df, min_pair_orders=10)
    assert n == 400 and len(pairs) == 2
    assert pairs["lift"].iloc[0] == pytest.approx(2.0)   # 0.5 / (0.5 * 0.5)
