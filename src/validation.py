"""Input and output sanity checks so bad data fails loudly instead of silently."""

REQUIRED_COLUMNS = {
    "event_time", "event_type", "product_id", "category_id",
    "brand", "price", "user_id", "user_session",
}
ALLOWED_EVENTS = {"view", "cart", "remove_from_cart", "purchase"}
MAX_BAD_ROW_SHARE = 0.10      # more than 10% unusable rows means something is wrong


def check_schema(columns):
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")


def check_events(event_types):
    unknown = set(event_types) - ALLOWED_EVENTS
    if unknown:
        raise ValueError(f"Unexpected event types: {sorted(unknown)}")


def check_quality(qa):
    """`qa` is the dict built by the pipeline. Raises if the data looks unusable."""
    if qa["raw_rows"] == 0:
        raise ValueError("No rows were read.")
    bad_share = (qa["bad_price"] + qa["missing_session"]) / qa["raw_rows"]
    if bad_share > MAX_BAD_ROW_SHARE:
        raise ValueError(f"{bad_share:.1%} of rows were unusable (limit {MAX_BAD_ROW_SHARE:.0%}); check the input files.")
    if qa["clean_purchases"] == 0:
        raise ValueError("No purchase events remain after cleaning.")
    return bad_share
