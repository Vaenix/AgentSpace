"""Map the local review windows back to seconds in the original recording."""

CLEAN_WINDOWS = {("P30_08-19", 2): (133.133, 145.760)}


def review_bounds(pair, event):
    clean = CLEAN_WINDOWS.get((pair["pair_id"], event["ordinal"]))
    if clean:
        return clean
    start = max(0, event["sound_s"][0] - 3)
    end = min(pair["duration_s"], max(event["action_s"][0] + 5, event["sound_s"][1] + 6))
    return start, end
