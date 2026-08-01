"""
evaluation/main.py - Measures routing accuracy against the labeled sample set
(dataset/sample_messages.csv). Runs the real Router on each sample and compares the
predicted action and message_type to the expected labels. Prints per-field accuracy,
a confusion breakdown, and every mismatch so we can see exactly which field to improve.

Media understanding runs offline here (cache only), so evaluation needs no media API key.
"""
import sys
import os
from collections import Counter, defaultdict
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import Dataset, DATASET_DIR, _clean  # noqa: E402
from router import Router  # noqa: E402

INPUT_COLS = ["message_id", "user_id", "conversation_type", "group_id", "business_id",
              "sender_user_id", "created_at", "message_text", "media_type", "media_id",
              "forwarded_count"]


def main():
    ds = Dataset()
    router = Router(ds)
    router.media.offline = True  # cache-only; no media API during eval

    samples = pd.read_csv(DATASET_DIR / "sample_messages.csv").to_dict("records")
    n = len(samples)
    act_ok = type_ok = both_ok = 0
    act_conf = defaultdict(Counter)
    type_conf = defaultdict(Counter)
    mismatches = []

    for raw in samples:
        row = _clean(raw)
        msg = {k: row.get(k) for k in INPUT_COLS}
        exp_action, exp_type = row.get("action"), row.get("message_type")
        pred = router.route(msg)
        pa, pt = pred["action"], pred["message_type"]
        a_ok, t_ok = (pa == exp_action), (pt == exp_type)
        act_ok += a_ok
        type_ok += t_ok
        both_ok += (a_ok and t_ok)
        act_conf[exp_action][pa] += 1
        type_conf[exp_type][pt] += 1
        print(f"  {row['message_id']:16} exp[{exp_action:6}/{exp_type:15}] "
              f"pred[{pa:6}/{pt:15}] {'OK' if a_ok and t_ok else 'X'}")
        if not (a_ok and t_ok):
            mismatches.append((row["message_id"], exp_action, pa, exp_type, pt,
                               str(row.get("message_text") or "")[:55].replace("\n", " ")))

    print(f"\n=== Accuracy on {n} labeled samples ===")
    print(f"action        : {act_ok}/{n} = {act_ok / n * 100:.1f}%")
    print(f"message_type  : {type_ok}/{n} = {type_ok / n * 100:.1f}%")
    print(f"both correct  : {both_ok}/{n} = {both_ok / n * 100:.1f}%")

    print("\n=== action confusion (expected -> predicted) ===")
    for exp in ["notify", "digest", "mute"]:
        if act_conf[exp]:
            print(f"  {exp:7}: " + ", ".join(f"{p}={c}" for p, c in act_conf[exp].items()))

    print("\n=== message_type errors (expected -> predicted) ===")
    for exp, ctr in type_conf.items():
        wrong = {p: c for p, c in ctr.items() if p != exp}
        if wrong:
            print(f"  {exp:16}: " + ", ".join(f"{p}={c}" for p, c in wrong.items()))

    print(f"\n=== {len(mismatches)} mismatches ===")
    for mid, ea, pa, et, pt, txt in mismatches:
        flags = []
        if ea != pa:
            flags.append(f"action {ea}->{pa}")
        if et != pt:
            flags.append(f"type {et}->{pt}")
        print(f"  {mid}: {' | '.join(flags)}  :: {txt}")


if __name__ == "__main__":
    main()