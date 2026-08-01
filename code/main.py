"""
main.py - Entry point. Routes every message in dataset/messages.csv and writes
dataset/output.csv with the required schema:
message_id, action, message_type, reason, confidence, evidence_message_ids
"""
import csv
from data_loader import Dataset, DATASET_DIR
from router import Router

OUTPUT_PATH = DATASET_DIR / "output.csv"
COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def main():
    ds = Dataset()
    router = Router(ds)
    messages = ds.all_messages()
    rows = []
    for i, m in enumerate(messages, 1):
        row = router.route(m)
        rows.append(row)
        print(f"[{i:3}/{len(messages)}] {row['message_id']:8} {row['action']:6} "
              f"{row['message_type']:15} conf={row['confidence']}")
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()