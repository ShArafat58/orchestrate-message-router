"""
data_loader.py - Loads and indexes every dataset CSV for the Message Notification Router.

Gives O(1) lookups and the core joins needed to assemble per-message context:
user profile, group + this user's membership, business + this user's relationship,
historical messages, how the user reacted to them, and media file paths.
No model calls here - this is the deterministic data layer.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"


def _clean(row: dict) -> dict:
    """Plain dict with NaN turned into None so downstream code never sees NaN."""
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


class Dataset:
    """Loads all CSVs once and exposes indexed lookups + joins."""

    def __init__(self, dataset_dir=DATASET_DIR):
        self.dir = Path(dataset_dir)
        self._load()
        self._index()

    def _read(self, name):
        return pd.read_csv(self.dir / f"{name}.csv")

    def _load(self):
        self.messages = self._read("messages")
        self.users = self._read("users")
        self.groups = self._read("groups")
        self.group_members = self._read("group_members")
        self.businesses = self._read("business_accounts")
        self.user_business = self._read("user_business_history")
        self.message_history = self._read("message_history")
        self.message_events = self._read("message_events")
        self.images = self._read("images")
        self.voice_notes = self._read("voice_notes")
        self.daily_summary = self._read("daily_notification_summary")

    def _index(self):
        self._users = {r["user_id"]: _clean(r) for r in self.users.to_dict("records")}
        self._groups = {r["group_id"]: _clean(r) for r in self.groups.to_dict("records")}
        self._members = {(r["group_id"], r["user_id"]): _clean(r)
                         for r in self.group_members.to_dict("records")}
        self._businesses = {r["business_id"]: _clean(r) for r in self.businesses.to_dict("records")}
        self._relationship = {(r["user_id"], r["business_id"]): _clean(r)
                              for r in self.user_business.to_dict("records")}
        self._history = {r["message_id"]: _clean(r) for r in self.message_history.to_dict("records")}
        self._events = {r["message_id"]: _clean(r) for r in self.message_events.to_dict("records")}
        self._images = {r["image_id"]: r["file_path"] for r in self.images.to_dict("records")}
        self._voice = {r["voice_note_id"]: r["file_path"] for r in self.voice_notes.to_dict("records")}

    # ---- single-record lookups ----
    def user(self, user_id):
        return self._users.get(user_id)

    def group(self, group_id):
        return self._groups.get(group_id)

    def membership(self, group_id, user_id):
        return self._members.get((group_id, user_id))

    def business(self, business_id):
        return self._businesses.get(business_id)

    def relationship(self, user_id, business_id):
        return self._relationship.get((user_id, business_id))

    def history(self, message_id):
        return self._history.get(message_id)

    def events(self, message_id):
        return self._events.get(message_id)

    def media_path(self, media_type, media_id):
        if not media_id:
            return None
        rel = self._images.get(media_id) if media_type == "image" else \
              self._voice.get(media_id) if media_type == "voice" else None
        return str(self.dir / rel) if rel else None

    def all_messages(self):
        return [_clean(r) for r in self.messages.to_dict("records")]


if __name__ == "__main__":
    ds = Dataset()
    print("Loaded tables:")
    for name in ["messages", "users", "groups", "group_members", "businesses",
                 "user_business", "message_history", "message_events",
                 "images", "voice_notes", "daily_summary"]:
        print(f"  {name:16} {len(getattr(ds, name)):>4} rows")

    print("\nSample join for the first message:")
    msg = ds.all_messages()[0]
    print("  message_id   :", msg["message_id"])
    print("  user_id      :", msg["user_id"], "| conversation:", msg["conversation_type"])
    print("  media_type   :", msg["media_type"], "| media_path:",
          ds.media_path(msg["media_type"], msg["media_id"]))
    u = ds.user(msg["user_id"])
    print("  user DND     :", u["do_not_disturb_window"] if u else None)
    if msg["business_id"]:
        b = ds.business(msg["business_id"])
        rel = ds.relationship(msg["user_id"], msg["business_id"])
        print("  business     :", b["brand_name"], "| verified:", b["verified"])
        print("  domain used  :", b["domain_used_by_sender"], "vs official:", b["official_domain"])
        print("  relationship :", rel["why_user_knows_account"] if rel else "NONE (new/unknown sender)")
    if msg["group_id"]:
        g = ds.group(msg["group_id"])
        mem = ds.membership(msg["group_id"], msg["user_id"])
        print("  group        :", g["group_name"], "| type:", g["group_type"])
        print("  membership   :", f"role={mem['role']}, muted_by_user={mem['group_muted_by_user']}" if mem else "NONE")

    print("\nData layer OK")