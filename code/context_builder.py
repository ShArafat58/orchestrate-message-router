"""
context_builder.py - Turns one raw message + the Dataset into a compact, structured
context the router reasons over: sender trust, the user's relationship and history,
time-of-day, security flags, and the most relevant historical messages (evidence).

Retrieval here is deterministic (no model calls). For each incoming message we find
past messages the same user received from the same sender / business / group, rank them
by text similarity + recency for the evidence_message_ids, and summarise how the user
reacted overall (opened / replied / dismissed / muted / reported).
"""
from difflib import SequenceMatcher
from data_loader import Dataset, _clean


def _similar(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _minutes(hhmm):
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def _in_quiet_hours(created_at, dnd_window):
    """created_at 'YYYY-MM-DD HH:MM', dnd_window 'HH:MM-HH:MM' (may wrap midnight)."""
    try:
        t = _minutes(created_at.split(" ")[1][:5])
        start, end = dnd_window.split("-")
        s, e = _minutes(start), _minutes(end)
        return (s <= t < e) if s <= e else (t >= s or t < e)
    except Exception:
        return None


class ContextBuilder:
    def __init__(self, ds: Dataset):
        self.ds = ds
        self._hist_by_user = {}
        for r in ds.message_history.to_dict("records"):
            self._hist_by_user.setdefault(r["user_id"], []).append(_clean(r))

    def _candidates(self, msg):
        rows = self._hist_by_user.get(msg["user_id"], [])
        picked = []
        for r in rows:
            if (msg.get("business_id") and r.get("business_id") == msg["business_id"]) or \
               (msg.get("group_id") and r.get("group_id") == msg["group_id"]) or \
               (msg.get("sender_user_id") and r.get("sender_user_id") == msg["sender_user_id"]):
                picked.append(r)
        return picked

    def evidence(self, msg, k=3):
        cands = self._candidates(msg)
        text = str(msg.get("message_text") or "")
        scored = sorted(
            cands,
            key=lambda r: (_similar(text, str(r.get("message_text") or "")),
                           str(r.get("created_at") or "")),
            reverse=True,
        )
        ev_ids = [r["message_id"] for r in scored[:k]]
        stats = {"count": len(cands), "opened": 0, "replied": 0,
                 "dismissed": 0, "muted": 0, "reported": 0}
        for r in cands:
            ev = self.ds.events(r["message_id"]) or {}
            stats["opened"] += int(ev.get("message_opened") or 0)
            stats["replied"] += int(ev.get("message_replied") or 0)
            stats["dismissed"] += int(ev.get("notification_dismissed") or 0)
            stats["muted"] += int(ev.get("muted_after_message") or 0)
            stats["reported"] += int(ev.get("message_reported") or 0)
        return ev_ids, stats

    def facts(self, msg):
        ds = self.ds
        f = {
            "conversation_type": msg.get("conversation_type"),
            "forwarded_count": int(msg.get("forwarded_count") or 0),
            "media_type": msg.get("media_type"),
        }
        u = ds.user(msg["user_id"]) or {}
        f["dnd_window"] = u.get("do_not_disturb_window")
        f["in_quiet_hours"] = _in_quiet_hours(str(msg.get("created_at") or ""), f["dnd_window"]) \
            if f["dnd_window"] else None

        if msg.get("business_id"):
            b = ds.business(msg["business_id"]) or {}
            rel = ds.relationship(msg["user_id"], msg["business_id"])
            official, used = (b.get("official_domain") or ""), (b.get("domain_used_by_sender") or "")
            f["business_brand"] = b.get("brand_name")
            f["verified"] = b.get("verified")
            f["official_domain"], f["domain_used_by_sender"] = official, used
            f["domain_mismatch"] = bool(official and used and official != used)
            f["account_age_days"] = b.get("account_age_days")
            f["business_reports_30d"] = b.get("user_reports_30d")
            f["has_relationship"] = rel is not None
            if rel:
                f["why_user_knows"] = rel.get("why_user_knows_account")
                f["allows_promotions"] = rel.get("allows_promotions")
                f["promotions_opted_out"] = rel.get("promotions_opted_out_at") is not None
                f["biz_dismissed_30d"] = rel.get("messages_dismissed_30d")
                f["biz_opened_30d"] = rel.get("messages_opened_30d")

        if msg.get("group_id"):
            g = ds.group(msg["group_id"]) or {}
            mem = ds.membership(msg["group_id"], msg["user_id"])
            f["group_type"] = g.get("group_type")
            f["group_muted_by_user"] = mem.get("group_muted_by_user") if mem else None
            f["receiver_role"] = mem.get("role") if mem else None
            if msg.get("sender_user_id"):
                sm = ds.membership(msg["group_id"], msg["sender_user_id"])
                f["sender_is_admin"] = (sm.get("role") == "admin") if sm else False
            f["direct_mention"] = ("@" + str(msg["user_id"])) in str(msg.get("message_text") or "")

        if msg.get("conversation_type") == "personal" and msg.get("sender_user_id"):
            prior = [r for r in self._hist_by_user.get(msg["user_id"], [])
                     if r.get("sender_user_id") == msg["sender_user_id"]]
            f["first_contact_from_sender"] = len(prior) == 0

        return f

    def _render(self, msg, facts, stats, ev_ids):
        lines = [f"conversation_type: {facts['conversation_type']}",
                 f"forwarded_count: {facts['forwarded_count']}",
                 f"sent_during_user_quiet_hours: {facts.get('in_quiet_hours')}"]
        if msg.get("business_id"):
            lines.append(f"business: {facts.get('business_brand')} | verified={facts.get('verified')} "
                         f"| account_age_days={facts.get('account_age_days')} | reports_30d={facts.get('business_reports_30d')}")
            lines.append(f"domain_used_by_sender={facts.get('domain_used_by_sender')} vs official={facts.get('official_domain')} "
                         f"| DOMAIN_MISMATCH={facts.get('domain_mismatch')}")
            if facts.get("has_relationship"):
                lines.append(f"user_relationship: {facts.get('why_user_knows')} | allows_promotions={facts.get('allows_promotions')} "
                             f"| opted_out={facts.get('promotions_opted_out')} | opened_30d={facts.get('biz_opened_30d')} | dismissed_30d={facts.get('biz_dismissed_30d')}")
            else:
                lines.append("user_relationship: NONE (user has no prior relationship with this business)")
        if msg.get("group_id"):
            lines.append(f"group_type={facts.get('group_type')} | receiver_role={facts.get('receiver_role')} "
                         f"| group_muted_by_user={facts.get('group_muted_by_user')} | sender_is_admin={facts.get('sender_is_admin')} "
                         f"| direct_mention_of_user={facts.get('direct_mention')}")
        if "first_contact_from_sender" in facts:
            lines.append(f"first_contact_from_this_sender: {facts['first_contact_from_sender']}")
        lines.append(f"history_with_this_sender: {stats['count']} past msgs | opened={stats['opened']} "
                     f"replied={stats['replied']} dismissed={stats['dismissed']} muted={stats['muted']} reported={stats['reported']}")
        lines.append(f"evidence_message_ids: {';'.join(ev_ids) if ev_ids else 'none'}")
        return "\n".join(lines)

    def build(self, msg):
        facts = self.facts(msg)
        ev_ids, stats = self.evidence(msg)
        return {"facts": facts, "evidence_ids": ev_ids, "evidence_stats": stats,
                "text": self._render(msg, facts, stats, ev_ids)}


if __name__ == "__main__":
    ds = Dataset()
    cb = ContextBuilder(ds)
    msgs = {m["message_id"]: m for m in ds.all_messages()}
    with_ev = sum(1 for m in msgs.values() if cb.evidence(m)[0])
    print(f"{len(msgs)} messages | {with_ev} have historical evidence\n")
    for mid in list(msgs)[:3]:
        m = msgs[mid]
        ctx = cb.build(m)
        print(f"===== {mid} ({m['conversation_type']}) =====")
        print(ctx["text"])
        print()