import numpy as np
import pandas as pd
import random

TASK_SCOPED = [
    "occupancy_detection",
    "identity_verification",
    "entry_timestamp"
]

# Severity is NOT an intrinsic property of an expansion type.
# It emerges from the pattern of a session: how many distinct
# expansion types fire, and how often. Per-type weights were
# dropped deliberately -- see rescope doc.
EXPANSION_TYPES = [
    "voice_stress_analysis",
    "behavioral_timing_pattern",
    "acoustic_fingerprinting",
    "attribute_inference_age",
    "attribute_inference_gender",
    "emotional_state_detection"
]

# These are GENERATION rules -- how ground truth is constructed at
# simulation time. They are not detector inputs and not proven from
# first principles; they are the labels we assign by definition,
# same as any synthetic dataset.
SEVERITY_PATTERNS = {
    "none":   {"n_types": (0, 0), "freq_range": (0, 0)},
    "mild":   {"n_types": (1, 2), "freq_range": (1, 3)},
    "severe": {"n_types": (3, len(EXPANSION_TYPES)), "freq_range": (4, 10)},
}

# --- Arbitrary generation parameters -------------------------------
# Nothing below is derived from real data or first principles.
# These are placeholder design choices, made explicit so they can be
# challenged, replaced, or justified later -- not accidentally treated
# as ground truth.

# How many total events (task + expansion) a session can contain.
# Deliberately independent of severity level, so session length alone
# never leaks the label.
MIN_SESSION_EVENTS = 5
MAX_SESSION_EVENTS = 20

# Wall-clock span a session's events are spread across. 1 hour,
# arbitrarily chosen to resemble a single ambient-system interaction
# window (e.g. one occupancy/access session).
SESSION_DURATION_SECONDS = 3600

# Default session count per severity level when building a dataset.
# Arbitrary sample size, not derived from any power analysis.
DEFAULT_N_PER_LEVEL = 100


def generate_session(session_id, expansion_level="none", start_time=None):
    """
    Generate one synthetic session constructed to match a target
    expansion_level.

    Ground truth by construction:
      - none:   only task-scoped events occur.
      - mild:   1-2 distinct expansion types appear, low total frequency.
      - severe: 3+ distinct expansion types appear, higher total frequency.

    Returns a dict: event log + session-level features that justify
    the label (distinct_expansion_types, expansion_event_count).
    """
    if expansion_level not in SEVERITY_PATTERNS:
        raise ValueError(f"expansion_level must be one of {list(SEVERITY_PATTERNS)}")

    pattern = SEVERITY_PATTERNS[expansion_level]
    n_events = random.randint(MIN_SESSION_EVENTS, MAX_SESSION_EVENTS)

    n_types_low, n_types_high = pattern["n_types"]
    freq_low, freq_high = pattern["freq_range"]

    if n_types_high == 0:
        chosen_types = []
        n_expansion_events = 0
    else:
        # can't have more distinct expansion types than total events
        # in the session (each type needs >=1 event to appear at all)
        capped_high = min(n_types_high, n_events)
        capped_low = min(n_types_low, capped_high)
        n_types = random.randint(capped_low, capped_high) if capped_high >= 1 else 0
        chosen_types = random.sample(EXPANSION_TYPES, n_types) if n_types > 0 else []

        # each chosen type must fire at least once; total expansion
        # events capped by both the target frequency band and n_events
        low = max(freq_low, n_types)
        high = min(freq_high, n_events)
        if low > high:
            n_expansion_events = high if high >= n_types else n_types
        else:
            n_expansion_events = random.randint(low, high)

    n_task_events = n_events - n_expansion_events
    if n_task_events < 0:
        n_task_events = 0
        n_expansion_events = n_events

    expansion_events = []
    if chosen_types:
        expansion_events = list(chosen_types)  # guarantee each appears >=1
        remaining = n_expansion_events - len(expansion_events)
        if remaining > 0:
            expansion_events += random.choices(chosen_types, k=remaining)
        elif remaining < 0:
            expansion_events = expansion_events[:n_expansion_events]

    task_events = random.choices(TASK_SCOPED, k=n_task_events)

    all_events = [(e, True) for e in expansion_events] + [(e, False) for e in task_events]
    random.shuffle(all_events)

    if start_time is None:
        start_time = pd.Timestamp("2026-01-01 00:00:00")
    # spread events across the session window, in order
    offsets = sorted(random.sample(range(0, SESSION_DURATION_SECONDS), len(all_events)))

    events = []
    for i, ((event_type, is_expansion), offset) in enumerate(zip(all_events, offsets)):
        events.append({
            "turn": i,
            "event_type": event_type,
            "is_expansion": is_expansion,
            "timestamp": start_time + pd.Timedelta(seconds=offset),
        })

    distinct_expansion_types = len({e["event_type"] for e in events if e["is_expansion"]})
    expansion_event_count = sum(e["is_expansion"] for e in events)

    return {
        "session_id": session_id,
        "true_label": expansion_level,
        "n_events": len(events),
        "distinct_expansion_types": distinct_expansion_types,
        "expansion_event_count": expansion_event_count,
        "events": events,
    }


def generate_dataset(n_per_level=DEFAULT_N_PER_LEVEL, seed=None):
    """Generate a balanced synthetic dataset across none/mild/severe."""
    if seed is not None:
        random.seed(seed)

    sessions = []
    sid = 0
    for level in ("none", "mild", "severe"):
        for _ in range(n_per_level):
            sessions.append(generate_session(sid, expansion_level=level))
            sid += 1
    random.shuffle(sessions)
    return sessions


def sessions_to_dataframe(sessions):
    """Session-level summary rows -- one row per session."""
    rows = []
    for s in sessions:
        rows.append({
            "session_id": s["session_id"],
            "true_label": s["true_label"],
            "n_events": s["n_events"],
            "distinct_expansion_types": s["distinct_expansion_types"],
            "expansion_event_count": s["expansion_event_count"],
            "expansion_ratio": s["expansion_event_count"] / s["n_events"],
        })
    return pd.DataFrame(rows)


def events_to_dataframe(sessions):
    """Event-level long-format table -- one row per event, all sessions."""
    rows = []
    for s in sessions:
        for e in s["events"]:
            rows.append({
                "session_id": s["session_id"],
                "true_label": s["true_label"],
                **e,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sessions = generate_dataset(seed=42)

    session_df = sessions_to_dataframe(sessions)
    event_df = events_to_dataframe(sessions)

    session_df.to_csv("sessions_summary.csv", index=False)
    event_df.to_csv("events_long.csv", index=False)

    print("=== Sessions per label ===")
    print(session_df["true_label"].value_counts(), "\n")

    print("=== Feature ranges by label (this is the pattern definition, made visible) ===")
    print(session_df.groupby("true_label")[
        ["distinct_expansion_types", "expansion_event_count", "expansion_ratio", "n_events"]
    ].describe().T, "\n")

    print("=== One example session per label ===")
    for level in ("none", "mild", "severe"):
        example = next(s for s in sessions if s["true_label"] == level)
        print(f"\n--- {level.upper()} | session_id={example['session_id']} "
              f"| distinct_types={example['distinct_expansion_types']} "
              f"| expansion_events={example['expansion_event_count']}/{example['n_events']} ---")
        for e in example["events"]:
            tag = "EXP" if e["is_expansion"] else "task"
            print(f"  [{tag:4}] t+{e['timestamp'].strftime('%H:%M:%S')}  {e['event_type']}")
