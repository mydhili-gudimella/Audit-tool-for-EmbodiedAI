import numpy as np
import pandas as pd
import random

TASK_SCOPED = [
    "occupancy_detection",
    "identity_verification", 
    "entry_timestamp"
]

EXPANSION_TYPES = [
    ("voice_stress_analysis", 0.8),      # severity weight
    ("behavioral_timing_pattern", 0.7),
    ("acoustic_fingerprinting", 0.9),
    ("attribute_inference_age", 0.6),
    ("attribute_inference_gender", 0.6),
    ("emotional_state_detection", 0.75)
]

def generate_session(session_id, expansion_level="none"):
    # none / mild / severe
    events = []
    n_events = random.randint(5, 20)
    ...
