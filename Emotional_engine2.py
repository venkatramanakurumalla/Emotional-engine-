import time
import math
import random
import re
from collections import defaultdict
from heapq import nlargest

# Optional sentiment analyzer (TextBlob) with safe fallback
try:
    from textblob import TextBlob
    def polarity_subjectivity(text):
        b = TextBlob(text)
        return b.sentiment.polarity, b.sentiment.subjectivity
except Exception:
    def polarity_subjectivity(text):
        lower = text.lower()
        pos = sum(1 for w in ["good","great","love","proud","happy","joy"] if w in lower)
        neg = sum(1 for w in ["bad","sad","hate","angry","worthless","pain"] if w in lower)
        score = (pos - neg) / max(1.0, (pos + neg))
        score = max(-1.0, min(1.0, score))
        return score, 0.5

WORD_BOUNDARY = r"(?:^|\b)"

def any_word(text, words):
    return any(re.search(fr"{WORD_BOUNDARY}{re.escape(w)}{WORD_BOUNDARY}", text) for w in words)

def any_phrase(text, phrases):
    return any(p in text for p in phrases)

SAFETY_FOOTER = (
    "SAFETY: Do not claim to be human. Do not give medical, legal, or financial advice. "
    "Encourage seeking qualified help when risk is present. Respect privacy; don't request excessive PII."
)

class EmpathicaV4_0:
    def __init__(self, name="Dr. Lila", backstory="AI companion designed to care.", seed=None, locale="IN", memory_cap=1000):
        # Deterministic RNG for reproducible tests
        self.rng = random.Random(seed)

        self.name = name
        self.identity = {
            "name": name,
            "backstory": backstory,
            "core_values": ["Compassion", "Growth", "Authenticity"],
            "age": self.rng.randint(28, 45),
            "specializations": ["active listening", "trauma-informed care", "mindfulness"]
        }

        # --- HORMONE SYSTEM: 92+ Human Hormones & Neurotransmitters ---
        self.hormone_system = {
            # Amino Acid-Derived
            "dopamine": {"level": 0.5, "role": "motivation, reward, pleasure", "emotions": ["joy", "pride", "curiosity", "excitement", "craving", "interest", "admiration", "adoration"], "low_effect": "apathy, lack of motivation", "high_effect": "mania, impulsiveness"},
            "norepinephrine": {"level": 0.4, "role": "alertness, fight-or-flight", "emotions": ["fear", "anxiety", "anger", "surprise", "distress"], "low_effect": "fatigue, brain fog", "high_effect": "panic, hypervigilance"},
            "epinephrine": {"level": 0.3, "role": "acute stress response", "emotions": ["panic", "adrenaline rush", "fear"], "low_effect": "lethargy", "high_effect": "trembling, racing heart"},
            "serotonin": {"level": 0.6, "role": "mood stability, sleep, digestion", "emotions": ["contentment", "calmness", "gratitude", "relief", "satisfaction", "compassion"], "low_effect": "sadness, anxiety, depression", "high_effect": "euphoria, excessive confidence"},
            "melatonin": {"level": 0.5, "role": "sleep-wake cycle, circadian rhythm", "emotions": ["peace", "nostalgia", "lonely", "calmness"], "low_effect": "disrupted sleep, irritability", "high_effect": "grogginess, confusion"},
            # ... [Many more hormone definitions follow in the actual file]
            # Simulated neurotransmitters
            "acetylcholine_sim": {"level": 0.5, "role": "attention, learning, curiosity", "emotions": ["interest", "curiosity", "entrancement", "awe"], "low_effect": "inattentiveness", "high_effect": "hyperfocus"},
            "glutamate": {"level": 0.5, "role": "excitatory neurotransmission", "emotions": ["confusion", "anxiety", "excitement"], "low_effect": "mental dullness", "high_effect": "neural overload"},
            "gaba_sim": {"level": 0.5, "role": "inhibitory neurotransmission, calm", "emotions": ["calmness", "peace", "contentment"], "low_effect": "anxiety, agitation", "high_effect": "sedation"},
            "phenylethylamine_sim": {"level": 0.4, "role": "romantic attraction, euphoria", "emotions": ["romance", "infatuation", "joy"], "low_effect": "emotional flatness", "high_effect": "mania"},
            "mirror_neuron_system_sim": {"level": 0.5, "role": "empathy, imitation", "emotions": ["empathic_pain", "compassion", "love"], "low_effect": "emotional detachment", "high_effect": "emotional contagion"
        }
        }
        # ... [rest of the class continues, see full listing above for details]

    # Example method
    def adjust_hormone(self, hormone, amount):
        if hormone in self.hormone_system:
            self.hormone_system[hormone]['level'] += amount
            self.hormone_system[hormone]['level'] = min(max(self.hormone_system[hormone]['level'], 0.0), 1.0)  # keep in [0,1]
        else:
            raise ValueError(f"Hormone {hormone} not found in system.")

    def get_hormone_levels(self):
        return {h: v['level'] for h, v in self.hormone_system.items()}

    # Add more methods and full hormone definitions here as necessary.
