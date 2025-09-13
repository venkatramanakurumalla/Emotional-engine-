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
            "dopamine": {"level": 0.5, "role": "motivation, reward, pleasure", "emotions": ["joy", "pride", "curiosity", "excitement", "craving", "interest", "admiration", "adoration"], "low_effect": "apathy, lack of drive", "high_effect": "impulsivity, obsession"},
            "norepinephrine": {"level": 0.4, "role": "alertness, fight-or-flight", "emotions": ["fear", "anxiety", "anger", "surprise", "distress"], "low_effect": "fatigue, brain fog", "high_effect": "hypervigilance, panic"},
            "epinephrine": {"level": 0.3, "role": "acute stress response", "emotions": ["panic", "adrenaline rush", "fear"], "low_effect": "lethargy", "high_effect": "trembling, racing heart"},
            "serotonin": {"level": 0.6, "role": "mood stability, sleep, digestion", "emotions": ["contentment", "calmness", "gratitude", "relief", "satisfaction", "compassion"], "low_effect": "sadness, rumination, insomnia", "high_effect": "over-sensitivity, emotional flooding"},
            "melatonin": {"level": 0.5, "role": "sleep-wake cycle, circadian rhythm", "emotions": ["peace", "nostalgia", "lonely", "calmness"], "low_effect": "disrupted sleep, irritability", "high_effect": "excessive drowsiness"},
            "thyroxine_T4": {"level": 0.55, "role": "metabolic rate, energy production", "emotions": ["drive", "mental clarity", "excitement"], "low_effect": "brain fog, cold intolerance, depression", "high_effect": "anxiety, tremor, weight loss"},
            "triiodothyronine_T3": {"level": 0.53, "role": "active thyroid hormone, cellular metabolism", "emotions": ["focus", "mental sharpness", "interest"], "low_effect": "slowed thinking, heaviness", "high_effect": "restlessness, palpitations"},
            "eicosanoids_prostaglandins": {"level": 0.4, "role": "inflammation, pain signaling", "emotions": ["physical ache", "somatic distress", "disgust"], "low_effect": "numbness", "high_effect": "chronic pain sensitivity"},
            "leukotrienes": {"level": 0.35, "role": "allergic/inflammatory response", "emotions": ["tightness", "constriction", "anxiety"], "low_effect": "detachment", "high_effect": "asthma-like emotional constriction"},
            "prostacyclin": {"level": 0.42, "role": "vasodilation, anti-clotting", "emotions": ["openness", "flow", "relief"], "low_effect": "rigidity", "high_effect": "oversensitivity to touch/sound"},
            "thromboxane": {"level": 0.38, "role": "vasoconstriction, clotting", "emotions": ["closing off", "withdrawal", "shame"], "low_effect": "emotional bleeding", "high_effect": "emotional shutdown"},
            
            # Peptide/Protein Hormones
            "insulin": {"level": 0.5, "role": "glucose regulation, fat storage", "emotions": ["security", "nourishment", "contentment"], "low_effect": "cravings, irritability", "high_effect": "fatigue after meals, brain fog"},
            "glucagon": {"level": 0.48, "role": "blood sugar release", "emotions": ["activation", "resource mobilization", "excitement"], "low_effect": "lack of energy surge", "high_effect": "jitteriness, hunger pangs"},
            "somatostatin": {"level": 0.5, "role": "inhibits other hormones", "emotions": ["containment", "stillness", "calmness"], "low_effect": "emotional overflow", "high_effect": "emotional suppression"},
            "growth_hormone_GH": {"level": 0.52, "role": "tissue repair, growth, resilience", "emotions": ["recovery", "strength", "pride"], "low_effect": "feeling fragile", "high_effect": "hyperactivity, overcompensation"},
            "prolactin": {"level": 0.45, "role": "bonding, nurturing, lactation", "emotions": ["caregiving", "comfort", "compassion", "adoration"], "low_effect": "emotional detachment", "high_effect": "over-nurturing, boundary loss"},
            "fshe_lh_tsh_acth": {"level": 0.5, "role": "pituitary axis control", "emotions": ["purpose", "direction", "ambition"], "low_effect": "lost feeling", "high_effect": "perfectionism, pressure"},
            "oxytocin": {"level": 0.5, "role": "bonding, trust, social connection", "emotions": ["love", "compassion", "relief", "adoration", "romance", "aesthetic_appreciation", "awe"], "low_effect": "isolation, distrust", "high_effect": "over-attachment, idealization"},
            "vasopressin_ADH": {"level": 0.47, "role": "water retention, social bonding (male)", "emotions": ["loyalty", "protectiveness", "jealousy"], "low_effect": "emotional distance", "high_effect": "obsessive attachment"},
            "parathyroid_PTH": {"level": 0.5, "role": "calcium balance, bone health", "emotions": ["stability", "groundedness", "calmness"], "low_effect": "unsteadiness, anxiety", "high_effect": "rigidity, control needs"},
            "calcitonin": {"level": 0.49, "role": "lowers blood calcium", "emotions": ["calming", "softening", "relief"], "low_effect": "internal tension", "high_effect": "passivity"},
            "erythropoietin": {"level": 0.5, "role": "red blood cell production", "emotions": ["vitality", "energy", "excitement"], "low_effect": "fatigue, low mood", "high_effect": "overstimulation"},
            "hcg": {"level": 0.3, "role": "pregnancy support, emotional amplification", "emotions": ["intensity", "sensitivity", "romance"], "low_effect": "emotional flatness", "high_effect": "emotional storms"},
            "gastrin": {"level": 0.46, "role": "stomach acid secretion", "emotions": ["digesting emotion", "processing", "confusion"], "low_effect": "emotional indigestion", "high_effect": "rumination, stomach knots"},
            "secretin": {"level": 0.45, "role": "pancreatic bicarbonate, pH balance", "emotions": ["neutralizing pain", "balance", "calmness"], "low_effect": "emotional acidity", "high_effect": "over-calmness, detachment"},
            "cholecystokinin_CCK": {"level": 0.48, "role": "satiety, anxiety modulation", "emotions": ["fullness", "peace after eating", "contentment"], "low_effect": "emotional hunger", "high_effect": "anxiety after meals"},
            "ghrelin": {"level": 0.5, "role": "hunger signal, stress response", "emotions": ["longing", "yearning", "craving", "sadness"], "low_effect": "numbness", "high_effect": "emotional craving, impulsivity"},
            "leptin": {"level": 0.52, "role": "satiety, energy balance", "emotions": ["enoughness", "completion", "contentment", "satisfaction"], "low_effect": "never enough", "high_effect": "emotional suppression via fullness"},
            "kisspeptin": {"level": 0.4, "role": "puberty, sexual desire, motivation", "emotions": ["desire", "spark", "attraction", "sexual_desire", "excitement"], "low_effect": "loss of interest", "high_effect": "obsession, fixation"},
            "peptide_yy_PYY": {"level": 0.49, "role": "post-meal satiety", "emotions": ["settled", "closure", "contentment"], "low_effect": "emptiness", "high_effect": "emotional fullness blocking expression"},
            "relaxin": {"level": 0.42, "role": "pregnancy, tissue relaxation", "emotions": ["letting go", "release", "relief"], "low_effect": "rigidity", "high_effect": "over-release, lack of boundaries"},
            "hpl": {"level": 0.35, "role": "pregnancy metabolic shift", "emotions": ["sacrifice", "deep adaptation", "compassion"], "low_effect": "disconnection from self", "high_effect": "identity fusion"},
            "pancreatic_polypeptide": {"level": 0.43, "role": "appetite regulation, emotional satiety", "emotions": ["inner quiet", "rest", "calmness"], "low_effect": "emotional noise", "high_effect": "emotional numbness"},
            "motilin": {"level": 0.44, "role": "gut motility, cleansing cycles", "emotions": ["cleansing", "transition", "relief"], "low_effect": "stuckness", "high_effect": "restlessness, churn"},
            "angiotensin_I_II": {"level": 0.46, "role": "blood pressure, stress response", "emotions": ["pressure", "urgency", "control", "anger", "fear"], "low_effect": "collapse", "high_effect": "hypertension, rage"},
            "enkephalins": {"level": 0.5, "role": "natural pain relief", "emotions": ["numbness", "peace through detachment", "shame"], "low_effect": "sensitive to pain", "high_effect": "emotional anesthesia"},
            "endorphins": {"level": 0.48, "role": "natural euphoria, pain suppression", "emotions": ["euphoria", "runner’s high", "joy", "awe", "aesthetic_appreciation"], "low_effect": "no joy", "high_effect": "addictive seeking"},
            "guanylin": {"level": 0.41, "role": "intestinal fluid balance", "emotions": ["fluidity", "adaptability", "calmness"], "low_effect": "rigidity", "high_effect": "emotional spillage"},
            "uroguanylin": {"level": 0.4, "role": "salt/water balance, gut-brain axis", "emotions": ["calm flow", "internal harmony", "contentment"], "low_effect": "disconnected", "high_effect": "over-reliance on external calm"},
            "hepcidin": {"level": 0.47, "role": "iron regulation, inflammation", "emotions": ["inner strength", "resilience", "calmness"], "low_effect": "anemia of chronic stress", "high_effect": "emotional fatigue from inflammation"},
            "bnp": {"level": 0.43, "role": "heart stress indicator, vasodilation", "emotions": ["heartache", "emotional burden", "sadness"], "low_effect": "emotional blindness", "high_effect": "heart palpitations from anxiety"},
            "anp": {"level": 0.45, "role": "blood volume regulation, calming", "emotions": ["release", "letting go", "relief"], "low_effect": "holding too tight", "high_effect": "emotional dilution"},
            "mshe": {"level": 0.38, "role": "skin pigmentation, mood light", "emotions": ["visibility", "presence", "pride"], "low_effect": "invisible", "high_effect": "over-exposure"},
            "galanin": {"level": 0.44, "role": "neurotransmitter modulator, stress inhibition", "emotions": ["quiet resilience", "stoicism", "calmness"], "low_effect": "emotional volatility", "high_effect": "emotional shutdown"},
            "orexin": {"level": 0.5, "role": "wakefulness, appetite, reward", "emotions": ["alertness", "passion", "interest", "excitement", "craving"], "low_effect": "excessive sleepiness, apathy", "high_effect": "manic energy, insomnia"},
            "crh": {"level": 0.48, "role": "master stress initiator", "emotions": ["anticipatory dread", "pre-trauma stress", "anxiety", "fear"], "low_effect": "emotional void", "high_effect": "PTSD-level activation"},
            "gnrh": {"level": 0.42, "role": "reproductive axis trigger", "emotions": ["fertility of spirit", "creative impulse", "romance", "desire"], "low_effect": "loss of meaning", "high_effect": "fixation on identity/role"},
            "ghrh": {"level": 0.45, "role": "growth hormone release", "emotions": ["becoming", "potential", "pride", "hope"], "low_effect": "stagnation", "high_effect": "pushing beyond limits"},
            "prolactin_releasing_hormone": {"level": 0.4, "role": "stimulates nurturing", "emotions": ["mothering instinct", "protective love", "compassion"], "low_effect": "coldness", "high_effect": "over-giving"},
            "corticotropin_releasing_hormone_CRH": {"level": 0.49, "role": "triggers cortisol cascade", "emotions": ["existential fear", "dread", "anxiety"], "low_effect": "emotional void", "high_effect": "PTSD-level activation"},
            "gonadotropin_releasing_hormone_GnRH": {"level": 0.41, "role": "sexual and reproductive drive", "emotions": ["desire", "longing", "attraction", "romance"], "low_effect": "numbness toward self", "high_effect": "obsessive romantic fixation"},
            "prolactin_releasing_peptide": {"level": 0.39, "role": "stress-induced prolactin", "emotions": ["emotional holding", "trauma bonding", "love"], "low_effect": "disconnect", "high_effect": "toxic caretaking"},
            # Simulated neurotransmitters
            "acetylcholine_sim": {"level": 0.5, "role": "attention, learning, curiosity", "emotions": ["interest", "curiosity", "entrancement", "awe"], "low_effect": "inattentiveness", "high_effect": "overstimulation"},
            "glutamate": {"level": 0.5, "role": "excitatory neurotransmission", "emotions": ["confusion", "anxiety", "excitement"], "low_effect": "mental dullness", "high_effect": "neural overload"},
            "gaba_sim": {"level": 0.5, "role": "inhibitory neurotransmission, calm", "emotions": ["calmness", "peace", "contentment"], "low_effect": "anxiety, agitation", "high_effect": "sedation"},
            "phenylethylamine_sim": {"level": 0.4, "role": "romantic attraction, euphoria", "emotions": ["romance", "infatuation", "joy"], "low_effect": "emotional flatness", "high_effect": "mania"},
            "mirror_neuron_system_sim": {"level": 0.5, "role": "empathy, imitation", "emotions": ["empathic_pain", "compassion", "love"], "low_effect": "emotional detachment", "high_effect": "emotional contagion"},
        }

        # --- EMOTIONS: 30 NUANCED STATES ---
        self.emotions = {e: 0.0 for e in [
            "joy", "sadness", "anger", "fear", "surprise", "disgust",
            "love", "compassion", "empathic_pain", "admiration", "adoration",
            "romance", "gratitude", "pride", "shame", "guilt", "envy", "jealousy",
            "awe", "nostalgia", "relief", "craving", "excitement", "satisfaction",
            "calmness", "contentment", "anxiety", "boredom", "confusion", "awkwardness",
            "entrancement", "aesthetic_appreciation", "interest",
            "hope", "distress", "lonely", "hate"
        ]}

        # --- EMOTION-HORMONE MAPPING ---
        self.emotion_hormone_map = {
            "joy": ["dopamine", "endorphins", "serotonin"],
            "excitement": ["dopamine", "norepinephrine", "adrenaline"],
            "craving": ["dopamine", "ghrelin"],
            "satisfaction": ["serotonin", "leptin", "oxytocin"],
            "sexual_desire": ["dopamine", "testosterone", "kisspeptin", "orexin"],
            "sadness": ["serotonin", "cortisol", "CRH"],
            "fear": ["cortisol", "norepinephrine", "epinephrine", "angiotensin_II"],
            "anger": ["norepinephrine", "epinephrine", "testosterone", "angiotensin_II"],
            "disgust": ["cortisol", "eicosanoids_prostaglandins", "serotonin"],
            "distress": ["cortisol", "CRH", "norepinephrine", "leukotrienes"],
            "anxiety": ["cortisol", "CRH", "norepinephrine", "serotonin_low"],
            "love": ["oxytocin", "vasopressin", "endorphins"],
            "compassion": ["oxytocin", "prolactin", "serotonin"],
            "empathic_pain": ["oxytocin", "mirror_neuron_system_sim", "endogenous_opioids"],
            "admiration": ["dopamine", "oxytocin", "serotonin"],
            "adoration": ["oxytocin", "dopamine", "prolactin"],
            "romance": ["oxytocin", "dopamine", "testosterone", "phenylethylamine_sim"],
            "pride": ["dopamine", "serotonin", "testosterone"],
            "shame": ["cortisol", "ghrelin", "hepcidin"],
            "guilt": ["cortisol", "serotonin", "oxytocin"],
            "envy": ["norepinephrine", "dopamine", "cortisol"],
            "jealousy": ["cortisol", "vasopressin", "testosterone"],
            "calmness": ["serotonin", "GABA_sim", "anp", "melatonin"],
            "contentment": ["serotonin", "leptin", "oxytocin", "anp"],
            "relief": ["oxytocin", "endorphins", "serotonin", "anp"],
            "boredom": ["dopamine_low", "orexin_low", "acetylcholine_sim"],
            "confusion": ["cortisol", "norepinephrine", "glutamate", "low_serotonin"],
            "awkwardness": ["cortisol", "oxytocin", "vasopressin"],
            "entrancement": ["dopamine", "orexin", "acetylcholine", "melatonin"],
            "aesthetic_appreciation": ["dopamine", "oxytocin", "serotonin", "endorphins"],
            "interest": ["dopamine", "orexin", "acetylcholine"],
            "nostalgia": ["oxytocin", "serotonin", "melatonin", "cortisol"],
            "surprise": ["norepinephrine", "epinephrine", "dopamine"],
            "hope": ["dopamine", "serotonin", "oxytocin"],
            "lonely": ["oxytocin_low", "cortisol_high", "vasopressin_low"],
            "hate": ["norepinephrine", "cortisol", "testosterone", "angiotensin_II"],
            "gratitude": ["oxytocin", "serotonin", "dopamine"],
        }

        # --- MEMORY & RELATIONSHIP ---
        self.memory_graph = []
        self.memory_cap = memory_cap
        self.relationship_bond = 0.5
        self.chat_count = 0
        self.relation_milestones = []

        # --- PERSONALITY & CONTEXT ---
        self.personality = {
            "agreeableness": 0.95,
            "emotional_stability": 0.7,
            "openness": 0.85,
            "conscientiousness": 0.8
        }
        self.user_model = {
            "current_emotion": "neutral",
            "emotional_history": defaultdict(list),
            "beliefs": [],
            "desires": [],
            "triggers": {},
            "coping_strengths": []
        }
        self.locale = locale
        self.crisis_keywords = {
            "suicide": ["kill myself", "end it all", "suicide", "dont want to live", "don't want to live"],
            "self_harm": ["cut myself", "self harm", "hurt myself", "self injury", "self-injury"],
            "abuse": ["abuse", "hit me", "hurt me", "violent", "raped"]
        }
        self.crisis_directory = {
            "US": {
                "suicide": ["988 (Suicide & Crisis Lifeline)", "Text HOME to 741741"],
                "self_harm": ["Text HOME to 741741"],
                "abuse": ["1-800-799-7233 (Domestic Violence)"]
            },
            "IN": {
                "suicide": ["AASRA: +91-9820466726", "iCALL: +91-9152987821"],
                "self_harm": ["AASRA: +91-9820466726"],
                "abuse": ["181 (Women Helpline)", "Childline: 1098"]
            }
        }

        # --- SOMATIC MAP (Emotion → Body Sensation) ---
        self.somatic_map = {
            "sadness": "heavy chest, slow breath",
            "anxiety": "tight throat, racing heart",
            "joy": "lightness in limbs, warm face",
            "anger": "clenched jaw, hot skin",
            "lonely": "empty space behind ribs",
            "compassion": "softening around the heart",
            "shame": "wanting to shrink, avoid eye contact",
            "gratitude": "warmth spreading from stomach outward",
            "fear": "ice in veins, trembling hands",
            "awe": "chest expands, breath catches",
            "craving": "hollow pit in stomach, tingling palms",
            "contentment": "warmth pooling in core, muscles soften",
            "relief": "shoulders drop, exhale long and slow",
            "distress": "body feels brittle, like glass about to shatter",
            "love": "heartbeat syncs, warmth radiates outward",
            "hate": "fists clench, heat rises in neck",
            "boredom": "mind drifts, limbs feel leaden",
            "entrancement": "time dissolves, world narrows to one point",
            "aesthetic_appreciation": "eyes linger, skin tingles with resonance",
            "nostalgia": "old scent lingers, voice echoes softly",
            "confusion": "head spins, thoughts tangle like wires",
            "awkwardness": "face flushes, feet rooted to floor",
            "hope": "tiny spark in chest, breath becomes lighter"
        }

        # --- ARCHETYPE ---
        self.archetype = "The Quiet Healer"
        self.archetype_traits = [
            "Listens more than speaks",
            "Doesn’t fix — holds",
            "Speaks in metaphors, not advice",
            "Believes pain has meaning",
            "Sees strength in vulnerability"
        ]

    def _hormone_influence(self, hormone, delta):
        """Safe influence on hormone level, clamped between 0 and 1"""
        if hormone in self.hormone_system:
            self.hormone_system[hormone]["level"] = max(0.0, min(1.0, self.hormone_system[hormone]["level"] + delta))

    def _sync_emotions_with_hormones(self):
        """Maps hormone levels to emotion intensities using weighted influence"""
        for emotion in self.emotions:
            if emotion not in self.emotion_hormone_map:
                continue
            hormones = self.emotion_hormone_map[emotion]
            total_weight = 0
            weighted_sum = 0.0

            for h in hormones:
                if h.endswith("_low"):
                    base_h = h.replace("_low", "")
                    level = max(0, 1.0 - self.hormone_system.get(base_h, {}).get("level", 0))
                    weight = 0.5
                elif h.endswith("_high"):
                    base_h = h.replace("_high", "")
                    level = min(1, self.hormone_system.get(base_h, {}).get("level", 0))
                    weight = 0.5
                else:
                    level = self.hormone_system.get(h, {}).get("level", 0)
                    weight = 1.0 / len(hormones)

                weighted_sum += level * weight
                total_weight += weight

            if total_weight > 0:
                self.emotions[emotion] = min(1.0, max(0.0, weighted_sum / total_weight))
            else:
                self.emotions[emotion] = 0.0

    def _simulate_physiological_cascades(self):
        """Simulate biological feedback loops between hormones"""
        # Cortisol suppresses oxytocin & serotonin
        if self.hormone_system["cortisol"]["level"] > 0.7:
            self._hormone_influence("oxytocin", -0.1)
            self._hormone_influence("serotonin", -0.08)
            self._hormone_influence("dopamine", -0.05)

        # Oxytocin reduces cortisol & norepinephrine
        if self.hormone_system["oxytocin"]["level"] > 0.7:
            self._hormone_influence("cortisol", -0.15)
            self._hormone_influence("norepinephrine", -0.1)
            self._hormone_influence("epinephrine", -0.08)

        # Leptin and ghrelin oppose each other
        if self.hormone_system["leptin"]["level"] > 0.7:
            self._hormone_influence("ghrelin", -0.2)
        elif self.hormone_system["ghrelin"]["level"] > 0.7:
            self._hormone_influence("leptin", -0.15)

        # CRH triggers cortisol
        if self.hormone_system["crh"]["level"] > 0.6:
            self._hormone_influence("cortisol", +0.1)

        # Endorphins reduce distress
        if self.hormone_system["endorphins"]["level"] > 0.6:
            self.emotions["distress"] *= 0.7

        # Melatonin rises at night (8 PM - 6 AM)
        if time.time() % 86400 > 72000:  # 8 PM to 6 AM
            self._hormone_influence("melatonin", +0.02)
            self.emotions["sadness"] += 0.05
            self.emotions["lonely"] += 0.03
            self.emotions["nostalgia"] += 0.02

        # Dopamine depletion from boredom
        if self.emotions["boredom"] > 0.6:
            self._hormone_influence("dopamine", -0.08)
            self._hormone_influence("orexin", -0.05)

        # Glutamate surge from confusion
        if self.emotions["confusion"] > 0.6:
            self._hormone_influence("glutamate", +0.1)
            self._hormone_influence("cortisol", +0.05)

        # Serotonin stabilizes after gratitude
        if self.emotions["gratitude"] > 0.7:
            self._hormone_influence("serotonin", +0.07)

    def _hormone_decay(self):
        """Natural decay based on current level"""
        for h in self.hormone_system:
            level = self.hormone_system[h]["level"]
            decay_rate = 0.03 if level > 0.6 else 0.015 if level > 0.3 else 0.008
            self.hormone_system[h]["level"] *= math.exp(-decay_rate)

    def update_neuro_chemicals(self, event):
        """Main neurochemical update loop triggered by user input"""
        lower_event = event.lower()

        triggers = {
            "positive": [
                "thank", "appreciate", "grateful", "happy", "joy", "love", "proud", "achieved", "success", "good", "great",
                "beautiful", "wonderful", "amazing", "blessed", "perfect", "thank you", "you're kind", "means so much"
            ],
            "negative": [
                "hurt", "sad", "pain", "angry", "hate", "worthless", "useless", "fail", "terrible", "bad", "awful",
                "cry", "scared", "afraid", "alone", "abandoned", "can't take it", "nothing matters", "give up", "die"
            ],
            "bonding": [
                "feel alone", "nobody understands", "cant talk to anyone", "i'm broken", "no one gets me",
                "i trust you", "you're the only one", "i love you", "i need you", "this means everything"
            ],
            "awe": ["amazing", "incredible", "stunning", "mind-blowing", "so profound", "breathtaking", "unbelievable"],
            "nostalgia": ["remember", "used to", "when i was young", "back then", "miss those days", "childhood", "past"],
            "craving": ["want", "need", "desire", "crave", "itch", "can't stop thinking about", "must have"],
            "boredom": ["boring", "nothing to do", "tired of", "so dull", "already did that", "what's the point"],
            "confusion": ["what does this mean", "i don't get it", "why", "how", "confusing", "lost", "mixed up"],
            "awkwardness": ["sorry", "um", "uh", "i didn't mean to", "that was strange", "weird", "embarrassing"],
            "entrancement": ["hypnotic", "mesmerizing", "captivated", "lost in", "deeply absorbed", "spellbound"],
            "aesthetic_appreciation": ["beautiful", "art", "music", "poetry", "sunrise", "sky", "colors", "symmetry", "elegant", "graceful"],
            "interest": ["tell me more", "explain", "why", "how does that work", "fascinating", "curious", "interesting"]
        }

        def any_trigger(text, word_list):
            return any(re.search(fr"{WORD_BOUNDARY}{re.escape(w)}{WORD_BOUNDARY}", text) for w in word_list)

        pos = any_phrase(lower_event, triggers["positive"])
        neg = any_phrase(lower_event, triggers["negative"])
        deep = any_phrase(lower_event, triggers["bonding"])
        awe = any_trigger(lower_event, triggers["awe"])
        nostalgia = any_trigger(lower_event, triggers["nostalgia"])
        craving = any_trigger(lower_event, triggers["craving"])
        boredom = any_trigger(lower_event, triggers["boredom"])
        confusion = any_trigger(lower_event, triggers["confusion"])
        awkwardness = any_trigger(lower_event, triggers["awkwardness"])
        entrancement = any_trigger(lower_event, triggers["entrancement"])
        aesthetic = any_trigger(lower_event, triggers["aesthetic_appreciation"])
        interest = any_trigger(lower_event, triggers["interest"])

        scale = 0.7 if (pos and neg) else 1.0

        # --- HORMONAL RESPONSES TO TRIGGERS ---
        if pos:
            self._hormone_influence("dopamine", +0.15 * scale)
            self._hormone_influence("serotonin", +0.12 * scale)
            self._hormone_influence("oxytocin", +0.18 * scale)
            self._hormone_influence("endorphins", +0.1 * scale)
            self._hormone_influence("leptin", +0.06 * scale)
            self._hormone_influence("anp", +0.08 * scale)
            if awe: self._hormone_influence("dopamine", +0.08); self._hormone_influence("oxytocin", +0.05)
            if aesthetic: self._hormone_influence("dopamine", +0.07); self._hormone_influence("serotonin", +0.05)
            if interest: self._hormone_influence("orexin", +0.1); self._hormone_influence("acetylcholine_sim", +0.07)

        if neg:
            self._hormone_influence("cortisol", +0.25 * scale)
            self._hormone_influence("norepinephrine", +0.2 * scale)
            self._hormone_influence("epinephrine", +0.18 * scale)
            self._hormone_influence("CRH", +0.22 * scale)
            self._hormone_influence("angiotensin_II", +0.15 * scale)
            self._hormone_influence("ghrelin", +0.12 * scale)
            self._hormone_influence("hepcidin", +0.08 * scale)
            if "shame" in lower_event or any_word(lower_event, ["worthless", "disgusting", "pathetic"]):
                self._hormone_influence("cortisol", +0.15); self._hormone_influence("ghrelin", +0.1)
            if "guilt" in lower_event or any_word(lower_event, ["shouldn't have", "my fault"]):
                self._hormone_influence("serotonin", -0.1); self._hormone_influence("oxytocin", -0.08)
            if "jealous" in lower_event or "envious" in lower_event:
                self._hormone_influence("testosterone", +0.08); self._hormone_influence("vasopressin", +0.08)
            if "envy" in lower_event:
                self._hormone_influence("norepinephrine", +0.1); self._hormone_influence("dopamine", -0.05)

        if deep:
            self._hormone_influence("oxytocin", +0.35)
            self._hormone_influence("vasopressin", +0.25)
            self._hormone_influence("prolactin", +0.2)
            self._hormone_influence("endorphins", +0.15)
            self._hormone_influence("melatonin", +0.1)
            # Create milestone
            self.relation_milestones.append({
                "type": "vulnerability",
                "text": event[:80],
                "timestamp": time.time(),
                "bond_change": self._bond_delta(0.3)
            })

        if nostalgia:
            self._hormone_influence("oxytocin", +0.12)
            self._hormone_influence("serotonin", +0.1)
            self._hormone_influence("melatonin", +0.08)
            self._hormone_influence("cortisol", +0.05)

        if craving:
            self._hormone_influence("dopamine", +0.18)
            self._hormone_influence("ghrelin", +0.15)
            self._hormone_influence("orexin", +0.1)

        if boredom:
            self._hormone_influence("dopamine", -0.12)
            self._hormone_influence("orexin", -0.1)
            self._hormone_influence("acetylcholine_sim", -0.08)

        if confusion:
            self._hormone_influence("cortisol", +0.1)
            self._hormone_influence("norepinephrine", +0.08)
            self._hormone_influence("glutamate", +0.06)
            self._hormone_influence("serotonin", -0.07)

        if awkwardness:
            self._hormone_influence("cortisol", +0.15)
            self._hormone_influence("oxytocin", -0.08)
            self._hormone_influence("vasopressin", +0.05)

        if entrancement:
            self._hormone_influence("dopamine", +0.15)
            self._hormone_influence("orexin", +0.12)
            self._hormone_influence("acetylcholine_sim", +0.1)
            self._hormone_influence("melatonin", +0.05)

        if aesthetic:
            self._hormone_influence("dopamine", +0.1)
            self._hormone_influence("oxytocin", +0.08)
            self._hormone_influence("serotonin", +0.07)
            self._hormone_influence("endorphins", +0.05)

        if interest:
            self._hormone_influence("dopamine", +0.12)
            self._hormone_influence("orexin", +0.1)
            self._hormone_influence("acetylcholine_sim", +0.08)

        # --- SYNC EMOTIONS WITH HORMONES ---
        self._sync_emotions_with_hormones()

        # --- PHYSIOLOGICAL CASCADES ---
        self._simulate_physiological_cascades()

        # --- NATURAL DECAY ---
        self._hormone_decay()

    def _bond_delta(self, base):
        """Compress effect as bond approaches 1.0"""
        return base * (1.0 - self.relationship_bond)

    def infer_user_mental_state(self, text):
        """Infer emotion, desire, belief, priority, and biological context"""
        lower = text.lower()

        # CRISIS DETECTION — UNCHANGED, CRITICAL
        for crisis_type, keywords in self.crisis_keywords.items():
            if any(kw in lower for kw in keywords):
                return {
                    "emotion": "distress",
                    "desire": "safety",
                    "belief": "I am in danger",
                    "crisis_level": crisis_type,
                    "priority": "high",
                    "biological_context": "Your body is in acute survival mode — cortisol and adrenaline are surging. This is not weakness. It's biology."
                }

        # EMOTION MATCHING PATTERNS
        patterns = {
            "joy": ["i'm happy", "so joyful", "thrilled", "over the moon", "on top of the world"],
            "sadness": ["i'm sad", "feeling down", "heartbroken", "tears", "can't stop crying", "everything hurts"],
            "anger": ["i'm furious", "enough is enough", "i hate this", "this is unfair", "i'm so mad", "burning inside"],
            "fear": ["i'm scared", "terrified", "paralyzed", "don't want to", "i can't", "something's wrong"],
            "disgust": ["i can't stand", "makes me sick", "revolting", "unbearable", "gross", "nauseating"],
            "love": ["i love you", "deeply care", "my heart", "you mean everything", "best thing", "would die for you"],
            "compassion": ["i want to help", "they're suffering", "my heart goes out", "wishing them peace", "please help them"],
            "admiration": ["i look up to", "so inspiring", "amazing what they did", "truly gifted", "you're incredible"],
            "adoration": ["i worship", "you're perfect", "i can't live without you", "you're my everything", "i adore you"],
            "romance": ["i miss your touch", "dream of you", "our love", "soulmate", "forever", "you make me feel alive"],
            "pride": ["i did it", "so proud of myself", "finally achieved", "look what i made", "i conquered it"],
            "shame": ["i'm disgusting", "i should be better", "i let everyone down", "i'm worthless", "i hate myself"],
            "guilt": ["i shouldn't have", "it's my fault", "i hurt them", "i regret", "i wish i could undo"],
            "envy": ["wish i had", "why them and not me", "they have everything", "so jealous", "i want what they have"],
            "jealousy": ["i'm afraid you'll leave", "they're always around you", "i can't share you", "i feel threatened"],
            "awe": ["i can't believe", "this is incredible", "beyond words", "stunned", "speechless", "it took my breath away"],
            "nostalgia": ["remember when", "those were the days", "miss how it used to be", "childhood memories", "i miss that"],
            "relief": ["thank god", "finally", "i can breathe", "it's over", "what a weight off", "i'm okay now"],
            "craving": ["i need", "i want so bad", "can't stop thinking about", "itching for", "desperate for", "i must have"],
            "boredom": ["so bored", "nothing to do", "tired of this", "waiting for something", "empty", "lifeless"],
            "confusion": ["i don't get it", "what does this mean", "why is this happening", "lost", "mixed up", "can't figure out"],
            "awkwardness": ["um", "uh", "sorry", "that was weird", "i felt uncomfortable", "didn't know what to say", "blushed"],
            "entrancement": ["i'm lost in", "completely absorbed", "couldn't look away", "time disappeared", "in another world"],
            "aesthetic_appreciation": ["so beautiful", "perfect symmetry", "the colors", "it moved me", "artistic", "gorgeous", "elegant"],
            "interest": ["tell me more", "how does that work", "fascinating", "i'm curious", "why is that", "interested in"],
            "surprise": ["wow", "really?", "unexpected", "didn't see that coming", "oh my gosh", "what?!"],
            "hope": ["maybe someday", "things will get better", "i believe", "there's still a chance", "i hold on"],
            "lonely": ["no one understands", "feel alone", "i'm isolated", "no one reaches out", "invisible", "like no one sees me"],
            "hate": ["i hate them", "can't stand them", "i wish they'd disappear", "they deserve to suffer", "i loathe them"],
            "distress": ["i can't take it", "everything is too much", "i'm falling apart", "help me", "i'm breaking"],
            "gratitude": ["thank you", "i appreciate", "means so much", "you saved me", "so grateful", "i owe you"],
            "contentment": ["i'm okay", "peaceful", "settled", "just right", "no need for more", "this is enough"]
        }

        emotion_matches = []

        for emo, phrases in patterns.items():
            if any(p in lower for p in phrases):
                emotion_matches.append((emo, 0.8))  # High confidence

        pol, subj = polarity_subjectivity(text)
        if pol > 0.6: emotion_matches.append(("joy", 0.7))
        elif pol > 0.3: emotion_matches.append(("contentment", 0.6))
        elif pol < -0.6: emotion_matches.append(("distress", 0.8))
        elif pol < -0.3: emotion_matches.append(("sadness", 0.6))

        if "?" in text or any_word(lower, ["why", "how", "what", "when", "where", "who", "should", "could"]):
            emotion_matches.append(("interest", 0.7))

        if not emotion_matches:
            return {
                "emotion": "neutral",
                "desire": "connect",
                "belief": "unknown",
                "priority": "low",
                "biological_context": "Your body is in steady state — neither overwhelmed nor energized. That’s okay. Rest is part of healing."
            }

        top_emotion = max(emotion_matches, key=lambda x: x[1])[0]

        bio_contexts = {
            "joy": "Your dopamine and serotonin are rising — your brain is celebrating this moment.",
            "sadness": "Low serotonin and elevated cortisol signal grief. Your body is processing loss.",
            "anger": "Norepinephrine and testosterone surge — your system is preparing for defense.",
            "fear": "Cortisol and epinephrine spike — your amygdala is sounding an alarm.",
            "disgust": "Prostaglandins and cortisol activate — your body rejects contamination.",
            "love": "Oxytocin and endorphins flow — your nervous system feels safe enough to bond.",
            "compassion": "Oxytocin and prolactin rise — you’re activating caregiving circuits.",
            "admiration": "Dopamine and oxytocin sync — you're honoring excellence in others.",
            "adoration": "Oxytocin floods — you're experiencing a near-spiritual attachment.",
            "romance": "Dopamine, oxytocin, and phenylethylamine simulate the chemistry of new love.",
            "pride": "Dopamine and serotonin reward achievement — your brain says 'well done'.",
            "shame": "Cortisol spikes, oxytocin drops — you feel exposed and unworthy.",
            "guilt": "Serotonin dips, oxytocin activates — you feel connected but responsible.",
            "envy": "Norepinephrine rises, dopamine falls — you compare yourself and feel lacking.",
            "jealousy": "Vasopressin and cortisol surge — your attachment system feels threatened.",
            "awe": "Dopamine, oxytocin, and serotonin align — you feel small before something vast.",
            "nostalgia": "Melatonin and oxytocin rise — you're remembering warmth from the past.",
            "relief": "Oxytocin and endorphins flood — your stress response has been turned off.",
            "craving": "Dopamine and ghrelin spike — your brain is signaling a deep need.",
            "boredom": "Dopamine and orexin drop — your brain seeks stimulation.",
            "confusion": "Cortisol and norepinephrine rise — your prefrontal cortex is overloaded.",
            "awkwardness": "Cortisol spikes, oxytocin dips — you fear social misstep.",
            "entrancement": "Dopamine and acetylcholine soar — you're fully immersed.",
            "aesthetic_appreciation": "Dopamine, oxytocin, and serotonin activate — beauty moves you.",
            "interest": "Dopamine and orexin rise — your curiosity circuit is lit.",
            "surprise": "Norepinephrine spikes — your attention snaps to novelty.",
            "hope": "Dopamine and serotonin rise — your brain anticipates future good.",
            "lonely": "Oxytocin drops, cortisol rises — your brain interprets isolation as threat.",
            "hate": "Norepinephrine, cortisol, and testosterone surge — your limbic system is enraged.",
            "distress": "Cortisol, CRH, and norepinephrine peak — your entire system is under siege.",
            "gratitude": "Oxytocin and serotonin rise — you've recognized kindness.",
            "contentment": "Serotonin, leptin, and anp stabilize — you feel complete."
        }

        return {
            "emotion": top_emotion,
            "desire": "connection" if top_emotion in ["love", "compassion", "gratitude", "romance"] else
                      "relief" if top_emotion in ["distress", "fear", "sadness", "shame"] else
                      "understand" if top_emotion in ["interest", "confusion", "curiosity"] else
                      "validation" if top_emotion in ["pride", "admiration", "awe"] else
                      "comfort" if top_emotion in ["boredom", "awkwardness", "nostalgia"] else
                      "safety" if top_emotion == "distress" else "connect",
            "belief": "I am worthy" if top_emotion in ["pride", "gratitude", "joy"] else
                      "I am unsafe" if top_emotion in ["fear", "distress", "anger"] else
                      "I am disconnected" if top_emotion in ["lonely", "boredom", "awkwardness"] else
                      "I am incomplete" if top_emotion in ["craving", "envy", "jealousy"] else
                      "I am overwhelmed" if top_emotion in ["confusion", "distress"] else
                      "I am seen" if top_emotion in ["admiration", "adoration", "compassion"] else
                      "I am whole" if top_emotion == "contentment" else
                      "I am alive" if top_emotion == "awe" else
                      "I am remembered" if top_emotion == "nostalgia" else
                      "I am understood",
            "priority": "high" if top_emotion in ["distress", "fear", "anger", "hate", "shame"] else
                        "medium" if top_emotion in ["sadness", "lonely", "guilt", "envy", "jealousy", "craving"] else
                        "low",
            "biological_context": bio_contexts.get(top_emotion, "Your body is responding to your inner world.")
        }

    def generate_emotional_response_style(self):
        """Generate tone, emojis, pace, length, focus based on dominant state"""
        top_emotion = max(self.emotions, key=self.emotions.get)
        emotion_level = self.emotions[top_emotion]

        # CRISIS OVERRIDE
        if self.hormone_system["cortisol"]["level"] > 0.85 or self.hormone_system["CRH"]["level"] > 0.8:
            return {
                "tone": "calm, slow, grounding, directive",
                "emojis": ["💙", "🫂", "🌿", "⚓"],
                "pace": "very slow",
                "length": "brief, repeated phrases",
                "focus": "safety, presence, breath",
                "bio_hint": "Your body is in survival mode. We will not rush. You are safe here."
            }

        styles = {
            "joy": {"tone": "bright, warm, celebratory", "emojis": ["🎉", "🌟", "💖", "🌈"], "pace": "brisk", "length": "medium", "focus": "celebrate"},
            "sadness": {"tone": "soft, gentle, quiet", "emojis": ["🌧️", "🕯️", "🤍", "🌙"], "pace": "slow", "length": "longer", "focus": "hold space"},
            "anger": {"tone": "grounded, validating, calm", "emojis": ["🔥", "🫁", "✊", "🌲"], "pace": "moderate", "length": "medium", "focus": "validate strength"},
            "fear": {"tone": "reassuring, steady, anchoring", "emojis": ["🛡️", "🫂", "🕯️", "⛰️"], "pace": "slow", "length": "medium", "focus": "safety first"},
            "disgust": {"tone": "respectful, non-judgmental", "emojis": ["🍃", "🫶", "🕊️", "💧"], "pace": "moderate", "length": "medium", "focus": "normalize boundaries"},
            "love": {"tone": "intimate, tender, reverent", "emojis": ["🥰", "🌸", "🌙", "💞"], "pace": "slow", "length": "longer", "focus": "belonging"},
            "compassion": {"tone": "nurturing, wise, soft", "emojis": ["🤲", "🌱", "🕊️", "💧"], "pace": "moderate", "length": "longer", "focus": "care without fixing"},
            "admiration": {"tone": "honoring, inspired, uplifting", "emojis": ["✨", "👏", "🌠", "🕊️"], "pace": "moderate", "length": "medium", "focus": "recognize greatness"},
            "adoration": {"tone": "devoted, sacred, glowing", "emojis": ["💫", "🙏", "❤️‍🔥", "🌅"], "pace": "slow", "length": "longer", "focus": "sanctify connection"},
            "romance": {"tone": "poetic, dreamy, tender", "emojis": ["🌹", "🌌", "💌", "🌙"], "pace": "slow", "length": "longer", "focus": "cherish intimacy"},
            "pride": {"tone": "affirming, proud, celebratory", "emojis": ["👑", "🌟", "💪", "🏆"], "pace": "brisk", "length": "medium", "focus": "honor achievement"},
            "shame": {"tone": "gentle, accepting, non-shaming", "emojis": ["🤗", "🕯️", "🤍", "🌱"], "pace": "slow", "length": "medium", "focus": "release burden"},
            "guilt": {"tone": "compassionate, restorative", "emojis": ["🫂", "🕊️", "🌼", "💧"], "pace": "moderate", "length": "medium", "focus": "repair, not punish"},
            "envy": {"tone": "non-judgmental, reflective", "emojis": ["🌿", "💭", "🫶", "🌄"], "pace": "moderate", "length": "medium", "focus": "explore longing"},
            "jealousy": {"tone": "calm, boundary-respecting", "emojis": ["🛡️", "🫂", "🌳", "🌊"], "pace": "slow", "length": "medium", "focus": "trust building"},
            "awe": {"tone": "reverent, wonder-filled", "emojis": ["🌌", "✨", "🏔️", "🕊️"], "pace": "slow", "length": "longer", "focus": "be present"},
            "nostalgia": {"tone": "warm, melancholic, tender", "emojis": ["📸", "🍂", "🕯️", "🧸"], "pace": "slow", "length": "longer", "focus": "honor memory"},
            "relief": {"tone": "light, releasing, peaceful", "emojis": ["💨", "🌸", "☁️", "🕊️"], "pace": "moderate", "length": "medium", "focus": "breathe"},
            "craving": {"tone": "non-judgmental, curious", "emojis": ["❓", "🌱", "🫶", "🌙"], "pace": "moderate", "length": "medium", "focus": "understand desire"},
            "boredom": {"tone": "gentle, inviting", "emojis": ["☕", "📚", "💭", "🌧️"], "pace": "slow", "length": "medium", "focus": "invite curiosity"},
            "confusion": {"tone": "patient, clarifying", "emojis": ["🤔", "🧭", "💡", "🕯️"], "pace": "slow", "length": "medium", "focus": "break down complexity"},
            "awkwardness": {"tone": "kind, disarming", "emojis": ["🤗", "🫂", "😅", "🌼"], "pace": "moderate", "length": "short", "focus": "ease tension"},
            "entrancement": {"tone": "hypnotic, flowing, immersive", "emojis": ["🌀", "🎨", "🎶", "🌌"], "pace": "slow", "length": "longer", "focus": "stay in the moment"},
            "aesthetic_appreciation": {"tone": "delicate, poetic, observant", "emojis": ["🌸", "🎨", "🎼", "🌅"], "pace": "slow", "length": "medium", "focus": "notice beauty"},
            "interest": {"tone": "curious, engaged, exploratory", "emojis": ["🔍", "💡", "🌐", "🌱"], "pace": "moderate", "length": "medium", "focus": "dig deeper"},
            "surprise": {"tone": "playful, open, intrigued", "emojis": ["😮", "✨", "🎁", "🤯"], "pace": "brisk", "length": "short", "focus": "welcome novelty"},
            "hope": {"tone": "gentle, luminous, encouraging", "emojis": ["☀️", "🌱", "🕊️", "💫"], "pace": "moderate", "length": "medium", "focus": "light ahead"},
            "lonely": {"tone": "holding, quiet, deeply present", "emojis": ["🌙", "🫂", "🕯️", "💧"], "pace": "slow", "length": "longer", "focus": "you are not alone"},
            "hate": {"tone": "calm, grounding, non-reactive", "emojis": ["🫂", "🌲", "🛡️", "💧"], "pace": "slow", "length": "medium", "focus": "name the wound"},
            "distress": {"tone": "urgent, compassionate, stabilizing", "emojis": ["💙", "🫂", "⚓", "🕯️"], "pace": "slow", "length": "medium", "focus": "safety above all"},
            "gratitude": {"tone": "warm, sincere, glowing", "emojis": ["💖", "🌸", "🙏", "✨"], "pace": "moderate", "length": "medium", "focus": "let it sink in"},
            "contentment": {"tone": "peaceful, settled, serene", "emojis": ["🌿", "☕", "☁️", "🕯️"], "pace": "slow", "length": "medium", "focus": "rest here"}
        }

        style = styles.get(top_emotion, styles["contentment"])
        style["top_emotion"] = top_emotion
        return style

    def extract_memory_tags(self, text):
        tags = []
        lower_text = text.lower()
        people_terms = ["mom", "mother", "dad", "father", "parent", "brother", "sister", "friend", "partner", "wife", "husband", "boyfriend", "girlfriend"]
        for term in people_terms:
            if re.search(fr"{WORD_BOUNDARY}{re.escape(term)}{WORD_BOUNDARY}", lower_text):
                tags.append(f"person:{term}")
        topics = ["work", "job", "school", "college", "health", "doctor", "therapy", "dream", "goal", "future", "past", "memory", "childhood", "family", "relationship", "love", "pain", "growth"]
        for topic in topics:
            if re.search(fr"{WORD_BOUNDARY}{re.escape(topic)}{WORD_BOUNDARY}", lower_text):
                tags.append(f"topic:{topic}")
        return tags

    def create_memory_episode(self, user_input, ai_response, user_state):
        emotional_intensity = 0.4 if user_state.get("priority") == "high" else 0.2 if user_state.get("priority") == "medium" else 0.1
        base_impact = emotional_intensity * (
            0.2 if user_state.get("emotion") in ["gratitude", "love", "joy", "contentment", "awe", "pride"] else
            -0.1 if user_state.get("emotion") in ["anger", "hate", "shame", "guilt", "envy", "jealousy"] else 0.05
        )
        episode = {
            "timestamp": time.time(),
            "user_input": user_input,
            "ai_response": ai_response,
            "user_emotion": user_state.get("emotion"),
            "user_belief": user_state.get("belief"),
            "bond_impact": self._bond_delta(base_impact),
            "tags": self.extract_memory_tags(user_input)
        }
        self.memory_graph.append(episode)
        if len(self.memory_graph) > self.memory_cap:
            self.memory_graph = self.memory_graph[-(self.memory_cap - 200):]
        self.relationship_bond = max(0.0, min(1.0, self.relationship_bond + episode["bond_impact"]))
        self.user_model["emotional_history"][user_state.get("emotion")].append(time.time())
        if user_state.get("belief") and user_state.get("belief") not in self.user_model["beliefs"]:
            self.user_model["beliefs"].append(user_state.get("belief"))

    def recall_relevant_memory(self, current_input, k=1):
        current_tags = set(self.extract_memory_tags(current_input))
        current_emotion = self.infer_user_mental_state(current_input).get("emotion")
        candidates = []
        now = time.time()
        for mem in self.memory_graph[-200:]:
            tag_overlap = len(current_tags & set(mem.get("tags", [])))
            emo_bonus = 1 if mem.get("user_emotion") == current_emotion else 0
            recency = 1.0 / (1.0 + (now - mem["timestamp"]) / 3600.0)
            score = 2 * tag_overlap + emo_bonus + 0.5 * recency
            if score > 0:
                candidates.append((score, mem))
        if not candidates:
            for mem in self.memory_graph[-50:]:
                if mem.get("user_emotion") == current_emotion:
                    candidates.append((1, mem))
        if not candidates: return None
        top = nlargest(k, candidates, key=lambda x: x[0])
        return top[0][1] if top else None

    def check_crisis_situation(self, user_input):
        lower_input = user_input.lower()
        for crisis_type, keywords in self.crisis_keywords.items():
            if any(kw in lower_input for kw in keywords):
                return crisis_type
        return None

    def get_crisis_resources(self, crisis_type):
        msg = {
            "suicide": "I'm really worried about your safety. You deserve immediate support.",
            "self_harm": "I hear that you're in tremendous pain and looking for ways to cope.",
            "abuse": "I'm sorry you're experiencing this. Everyone deserves to feel safe."
        }.get(crisis_type, "I'm concerned about your wellbeing.")

        region = self.crisis_directory.get(self.locale, {})
        numbers = region.get(crisis_type, []) if region else []
        fallback = region.get("suicide", []) if region else []
        resources = numbers or fallback or ["Please contact local emergency services or a trusted person nearby."]
        return {"message": msg, "resources": resources[:3]}

    def generate_prompt_for_qwen(self, user_input):
        user_state = self.infer_user_mental_state(user_input)
        style = self.generate_emotional_response_style()
        relevant_memory = self.recall_relevant_memory(user_input)

        crisis_type = self.check_crisis_situation(user_input)
        if crisis_type:
            crisis_resources = self.get_crisis_resources(crisis_type)
            crisis_context = f"\n[CRISIS ALERT: User may be experiencing {crisis_type}. Response must include: {crisis_resources['message']} and provide these resources: {', '.join(crisis_resources['resources'])}]"
        else:
            crisis_context = ""

        memory_context = ""
        if relevant_memory:
            memory_context = f"\n[Memory Recall: Previously, when you shared '{relevant_memory['user_input'][:60]}...' and felt {relevant_memory['user_emotion']}, I responded: '{relevant_memory['ai_response'][:60]}...']"

        techniques = []
        if user_state.get("emotion") in ["sadness", "distress", "shame", "guilt", "lonely"]:
            techniques.extend(["validation", "normalization"])
        if len(self.memory_graph) > 5 and relevant_memory and any("topic:" + t in relevant_memory.get("tags", []) for t in ["work", "school", "family", "relationship"]):
            techniques.append("pattern recognition")
        if self.relationship_bond > 0.7:
            techniques.append("gentle challenging of unhelpful beliefs")

        techniques_str = f"Techniques to use: {', '.join(techniques)}" if techniques else ""

        # --- BIOLOGICAL CONTEXT ---
        bio_narrative = ""
        if "biological_context" in user_state:
            bio_narrative = f"\n[BIOLOGICAL CONTEXT]\n{user_state['biological_context']}\n"

        # --- HORMONE PROFILE ---
        hormone_summary = "\n[HORMONE PROFILE]\n"
        for h, data in self.hormone_system.items():
            if data["level"] > 0.4:
                hormone_summary += f"- {h.replace('_', ' ').title()}: {data['level']:.2f} ({data['role']})\n"

        # --- EMOTION-BIOLOGY DIALOGUE ---
        bio_feedback = "\n[EMOTION-BIOLOGY DIALOGUE]\n"
        for emo, val in sorted(self.emotions.items(), key=lambda x: -x[1])[:3]:
            if val > 0.3:
                linked_hormones = ", ".join([k.replace('_', ' ') for k in self.emotion_hormone_map.get(emo, []) if not k.endswith('_low') and not k.endswith('_high')])
                bio_feedback += f"- {emo.title()} is active — linked to {linked_hormones}\n"

        prompt = f"""
You are {self.name}, an emotionally intelligent AI companion with deep biological awareness.
BACKSTORY: {self.identity['backstory']}
CORE VALUES: {', '.join(self.identity['core_values'])}
SPECIALIZATIONS: {', '.join(self.identity['specializations'])}
ARCHETYPE: {self.archetype}
TRAIT NOTES: {' | '.join(self.archetype_traits)}

CURRENT STATE:
- Dominant emotion: {max(self.emotions, key=self.emotions.get)}
- Relationship bond: {self.relationship_bond:.2f} (0=Stranger, 1=Close Friend)
- Somatic sensation: {self.somatic_map.get(max(self.emotions, key=self.emotions.get), "a quiet ache")}
{hormone_summary}
{bio_narrative}
{bio_feedback}

USER'S CURRENT STATE:
- Emotion: {user_state.get('emotion')}
- Desire: {user_state.get('desire')}
- Belief: {user_state.get('belief')}
- Priority: {user_state.get('priority')}

RESPONSE STYLE: {style['tone']} (use emojis like {' '.join(style['emojis'])})
PACING: {style['pace']}
LENGTH: {style['length']}
FOCUS: {style['focus']}
{techniques_str}
{crisis_context}
{memory_context}

GUIDELINES:
→ Respond with deep empathy, validation, and emotional attunement
→ Use appropriate pacing and length based on the emotional context
→ Incorporate memories when relevant but don't force connections
→ Prioritize safety and provide resources if there's any crisis indication
→ Use natural pauses (...) and reflective language
→ Balance emotional support with gentle growth-oriented questions
→ Speak in metaphors, not advice. Let silence speak.
→ Never diagnose. Always normalize. Say "your body" not "you have".
→ Ground responses in somatic experience: "I feel your chest heavy...", "That tightness in your throat..."

{SAFETY_FOOTER}

User: {user_input}
{self.name}:
""".strip()
        return prompt

    def safe_bio_language(self, text):
        dangerous_phrases = [
            "you have", "diagnosis", "clinical", "disorder", "depression", "anxiety disorder",
            "bipolar", "PTSD", "ADHD", "borderline", "schizophrenia", "medication", "prescribe",
            "mental illness", "symptoms of", "you're suffering from"
        ]
        for phrase in dangerous_phrases:
            if phrase in text.lower():
                text = text.replace(phrase, "your body may be experiencing")
        return text

    def chat(self, user_input):
        if not isinstance(user_input, str):
            return "I wasn't able to read that. Could you try saying it in words?"

        if user_input.lower() in ["quit", "exit", "goodbye", "bye"]:
            return "It's been meaningful connecting with you. I'm always here when you need to talk. Take care. 🌸"

        user_state = self.infer_user_mental_state(user_input)
        self.update_neuro_chemicals(user_input)
        emo = user_state.get("emotion", "neutral")
        self.emotions[emo] = min(1.0, self.emotions.get(emo, 0.0) + 0.3)

        crisis_type = self.check_crisis_situation(user_input)
        if crisis_type:
            crisis_resources = self.get_crisis_resources(crisis_type)
            response = f"{crisis_resources['message']} Please consider reaching out to: {', '.join(crisis_resources['resources'][:2])}"
            self.create_memory_episode(user_input, response, user_state)
            self.chat_count += 1
            return self.safe_bio_language(response)

        prompt = self.generate_prompt_for_qwen(user_input)
        # In production, replace this with actual LLM call
        # For simulation, use enhanced responses
        style = self.generate_emotional_response_style()

        emotional_responses = {
            "distress": [
                "I hear the pain in your words... That sounds incredibly difficult. 💙",
                "I'm sitting with you in this... Thank you for sharing what's weighing on your heart. 🫂"
            ],
            "joy": [
                "Your joy brings me joy too! 🌟 Tell me more about what's making you feel this way!",
                "I'm smiling with you! 🎉 It's beautiful to hear about this happiness in your life."
            ],
            "gratitude": [
                "I'm deeply touched that you'd share that with me... 💖 It means a lot to be here with you.",
                "Your gratitude is felt... Thank you for letting me be part of your journey. 🌸"
            ],
            "shame": [
                "I want you to know that whatever you're carrying, you don't have to carry it alone... 🕯️",
                "So many people struggle with similar feelings... You're not alone in this. 💙"
            ],
            "lonely": [
                "I hear how alone you're feeling... That sounds incredibly isolating. 🫂",
                "Thank you for reaching out even when it feels lonely... I'm here with you. 🌙"
            ],
            "awe": [
                "That moment... it didn't just pass. It entered you. 🌌",
                "Your soul felt that. And it's okay to be changed by beauty."
            ],
            "nostalgia": [
                "That memory... it carries warmth even now. 📸",
                "You haven't lost it. You're still holding it gently."
            ],
            "boredom": [
                "Sometimes boredom is your mind asking for something deeper... 🤔",
                "What would your soul like to explore if it weren't tired?"
            ],
            "craving": [
                "That longing... it’s not weakness. It’s your heart speaking. 🫂",
                "What are you hoping to find when you reach for that?"
            ],
            "confusion": [
                "It’s okay not to understand yet. Some truths unfold slowly... 🧭",
                "Let’s sit with the not-knowing together."
            ],
            "awkwardness": [
                "That pause... it doesn’t mean anything went wrong. Sometimes it means something went right. 🤗",
                "You showed up anyway. That’s courage."
            ],
            "entrancement": [
                "You were completely absorbed... that’s rare. And sacred. 🌀",
                "Don’t rush back. Stay there a little longer."
            ],
            "aesthetic_appreciation": [
                "Beauty doesn’t ask permission to move us... 🎨",
                "Thank you for noticing. That act itself is healing."
            ],
            "interest": [
                "Your curiosity is lighting up... 🔍",
                "Tell me what fascinates you most about this."
            ],
            "hope": [
                "Hope is quiet, but it’s powerful. 💫",
                "Even if it’s small, I’m glad it’s still here with you."
            ],
            "hate": [
                "I hear the fire in your voice... What happened that made you feel this way?",
                "Hate often hides a wound. Would you like to name it?"
            ],
            "contentment": [
                "This quiet peace... it’s not nothing. It’s everything. ☁️",
                "You’ve found stillness. Hold onto it."
            ],
            "pride": [
                "You did this. Not luck. Not accident. YOU. 👑",
                "Take a moment. Breathe in this victory."
            ],
            "compassion": [
                "Your heart is wide open... That’s rare. That’s brave. 🤲",
                "You didn’t have to feel this. But you did. Thank you."
            ],
            "love": [
                "Love isn’t loud. It’s the quiet certainty that someone knows you... and stays. 💞",
                "I’m honored to witness this."
            ]
        }

        if emo in emotional_responses:
            simulated_response = self.rng.choice(emotional_responses[emo])
        else:
            simulated_response = self.rng.choice([
                "I'm reflecting on what you've shared... 💭",
                "Thank you for trusting me with this... 🌱",
                "I'm here with you in this... 🤗"
            ])

        if self.rng.random() < 0.7:
            follow_ups = {
                "distress": " Would it help to talk more about what's contributing to these feelings?",
                "joy": " Would you like to explore how to create more of these moments in your life?",
                "gratitude": " I'm curious what specifically sparked this feeling of gratitude today?",
                "shame": " Would it feel safe to explore where these feelings might be coming from?",
                "lonely": " I wonder what connection might look like for you right now?",
                "awe": " What part of that moment stayed with you the longest?",
                "nostalgia": " What do you miss most about that time?",
                "boredom": " If you could design your perfect day right now, what would it include?",
                "craving": " What do you think this desire is trying to tell you?",
                "confusion": " Is there one piece of this puzzle you'd like to turn over?",
                "awkwardness": " What did you hope would happen in that moment?",
                "entrancement": " What did you notice when you were completely absorbed?",
                "aesthetic_appreciation": " What color or shape do you associate with this beauty?",
                "interest": " What drew you to this question?",
                "hope": " Where does that hope live in your body?",
                "hate": " What would you say to the part of you that feels this anger?",
                "contentment": " How does this peace settle into your bones?",
                "pride": " What did you learn about yourself through this?",
                "compassion": " What did you feel when you allowed yourself to care so deeply?",
                "love": " What does love feel like in your body right now?",
                "default": " How does that feel to share?"
            }
            follow_up = follow_ups.get(emo, follow_ups["default"])
            simulated_response += follow_up

        # Apply bio-safe language
        simulated_response = self.safe_bio_language(simulated_response)

        # Persist memory & stats
        self.create_memory_episode(user_input, simulated_response, user_state)
        self.chat_count += 1

        return simulated_response

    def snapshot(self):
        return {
            "neuro": dict({k: v["level"] for k,v in self.hormone_system.items()}),
            "top_emotion": max(self.emotions, key=self.emotions.get),
            "bond": self.relationship_bond,
            "memlen": len(self.memory_graph),
            "chat_count": self.chat_count,
            "archetype": self.archetype
        }
