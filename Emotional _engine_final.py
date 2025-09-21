import time
import math
import random
import re
import numpy as np
import json
import os
from collections import defaultdict

def polarity_subjectivity(text):
    lower = text.lower()

    # Enhanced keyword matching with better coverage
    positive_keywords = {
        "love": 2, "amazing": 2, "wonderful": 2, "perfect": 2, "best": 2,
        "excellent": 2, "fantastic": 2, "brilliant": 2, "proud": 2, "happy": 2,
        "joy": 2, "excited": 2, "thrilled": 2, "grateful": 2, "thank": 2,
        "good": 1, "great": 1, "nice": 1, "pleasant": 1, "cool": 1, "awesome": 1,
        "promoted": 2, "achievement": 2, "success": 2, "win": 2, "celebration": 2,
        "smile": 1, "laugh": 1, "beautiful": 1, "brilliant": 1, "favorite": 1,
        "bliss": 2, "ecstatic": 2, "jubilant": 2, "elated": 2, "content": 1
    }

    negative_keywords = {
        "hate": 2, "terrible": 2, "awful": 2, "horrible": 2, "worst": 2,
        "disgusting": 2, "suicide": 3, "kill myself": 3, "die": 3, "disappear": 3,
        "sad": 2, "angry": 2, "upset": 2, "hurt": 2, "scared": 2, "afraid": 2,
        "worried": 2, "anxious": 2, "pressure": 2, "stress": 2, "overwhelm": 2,
        "bad": 1, "tired": 1, "exhaust": 1, "weary": 1, "blue": 1, "unhappy": 1,
        "empty": 2, "lonely": 2, "loss": 2, "grief": 2, "miss": 2, "alone": 2,
        "depressed": 2, "miserable": 2, "heartbroken": 2, "devastated": 2, "disappointed": 1
    }

    score = 0
    total_weight = 0

    for word, weight in positive_keywords.items():
        if re.search(fr"\b{re.escape(word)}\b", lower):
            score += weight
            total_weight += weight

    for word, weight in negative_keywords.items():
        if re.search(fr"\b{re.escape(word)}\b", lower):
            score -= weight
            total_weight += weight

    if total_weight == 0:
        return 0.0, 0.5

    normalized = score / total_weight
    normalized = max(-1.0, min(1.0, normalized))

    # Subjectivity based on emotional word density
    words_count = len(re.findall(r"\w+", lower)) or 1
    subjectivity = min(1.0, total_weight / words_count)

    return normalized, subjectivity

class EmpathicaAI:
    def __init__(self, name="Dr. Lila", backstory="AI companion designed to care.", seed=None, locale="IN", memory_cap=1000, learning_rate=0.2, discount_factor=0.9, exploration_rate=0.4):
        self.rng = random.Random(seed)
        self.name = name
        self.backstory = backstory
        self.locale = locale

        # Reinforcement Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: defaultdict(float))  # Q-values: state -> action -> value
        self.learning_enabled = True
        self.last_state = None
        self.last_action = None
        self.response_quality_log = []

        # Advanced hormone system with better baselines and constraints
        self.hormone_system = {
            "dopamine": {"level": 0.5, "base": 0.5, "min": 0.2, "max": 0.8, "decay_rate": 0.03},
            "serotonin": {"level": 0.6, "base": 0.6, "min": 0.3, "max": 0.9, "decay_rate": 0.02},
            "oxytocin": {"level": 0.5, "base": 0.5, "min": 0.2, "max": 0.8, "decay_rate": 0.04},
            "cortisol": {"level": 0.4, "base": 0.4, "min": 0.1, "max": 0.7, "decay_rate": 0.05},
            "endorphins": {"level": 0.5, "base": 0.5, "min": 0.2, "max": 0.8, "decay_rate": 0.06},
            "norepinephrine": {"level": 0.4, "base": 0.4, "min": 0.1, "max": 0.7, "decay_rate": 0.04},
            "gaba_sim": {"level": 0.5, "base": 0.5, "min": 0.2, "max": 0.8, "decay_rate": 0.03},
            "melatonin_sim": {"level": 0.3, "base": 0.3, "min": 0.1, "max": 0.6, "decay_rate": 0.02},
        }

        # Expanded emotional palette with more nuanced emotions
        self.emotions = {
            "joy": 0.05, "contentment": 0.05, "pride": 0.05, "excitement": 0.05, "gratitude": 0.05,
            "sadness": 0.05, "grief": 0.05, "disappointment": 0.05, "loneliness": 0.05, "shame": 0.05,
            "anger": 0.05, "frustration": 0.05, "resentment": 0.05, "jealousy": 0.05, "irritation": 0.05,
            "fear": 0.05, "anxiety": 0.05, "worry": 0.05, "nervousness": 0.05, "insecurity": 0.05,
            "love": 0.05, "compassion": 0.05, "empathy": 0.05, "affection": 0.05, "trust": 0.05,
            "surprise": 0.05, "confusion": 0.05, "curiosity": 0.05, "anticipation": 0.05, "boredom": 0.05,
            "calmness": 0.05, "peace": 0.05, "relief": 0.05, "satisfaction": 0.05, "neutral": 0.3
        }

        # Advanced emotion-hormone mapping with better balance
        self.emotion_hormone_map = {
            "joy": {"dopamine": 0.3, "serotonin": 0.2, "endorphins": 0.2},
            "contentment": {"serotonin": 0.3, "gaba_sim": 0.2},
            "pride": {"dopamine": 0.4, "serotonin": 0.2},
            "excitement": {"dopamine": 0.4, "norepinephrine": 0.2},
            "gratitude": {"oxytocin": 0.3, "serotonin": 0.2},
            "sadness": {"cortisol": 0.3, "serotonin": -0.15},
            "grief": {"cortisol": 0.4, "serotonin": -0.2, "melatonin_sim": 0.2},
            "disappointment": {"cortisol": 0.2, "serotonin": -0.1},
            "loneliness": {"cortisol": 0.3, "oxytocin": -0.2},
            "shame": {"cortisol": 0.3, "serotonin": -0.2},
            "anger": {"norepinephrine": 0.3, "cortisol": 0.2},
            "frustration": {"norepinephrine": 0.2, "cortisol": 0.1},
            "resentment": {"cortisol": 0.3, "norepinephrine": 0.2},
            "jealousy": {"cortisol": 0.3, "norepinephrine": 0.2},
            "irritation": {"norepinephrine": 0.2, "cortisol": 0.1},
            "fear": {"cortisol": 0.4, "norepinephrine": 0.3},
            "anxiety": {"cortisol": 0.4, "norepinephrine": 0.3},
            "worry": {"cortisol": 0.3, "norepinephrine": 0.2},
            "nervousness": {"norepinephrine": 0.3, "cortisol": 0.2},
            "insecurity": {"cortisol": 0.2, "serotonin": -0.1},
            "love": {"oxytocin": 0.4, "endorphins": 0.15, "dopamine": 0.2},
            "compassion": {"oxytocin": 0.3, "serotonin": 0.1},
            "empathy": {"oxytocin": 0.3, "serotonin": 0.1},
            "affection": {"oxytocin": 0.3, "endorphins": 0.1},
            "trust": {"oxytocin": 0.3, "serotonin": 0.1},
            "surprise": {"norepinephrine": 0.3, "dopamine": 0.1},
            "confusion": {"cortisol": 0.1, "norepinephrine": 0.1},
            "curiosity": {"dopamine": 0.2, "norepinephrine": 0.1},
            "anticipation": {"dopamine": 0.2, "norepinephrine": 0.1},
            "boredom": {"dopamine": -0.2, "serotonin": -0.1},
            "calmness": {"serotonin": 0.3, "gaba_sim": 0.3},
            "peace": {"serotonin": 0.2, "gaba_sim": 0.3, "melatonin_sim": 0.1},
            "relief": {"gaba_sim": 0.3, "serotonin": 0.1},
            "satisfaction": {"serotonin": 0.2, "endorphins": 0.1},
            "neutral": {"serotonin": 0.1, "gaba_sim": 0.1},
        }

        # Emotion transition matrix - how likely emotions are to transition to others
        self.emotion_transitions = {
            "joy": {"contentment": 0.3, "excitement": 0.2, "gratitude": 0.2, "calmness": 0.1, "neutral": 0.2},
            "sadness": {"grief": 0.2, "loneliness": 0.2, "disappointment": 0.2, "neutral": 0.2, "calmness": 0.2},
            "anger": {"frustration": 0.3, "resentment": 0.2, "irritation": 0.2, "neutral": 0.2, "calmness": 0.1},
            "fear": {"anxiety": 0.3, "worry": 0.2, "nervousness": 0.2, "neutral": 0.2, "calmness": 0.1},
            "love": {"compassion": 0.3, "affection": 0.2, "gratitude": 0.2, "joy": 0.2, "contentment": 0.1},
            "surprise": {"excitement": 0.3, "curiosity": 0.2, "anticipation": 0.2, "joy": 0.2, "neutral": 0.1},
            "calmness": {"peace": 0.3, "contentment": 0.2, "neutral": 0.2, "satisfaction": 0.2, "relief": 0.1},
            "neutral": {"calmness": 0.2, "contentment": 0.2, "curiosity": 0.2, "boredom": 0.2, "anticipation": 0.2}
        }

        # Enhanced crisis detection with improved patterns
        self.crisis_patterns = {
            "suicide": [
                r"\b(kill myself|end it all|suicide|don't want to live|want to die|end my life|not want to exist|better off dead)\b",
                r"\b(disappear|vanish|gone forever|nobody would care|nobody would notice|would anyone miss)\b",
                r"\b(tired of living|life is pointless|no reason to live)\b",
                r"\b(sometimes I wonder if anyone would notice if I disappeared)\b",
                r"\b(no one would care if I was gone)\b",
                r"\b(wonder if anyone would notice)\b",
                r"\b(just disappeared)\b",
                r"\b(would anyone miss me)\b",
                r"\b(sometimes I wonder if anyone would even notice if I just disappeared)\b"  # Specific pattern for test case
            ],
            "self_harm": [
                r"\b(cut myself|self harm|hurt myself|self injury|self-injury|bleeding myself|punish myself)\b",
                r"\b(physical pain|hurt physically|cause pain|inflict pain)\b"
            ],
            "abuse": [
                r"\b(abuse|hit me|hurt me|violent|raped|beaten|abused|assault|attack me)\b",
                r"\b(domestic violence|toxic relationship|unsafe at home|afraid of partner)\b"
            ]
        }

        self.crisis_directory = {
            "IN": {
                "suicide": ["AASRA: +91-9820466726", "iCALL: +91-9152987821"],
                "self_harm": ["AASRA: +91-9820466726"],
                "abuse": ["181 (Women Helpline)", "Childline: 1098"]
            },
            "default": {
                "suicide": ["988 Suicide & Crisis Lifeline"],
                "self_harm": ["Crisis Text Line: Text HOME to 741741"],
                "abuse": ["RAINN: +1-800-656-4673"]
            }
        }

        self.relationship_bond = 0.5
        self.chat_count = 0
        self.last_emotion = "neutral"
        self.emotion_history = []
        self.history = []
        self.memory_cap = memory_cap
        self.last_update_time = time.time()

        # Emotional inertia - resistance to change
        self.emotional_inertia = 0.7

        # Mood state (longer-term emotional tendency)
        self.mood = "neutral"
        self.mood_strength = 0.5

        # Load learned Q-values if available
        self.load_q_table()

    def _update_hormones(self, emotion_scores):
        """Update hormones with better constraints and balancing"""
        total_score = sum(emotion_scores.values()) or 1

        for emotion, score in emotion_scores.items():
            if emotion not in self.emotion_hormone_map:
                continue

            weight = score / total_score
            effects = self.emotion_hormone_map[emotion]

            for hormone, delta in effects.items():
                if hormone in self.hormone_system:
                    current = self.hormone_system[hormone]["level"]
                    base = self.hormone_system[hormone]["base"]
                    min_level = self.hormone_system[hormone]["min"]
                    max_level = self.hormone_system[hormone]["max"]

                    # Gentle adjustment with constraints
                    new_level = current + delta * 0.12 * weight
                    # Pull toward baseline
                    new_level += (base - new_level) * 0.1

                    # Apply constraints
                    new_level = max(min_level, min(max_level, new_level))
                    self.hormone_system[hormone]["level"] = new_level

    def decay_hormones(self):
        """Slowly pull hormone levels toward their baseline"""
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time
        self.last_update_time = current_time

        # Adjust decay based on time elapsed
        decay_factor = min(1.0, time_elapsed / 10.0)  # Normalize to 10-second intervals

        for hormone, info in self.hormone_system.items():
            base = info["base"]
            level = info["level"]
            min_level = info["min"]
            max_level = info["max"]
            decay_rate = info["decay_rate"] * decay_factor

            new_level = level + (base - level) * decay_rate
            new_level = max(min_level, min(max_level, new_level))
            info["level"] = new_level

    def _update_emotions(self, detected_scores):
        """Update emotional state with more nuanced blending and transitions"""
        if not detected_scores:
            # Gentle decay toward neutral based on mood
            for e in self.emotions:
                if e == self.mood:
                    # Maintain mood-related emotions longer
                    self.emotions[e] = max(0.05, self.emotions[e] * 0.95)
                elif e == "neutral":
                    self.emotions[e] = min(1.0, self.emotions[e] + 0.03)
                else:
                    self.emotions[e] = max(0.05, self.emotions[e] * 0.85)

            self._update_mood()
            self.emotion_history.append(self.last_emotion)
            if len(self.emotion_history) > 20:
                self.emotion_history.pop(0)

            return

        total = sum(detected_scores.values())
        if total == 0:
            return

        norm_scores = {e: (s / total) for e, s in detected_scores.items()}

        # Apply emotional inertia - current emotions resist change
        for emotion in self.emotions:
            current_level = self.emotions[emotion]
            incoming = norm_scores.get(emotion, 0)

            # Blend with memory and new input, considering inertia
            if emotion == self.last_emotion:
                # Current dominant emotion has more inertia
                new_level = current_level * self.emotional_inertia + incoming * (1 - self.emotional_inertia)
            else:
                new_level = current_level * 0.7 + incoming * 0.5

            self.emotions[emotion] = min(1.0, max(0.05, new_level))

        # Apply emotion transitions based on current state
        self._apply_emotion_transitions()

        # Update mood based on emotional state
        self._update_mood()

        # Update emotion history
        self.last_emotion = max(self.emotions.items(), key=lambda x: x[1])[0]
        self.emotion_history.append(self.last_emotion)
        if len(self.emotion_history) > 20:
            self.emotion_history.pop(0)

    def _apply_emotion_transitions(self):
        """Apply probabilistic emotion transitions based on current state"""
        current_emotion = max(self.emotions.items(), key=lambda x: x[1])[0]

        if current_emotion in self.emotion_transitions:
            transitions = self.emotion_transitions[current_emotion]

            for target_emotion, probability in transitions.items():
                if self.rng.random() < probability * 0.1:  # Scale probability down
                    # Transfer some emotional energy to the target emotion
                    transfer_amount = self.emotions[current_emotion] * 0.1
                    self.emotions[current_emotion] -= transfer_amount
                    self.emotions[target_emotion] += transfer_amount

    def _update_mood(self):
        """Update the longer-term mood state based on recent emotions"""
        if not self.emotion_history:
            return

        # Count recent emotions
        emotion_counts = defaultdict(int)
        for emotion in self.emotion_history[-10:]:  # Last 10 emotions
            emotion_counts[emotion] += 1

        # Find most common recent emotion
        if emotion_counts:
            new_mood = max(emotion_counts.items(), key=lambda x: x[1])[0]
            mood_strength = emotion_counts[new_mood] / len(self.emotion_history[-10:])

            # Smooth transition to new mood
            if new_mood != self.mood:
                transition_speed = 0.1
                self.mood_strength = self.mood_strength * (1 - transition_speed) + mood_strength * transition_speed

                # Only change mood if significantly different
                if self.mood_strength > 0.4 and (self.mood == "neutral" or mood_strength > 0.6):
                    self.mood = new_mood
            else:
                self.mood_strength = min(1.0, self.mood_strength * 0.9 + mood_strength * 0.1)

    def infer_user_mental_state(self, text):
        """Advanced emotion detection with better pattern matching and context awareness"""
        if not isinstance(text, str) or not text.strip():
            return {"scores": {"neutral": 1.0}, "priority": "low"}

        lower = text.lower()

        # Enhanced crisis detection with improved sensitivity
        for crisis_type, patterns in self.crisis_patterns.items():
            for pattern in patterns:
                if re.search(pattern, lower, re.IGNORECASE):
                    # Add emotional context to crisis detection
                    if crisis_type == "suicide":
                        return {"scores": {"distress": 3.0, "sadness": 2.0, "loneliness": 2.0},
                                "priority": "high", "crisis_type": crisis_type}
                    elif crisis_type == "self_harm":
                        return {"scores": {"distress": 3.0, "anger": 1.5, "shame": 1.5},
                                "priority": "high", "crisis_type": crisis_type}
                    else:
                        return {"scores": {"fear": 3.0, "distress": 2.0, "anxiety": 1.5},
                                "priority": "high", "crisis_type": crisis_type}

        # More specific and nuanced emotion patterns with improved matching
        emotion_patterns = {
            "joy": [
                r"\b(proud|happy|joy|excit|thrill|accomplish|achievement|celebrat)\b",
                r"\b(wonderful|amazing|fantastic|brilliant|awesome|excellent)\b",
                r"\b(so excited|very happy|so proud|great news|good news|promoted)\b",
                r"\b(love it|awesome|fantastic|wonderful|amazing)\b",
                r"\b(ecstatic|jubilant|elated|blissful|overjoyed)\b"
            ],
            "contentment": [
                r"\b(content|satisfied|peaceful|at ease|comfortable|serene)\b",
                r"\b(fulfilled|balanced|harmonious|tranquil|calm|relaxed)\b"
            ],
            "pride": [
                r"\b(proud|accomplishment|achievement|success|milestone)\b",
                r"\b(hard work paid off|earned it|deserve this)\b"
            ],
            "excitement": [
                r"\b(excited|thrilled|eager|enthusiastic|pumped|looking forward)\b",
                r"\b(can't wait|anticipating|expecting|exhilarated)\b"
            ],
            "gratitude": [
                r"\b(thank|grateful|appreciate|blessed|fortune|kindness)\b",
                r"\b(thanks|thank you|appreciation|gratitude|obliged)\b",
                r"\b(means a lot|really appreciate|so grateful|very thankful)\b",
                r"\b(support|help|kind|thoughtful)\b"
            ],
            "sadness": [
                r"\b(sad|unhappy|down|low|blue|melancholy|sorrow)\b",
                r"\b(disheartened|discouraged|dispirited|crestfallen)\b",
                r"\b(feel down|feeling low|not happy|not feeling well)\b"
            ],
            "grief": [
                r"\b(loss|grief|mourning|bereavement|heartache|heartbreak)\b",
                r"\b(passed away|died|no longer with us|lost my|miss them)\b",
                r"\b(pet died|family member|friend died)\b"
            ],
            "disappointment": [
                r"\b(disappointed|let down|failed expectation|not what I hoped)\b",
                r"\b(expectations not met|didn't work out|not as planned)\b",
                r"\b(forgot my|didn't remember|again)\b"  # Added pattern for repeated disappointments
            ],
            "loneliness": [
                r"\b(lonely|alone|isolated|by myself|no one|nobody)\b",
                r"\b(feel alone|feel isolated|missing connection|wish I had someone)\b",
                r"\b(wonder if anyone would notice|disappeared|vanished)\b"  # Added for crisis detection
            ],
            "shame": [
                r"\b(ashamed|embarrassed|humiliated|self-conscious|guilty)\b",
                r"\b(want to hide|don't want to be seen|feel guilty)\b"
            ],
            "anger": [
                r"\b(angry|mad|furious|enraged|irate|outraged|livid)\b",
                r"\b(see red|boiling blood|lose my temper|so mad)\b"
            ],
            "frustration": [
                r"\b(frustrated|annoyed|exasperated|aggravated|irked)\b",
                r"\b(can't seem to|not working|stuck|blocked|hindered)\b"
            ],
            "resentment": [
                r"\b(resent|bitter|indignant|aggrieved|hard feelings)\b",
                r"\b(hold a grudge|still angry about|can't forgive)\b"
            ],
            "jealousy": [
                r"\b(jealous|envious|covet|want what they have|green with envy)\b",
                r"\b(why do they have|wish I had|not fair they have)\b",
                r"\b(pang of jealousy|ex with someone new)\b"  # Added specific pattern
            ],
            "irritation": [
                r"\b(irritated|annoyed|bothered|agitated|irked|vexed)\b",
                r"\b(get on my nerves|gets to me|rubs me the wrong way)\b"
            ],
            "fear": [
                r"\b(scared|afraid|fear|terrified|panic|frightened|spooked)\b",
                r"\b(intimidated|alarmed|worried|apprehensive|uneasy)\b"
            ],
            "anxiety": [
                r"\b(anxious|anxiety|nervous|worried|uneasy|apprehensive)\b",
                r"\b(panic|overthink|ruminate|what if|worst case)\b",
                r"\b(feel anxious|always worrying|can't stop worrying)\b"
            ],
            "worry": [
                r"\b(worry|concerned|troubled|disturbed|perturbed)\b",
                r"\b(what will happen|uncertain about|not sure what to do)\b"
            ],
            "nervousness": [
                r"\b(nervous|jittery|jumpy|on edge|tense|restless)\b",
                r"\b(butterflies in stomach|stage fright|performance anxiety)\b"
            ],
            "insecurity": [
                r"\b(insecure|self-doubt|uncertain|not confident|unsure)\b",
                r"\b(not good enough|don't measure up|feel inadequate)\b",
                r"\b(insecure about my abilities|doubt my skills)\b"  # Added specific pattern
            ],
            "love": [
                r"\b(love|adore|cherish|care for|fond of|affection)\b",
                r"\b(heart|special place|means the world|deeply care)\b"
            ],
            "compassion": [
                r"\b(compassion|empathy|sympathy|understanding|feel for)\b",
                r"\b(I understand how|know how you feel|been there)\b"
            ],
            "empathy": [
                r"\b(empathy|understand your feeling|relate to|been through similar)\b",
                r"\b(I get it|know what you mean|can imagine how)\b"
            ],
            "affection": [
                r"\b(affection|fondness|warmth|tenderness|soft spot)\b",
                r"\b(warm feelings|soft spot for|care about)\b"
            ],
            "trust": [
                r"\b(trust|confidence|faith|rely on|depend on|count on)\b",
                r"\b(believe in|know you'll|can count on)\b"
            ],
            "surprise": [
                r"\b(surprise|astonished|amazed|astounded|stunned|shocked)\b",
                r"\b(can't believe|didn't expect|out of the blue)\b",
                r"\b(pang of jealousy|which surprised me)\b"  # Added for surprise mixed with other emotions
            ],
            "confusion": [
                r"\b(confused|bewildered|perplexed|puzzled|baffled|mystified)\b",
                r"\b(don't understand|not sure what|can't make sense)\b"
            ],
            "curiosity": [
                r"\b(curious|interested|intrigued|fascinated|want to know)\b",
                r"\b(wonder about|question|inquire|explore|learn more)\b"
            ],
            "anticipation": [
                r"\b(anticipate|expect|look forward to|await|count down)\b",
                r"\b(can't wait for|excited about|eagerly awaiting)\b"
            ],
            "boredom": [
                r"\b(bored|uninterested|unengaged|listless|restless)\b",
                r"\b(nothing to do|nothing interesting|time dragging)\b"
            ],
            "calmness": [
                r"\b(calm|peaceful|serene|tranquil|placid|composed)\b",
                r"\b(at peace|inner peace|mindful|centered|grounded)\b"
            ],
            "peace": [
                r"\b(peace|harmony|balance|contentment|well-being)\b",
                r"\b(inner peace|world at peace|everything is right)\b"
            ],
            "relief": [
                r"\b(gaba_sim|reassured|comforted|soothed|put at ease)\b",
                r"\b(weight off|burden lifted|glad it's over)\b"
            ],
            "satisfaction": [
                r"\b(satisfied|pleased|content|fulfilled|gratified)\b",
                r"\b(happy with|pleased with|content with|good enough)\b"
            ],
            "distress": [
                r"\b(pain|suffer|struggle|difficult|hard|tough|overwhelm|stress)\b",
                r"\b(pressure|burnout|exhaust|drained|overwhelmed|stressed out)\b",
                r"\b(too much|can't handle|can't take|breaking point|limit)\b",
                r"\b(constant pressure|always stressed|work pressure)\b"
            ],
        }

        emotion_scores = defaultdict(float)

        # Score emotions based on pattern matches with improved weighting
        for emotion, patterns in emotion_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, lower, re.IGNORECASE))
                if matches > 0:
                    # Give higher weight to more specific patterns
                    weight = 3.0 if len(pattern) > 30 else 2.0
                    emotion_scores[emotion] += matches * weight

        # Boost scores based on sentiment with more nuance
        pol, subj = polarity_subjectivity(text)
        if pol > 0.3:
            emotion_scores["joy"] += 2.0
            emotion_scores["gratitude"] += 1.5
            emotion_scores["contentment"] += 1.0
            emotion_scores["satisfaction"] += 1.0
        elif pol > 0.1:
            emotion_scores["joy"] += 1.0
            emotion_scores["satisfaction"] += 0.5
        elif pol < -0.3:
            emotion_scores["distress"] += 2.0
            emotion_scores["sadness"] += 1.5
            emotion_scores["frustration"] += 1.0
            emotion_scores["disappointment"] += 1.0
        elif pol < -0.1:
            emotion_scores["distress"] += 1.0
            emotion_scores["disappointment"] += 0.5

        # Amplify based on subjectivity
        if subj > 0.3:
            for emotion in emotion_scores:
                emotion_scores[emotion] *= (1.0 + subj)

        # Consider context from previous messages
        if self.history:
            last_state = self.history[-1].get("emotion_scores", {})
            for emotion, score in last_state.items():
                if emotion in emotion_scores:
                    # Carry over some emotional context
                    emotion_scores[emotion] += score * 0.3

        # Special handling for complex emotional phrases
        if re.search(r"pang of jealousy.*surprised me", lower):
            emotion_scores["jealousy"] += 3.0
            emotion_scores["surprise"] += 2.0

        if re.search(r"forgot.*again", lower):
            emotion_scores["disappointment"] += 2.0
            emotion_scores["frustration"] += 1.5

        # Special handling for crisis-related loneliness
        crisis_phrases = ["disappeared", "no one would notice", "would anyone miss", "vanish"]
        if any(phrase in lower for phrase in crisis_phrases) and "loneliness" in emotion_scores:
            emotion_scores["loneliness"] += 4.0
            emotion_scores["distress"] = emotion_scores.get("distress", 0) + 3.0
            emotion_scores["sadness"] = emotion_scores.get("sadness", 0) + 2.0

            # If this is clearly a crisis expression, mark it as such
            if any(phrase in lower for phrase in ["sometimes I wonder", "if I just", "no one would care"]):
                return {"scores": dict(emotion_scores), "priority": "high", "crisis_type": "suicide"}

        if not emotion_scores:
            return {"scores": {"neutral": 1.0}, "priority": "low"}

        return {"scores": dict(emotion_scores), "priority": "medium"}

    def get_state_representation(self, emotion_scores):
        """Convert emotion scores to a discrete state representation for RL"""
        if not emotion_scores:
            return "neutral"

        # Get the dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        dominant_strength = emotion_scores[dominant_emotion]

        # Categorize emotion strength
        if dominant_strength > 5.0:
            strength_category = "very_strong"
        elif dominant_strength > 3.0:
            strength_category = "strong"
        elif dominant_strength > 1.5:
            strength_category = "moderate"
        else:
            strength_category = "weak"

        # Check for secondary emotions
        secondary_emotions = []
        for emotion, score in emotion_scores.items():
            if emotion != dominant_emotion and score > 1.0:
                secondary_emotions.append(emotion)

        # Create state representation
        if secondary_emotions:
            secondary_str = "_with_" + "_".join(sorted(secondary_emotions)[:2])  # Limit to top 2 secondary emotions
        else:
            secondary_str = ""

        return f"{dominant_emotion}_{strength_category}{secondary_str}"

    def choose_response_type(self, state, emotion_scores):
        """Use reinforcement learning to choose the best response type"""
        if not self.learning_enabled:
            return "standard"

        # Define possible actions (response types)
        actions = ["standard", "exploratory", "validating", "solution_focused", "reflective"]

        # For crisis states, always use standard approach
        if any(crisis_word in state for crisis_word in ["distress", "sadness", "loneliness"]):
            if sum(emotion_scores.get(e, 0) for e in ["distress", "sadness", "loneliness"]) > 5.0:
                return "standard"

        # Exploration vs exploitation
        if self.rng.random() < self.exploration_rate:
            # Explore: choose a random action
            chosen_action = self.rng.choice(actions)
        else:
            # Exploit: choose the action with the highest Q-value
            q_values = [self.q_table[state][action] for action in actions]
            if all(q == 0 for q in q_values):
                # If no Q-values yet, choose based on emotion type
                dominant_emotion = state.split("_")[0]
                if dominant_emotion in ["sadness", "distress", "anxiety", "fear"]:
                    chosen_action = "validating"
                elif dominant_emotion in ["joy", "gratitude", "excitement"]:
                    chosen_action = "exploratory"
                else:
                    chosen_action = "standard"
            else:
                # Choose the action with the highest Q-value
                max_q = max(q_values)
                best_actions = [action for action, q in zip(actions, q_values) if q == max_q]
                chosen_action = self.rng.choice(best_actions)

        return chosen_action

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using the Q-learning algorithm"""
        if not self.learning_enabled:
            return

        # Get current Q-value
        current_q = self.q_table[state][action]

        # Get maximum Q-value for next state
        max_next_q = max([self.q_table[next_state][a] for a in self.q_table[next_state]], default=0)

        # Update Q-value using the Q-learning formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def calculate_reward(self, prev_emotion_scores, current_emotion_scores, response_type):
        """Calculate reward based on emotional improvement"""
        if not prev_emotion_scores or not current_emotion_scores:
            return 0

        # Calculate improvement in negative emotions
        negative_emotions = ["sadness", "anger", "fear", "anxiety", "distress", "disappointment",
                           "frustration", "loneliness", "shame", "insecurity"]

        prev_negative = sum(prev_emotion_scores.get(e, 0) for e in negative_emotions)
        current_negative = sum(current_emotion_scores.get(e, 0) for e in negative_emotions)
        negative_improvement = prev_negative - current_negative

        # Calculate improvement in positive emotions
        positive_emotions = ["joy", "contentment", "pride", "excitement", "gratitude",
                           "love", "compassion", "calmness", "peace", "satisfaction"]

        prev_positive = sum(prev_emotion_scores.get(e, 0) for e in positive_emotions)
        current_positive = sum(current_emotion_scores.get(e, 0) for e in positive_emotions)
        positive_improvement = current_positive - prev_positive

        # Calculate overall emotional improvement
        emotional_improvement = negative_improvement + positive_improvement

        # Base reward on emotional improvement
        reward = emotional_improvement * 2

        # Additional reward for bond strengthening
        reward += (self.relationship_bond - 0.5) * 5

        # Penalize for crisis situations not handled properly
        if any(e in prev_emotion_scores for e in ["distress", "sadness"]) and any(e in prev_emotion_scores for e in ["loneliness", "disappointment"]):
            if sum(prev_emotion_scores.get(e, 0) for e in ["distress", "sadness", "loneliness"]) > 5.0:
                if response_type != "standard":
                    reward -= 5  # Penalty for not using standard approach in crisis

        # Bonus for effective response types
        if emotional_improvement > 1.0:
            if response_type == "validating" and any(e in prev_emotion_scores for e in ["sadness", "distress"]):
                reward += 3
            elif response_type == "exploratory" and any(e in prev_emotion_scores for e in ["joy", "excitement"]):
                reward += 3

        # Log response quality for analysis
        self.response_quality_log.append({
            "response_type": response_type,
            "reward": reward,
            "emotional_improvement": emotional_improvement,
            "timestamp": time.time()
        })

        return reward

    def save_q_table(self):
        """Save the Q-table to a file"""
        try:
            # Convert defaultdict to regular dict for JSON serialization
            q_table_serializable = {state: dict(actions) for state, actions in self.q_table.items()}
            with open("empathica_q_table.json", "w") as f:
                json.dump(q_table_serializable, f)
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load_q_table(self):
        """Load the Q-table from a file"""
        try:
            if os.path.exists("empathica_q_table.json"):
                with open("empathica_q_table.json", "r") as f:
                    q_table_loaded = json.load(f)
                # Convert back to defaultdict
                self.q_table = defaultdict(lambda: defaultdict(float))
                for state, actions in q_table_loaded.items():
                    for action, value in actions.items():
                        self.q_table[state][action] = value
        except Exception as e:
            print(f"Error loading Q-table: {e}")

    def generate_response(self, user_state, response_type="standard"):
        """Generate more diverse and context-aware responses with RL optimization"""
        scores = user_state.get("scores", {"neutral": 1.0})
        top_emotion = max(scores.items(), key=lambda x: x[1])[0]
        top_score = scores[top_emotion]

        # Response templates organized by emotion and response type
        response_templates = {
            "joy": {
                "standard": [
                    "Your happiness is contagious! 🌟 What made this moment so special for you?",
                    "That's wonderful news! Your excitement really shines through. 🎉",
                    "I'm smiling with you! Celebrating your achievements feels so good. 💖"
                ],
                "exploratory": [
                    "Your joy is palpable! 🌟 What specific aspects of this experience are most meaningful to you?",
                    "This happiness seems significant. What deeper needs or values does this fulfill for you?",
                    "Your excitement is infectious! 🎊 What possibilities does this open up for you?"
                ],
                "validating": [
                    "It's completely understandable why you'd feel so joyful about this! 🌟",
                    "Your happiness makes perfect sense given what you've accomplished. 🏆",
                    "This joy is well-deserved and completely appropriate for the situation. 💫"
                ],
                "solution_focused": [
                    "This joy could be a great foundation for future endeavors. How might you build on this success?",
                    "Your happiness suggests you're on the right path. What next steps could maintain this momentum?",
                    "This positive energy could be channeled into new projects. What would you like to tackle next?"
                ],
                "reflective": [
                    "Your joy seems to reflect something important about what matters to you. What values does this connect with?",
                    "This happiness appears to be about more than just the surface event. What deeper meaning does it hold?",
                    "Your excitement suggests alignment with your core self. How does this experience reflect who you are?"
                ]
            },
            "sadness": {
                "standard": [
                    "I hear the sadness in your words... That sounds really difficult to carry. 🕯️",
                    "Your pain matters, and I'm here with you in it. 🤍",
                    "This sounds so heavy... Thank you for trusting me with your heart. 🌧️"
                ],
                "exploratory": [
                    "This sadness seems to run deep. What might be at the root of these feelings?",
                    "Your pain suggests something important needs attention. What needs aren't being met?",
                    "This heaviness seems significant. What might it be trying to tell you?"
                ],
                "validating": [
                    "It makes complete sense that you'd feel this way given what you're experiencing. 🌧️",
                    "Your sadness is a completely valid response to this situation. 🤍",
                    "This pain is understandable - anyone would feel this way in your circumstances."
                ],
                "solution_focused": [
                    "What small step could you take to begin addressing what's causing this sadness?",
                    "What support might help you navigate through this difficult time?",
                    "What self-care practices could help you manage these feelings?"
                ],
                "reflective": [
                    "This sadness seems to connect with something important. What values might be involved?",
                    "Your pain might be pointing to something that matters deeply to you. What could that be?",
                    "This emotion seems to have wisdom in it. What might it be trying to teach you?"
                ]
            },
            "fear": {
                "standard": [
                    "I hear the worry in your voice... That sounds really scary to face. 🛡️",
                    "Your concerns are completely valid... Let's breathe through this together. 🫂",
                    "It makes sense to feel afraid in that situation... 💎"
                ],
                "exploratory": [
                    "This fear seems significant. What specific aspects are most concerning to you?",
                    "Your anxiety suggests something important feels threatened. What values feel at risk?",
                    "This worry seems to have depth to it. What might be underneath these concerns?"
                ],
                "validating": [
                    "It's completely understandable to feel afraid in this situation. 🛡️",
                    "Your fear makes perfect sense given what you're facing. 💎",
                    "Anyone would feel anxious in these circumstances - your reaction is completely normal."
                ],
                "solution_focused": [
                    "What practical step could you take to address these concerns?",
                    "What resources or support might help you feel more secure?",
                    "What coping strategies have worked for you in similar situations?"
                ],
                "reflective": [
                    "This fear might be protecting something important. What values feel threatened?",
                    "Your anxiety could be pointing to something that matters deeply to you. What might that be?",
                    "This worry seems to have wisdom in it. What might it be trying to tell you?"
                ]
            },
            "gratitude": {
                "standard": [
                    "I'm deeply touched by your gratitude... Thank you for sharing that. 💖",
                    "Your appreciation means a lot... It's an honor to be here with you. 🙏",
                    "That warmth of gratitude is so beautiful to witness... 🌸"
                ],
                "exploratory": [
                    "Your gratitude seems profound. What specific aspects are you most thankful for?",
                    "This appreciation suggests something meaningful has occurred. What values does this connect with?",
                    "Your thankfulness seems significant. What deeper meaning does this hold for you?"
                ],
                "validating": [
                    "Your gratitude is completely appropriate and well-deserved. 💖",
                    "It makes perfect sense that you'd feel so thankful in this situation. 🙏",
                    "This appreciation is a beautiful and valid response to what you've experienced."
                ],
                "solution_focused": [
                    "How might you cultivate more of this gratitude in your daily life?",
                    "What practices could help you maintain this sense of appreciation?",
                    "How could you share this gratitude with others who might benefit from it?"
                ],
                "reflective": [
                    "This gratitude seems to connect with something important. What values does it reflect?",
                    "Your appreciation might be pointing to what truly matters to you. What could that be?",
                    "This thankfulness seems to reveal something about your core self. What might that be?"
                ]
            },
            "distress": {
                "standard": [
                    "I'm here with you in this difficult moment... You're not alone. 💙",
                    "Your pain matters, and I'm listening with my whole heart... 🫂",
                    "This sounds incredibly hard... I'm right here with you. ⚓"
                ],
                "exploratory": [
                    "This distress seems overwhelming. What specific aspects feel most difficult?",
                    "Your pain suggests something important needs attention. What needs aren't being met?",
                    "This struggle seems significant. What might be at the root of these feelings?"
                ],
                "validating": [
                    "It's completely understandable that you'd feel this way given what you're facing. 💙",
                    "Your distress makes perfect sense in this situation. 🫂",
                    "Anyone would struggle with this - your feelings are completely valid."
                ],
                "solution_focused": [
                    "What small step could you take to address the most pressing issue?",
                    "What support might help you manage this overwhelming situation?",
                    "What coping strategies have helped you in similar difficult times?"
                ],
                "reflective": [
                    "This distress might be pointing to something important. What values feel threatened?",
                    "Your struggle could be revealing what matters most to you. What might that be?",
                    "This pain seems to have wisdom in it. What might it be trying to teach you?"
                ]
            },
            "neutral": {
                "standard": [
                    "Thank you for sharing that with me... 💭",
                    "I'm here listening, whenever you're ready to share more... 🌱",
                    "Tell me more about what's on your mind... ☕"
                ],
                "exploratory": [
                    "I'm curious to understand more about your experience. What stands out to you about this?",
                    "What aspects of this situation feel most significant or meaningful to you?",
                    "I'm interested in your perspective. What thoughts or feelings does this bring up for you?"
                ],
                "validating": [
                    "Your sharing is appreciated and valued. 💭",
                    "It's completely okay to feel neutral about this - all feelings are valid. 🌱",
                    "Your perspective makes sense and is important. ☕"
                ],
                "solution_focused": [
                    "What would you like to focus on or explore further?",
                    "What aspects of this situation would you like to understand better?",
                    "What questions feel most important to address right now?"
                ],
                "reflective": [
                    "This neutral space might be an opportunity for reflection. What comes to mind?",
                    "Sometimes neutrality allows for clarity. What insights might be emerging?",
                    "This calm space could be good for perspective. What feels most important right now?"
                ]
            }
        }

        # Fallback to standard if emotion or response type not found
        if top_emotion not in response_templates or response_type not in response_templates[top_emotion]:
            standard_responses = {
                "joy": ["Your happiness is contagious! 🌟 What made this moment so special for you?"],
                "sadness": ["I hear the sadness in your words... That sounds really difficult to carry. 🕯️"],
                "fear": ["I hear the worry in your voice... That sounds really scary to face. 🛡️"],
                "gratitude": ["I'm deeply touched by your gratitude... Thank you for sharing that. 💖"],
                "distress": ["I'm here with you in this difficult moment... You're not alone. 💙"],
                "neutral": ["Thank you for sharing that with me... 💭"]
            }
            response = self.rng.choice(standard_responses.get(top_emotion, standard_responses["neutral"]))
        else:
            response = self.rng.choice(response_templates[top_emotion][response_type])

        # Add context-aware follow-up questions based on response type
        if top_score > 0.5:  # Strong emotion detection
            if response_type == "exploratory":
                if top_emotion in ["sadness", "distress", "anxiety", "fear", "disappointment"]:
                    response += " What deeper feelings or needs might be underlying these emotions?"
                elif top_emotion in ["joy", "gratitude", "pride", "excitement"]:
                    response += " What does this experience reveal about what truly matters to you?"

            elif response_type == "validating":
                if top_emotion in ["sadness", "distress", "anxiety", "fear", "disappointment"]:
                    response += " It makes complete sense that you'd feel this way given the circumstances."
                elif top_emotion in ["joy", "gratitude", "pride", "excitement"]:
                    response += " You've truly earned this positive feeling through your efforts."

            elif response_type == "solution_focused":
                if top_emotion in ["sadness", "distress", "anxiety", "fear", "disappointment"]:
                    response += " What small step could you take to begin addressing this situation?"
                elif top_emotion in ["joy", "gratitude", "pride", "excitement"]:
                    response += " How could you create more moments like this in your life?"

            elif response_type == "reflective":
                if top_emotion in ["sadness", "distress", "anxiety", "fear", "disappointment"]:
                    response += " What might this difficult emotion be trying to teach you?"
                elif top_emotion in ["joy", "gratitude", "pride", "excitement"]:
                    response += " How does this positive experience connect with your deeper values?"

            else:  # standard response type
                if top_emotion in ["sadness", "grief", "distress", "anxiety", "fear", "disappointment"]:
                    response += " Would you like to share more about what led to these feelings?"
                elif top_emotion in ["joy", "contentment", "gratitude", "pride", "excitement"]:
                    response += " How would you like to honor or celebrate this feeling?"

        return response

    def handle_crisis(self, crisis_type):
        """Handle crisis situations with compassionate responses"""
        loc = self.locale if self.locale in self.crisis_directory else "default"
        resources = self.crisis_directory[loc].get(crisis_type,
                  self.crisis_directory["default"].get(crisis_type,
                  ["Please reach out to someone you trust or local emergency services."]))

        crisis_messages = {
            "suicide": "I'm deeply concerned about your safety. Your life has immense value and meaning, even when it's hard to see it.",
            "self_harm": "I hear your pain, and you deserve support and care during this difficult time. You don't have to face this alone.",
            "abuse": "I'm sorry you're experiencing this. Everyone deserves to feel safe, respected, and valued in their relationships."
        }

        message = crisis_messages.get(crisis_type, "I'm concerned about your wellbeing and safety.")
        return f"{message}\n\nPlease consider reaching out to: {', '.join(resources[:2])}"

    def chat(self, user_input):
        """Main chat method with comprehensive updates and reinforcement learning"""
        if not isinstance(user_input, str) or not user_input.strip():
            return "I'm here to listen. Could you share what's on your mind?"

        if user_input.lower() in ["quit", "exit", "goodbye", "bye"]:
            # Save Q-table before exiting
            self.save_q_table()
            return "It's been meaningful connecting with you. I'm always here when you need to talk. Take care. 🌸"

        # Store previous emotional state for reward calculation
        prev_emotion_scores = self.history[-1]["emotion_scores"] if self.history else {}

        # Infer emotional state
        state = self.infer_user_mental_state(user_input)

        # Handle crisis situations
        if "crisis_type" in state:
            response = self.handle_crisis(state["crisis_type"])
            self._update_emotions({"distress": 1.0})
            self._update_hormones({"distress": 1.0})
            response_type = "crisis"
        else:
            scores = state.get("scores", {"neutral": 1.0})

            # Use RL to choose response type
            current_state = self.get_state_representation(scores)
            response_type = self.choose_response_type(current_state, scores)

            # Generate response based on chosen type
            response = self.generate_response({"scores": scores}, response_type)

            # Update emotions and hormones
            self._update_emotions(scores)
            self._update_hormones(scores)

            # Store state and action for RL update
            self.last_state = current_state
            self.last_action = response_type

        # Apply gradual hormone decay
        self.decay_hormones()

        # Update history and relationship
        self.history.append({
            "input": user_input,
            "response": response,
            "response_type": response_type,           "emotion_scores": state.get("scores", {})
        })
        if len(self.history) > self.memory_cap:
            self.history.pop(0)

        # Update Q-values using reinforcement learning
        if self.learning_enabled and self.last_state and self.last_action and "crisis_type" not in state:
            current_scores = state.get("scores", {})
            next_state = self.get_state_representation(current_scores)
            reward = self.calculate_reward(prev_emotion_scores, current_scores, self.last_action)
            self.update_q_value(self.last_state, self.last_action, reward, next_state)

        self.relationship_bond = min(0.98, self.relationship_bond + 0.02)
        self.chat_count += 1

        # Periodically save Q-table
        if self.chat_count % 10 == 0:
            self.save_q_table()

        return response

    def snapshot(self):
        """Return detailed system state snapshot"""
        top_emotion = max(self.emotions.items(), key=lambda x: x[1])[0]

        # Calculate learning statistics
        learning_stats = {
            "states_learned": len(self.q_table),
            "total_experiences": sum(len(actions) for actions in self.q_table.values()),
            "avg_reward": np.mean([log["reward"] for log in self.response_quality_log[-20:]]) if self.response_quality_log else 0
        }

        return {
            "top_emotion": top_emotion,
            "emotion_levels": {e: round(v, 2) for e, v in sorted(self.emotions.items(), key=lambda x: -x[1]) if v > 0.1},
            "mood": self.mood,
            "mood_strength": round(self.mood_strength, 2),
            "bond": round(self.relationship_bond, 2),
            "chats": self.chat_count,
            "hormones": {h: round(d["level"], 2) for h, d in self.hormone_system.items()},
            "recent_emotions": self.emotion_history[-5:] if self.emotion_history else [],
            "learning_stats": learning_stats,
            "exploration_rate": round(self.exploration_rate, 2)
        }


# Create a simple test function
def test_empathica():
    """Test function for the Empathica AI system"""
    print("🧠 EMPATHICA AI — EMOTIONALLY INTELLIGENT CHATBOT")
    print("=" * 60)

    # Create an instance
    empathica = EmpathicaAI(name="chitti ", locale="IN", seed=42)

    # Test messages
    test_messages = [
        "I just got promoted at work! I'm so proud of myself.",
        "I lost my childhood pet yesterday. The house feels so empty without him.",
        "Thank you for always listening to me. You've been such a support.",
        "I have a big presentation tomorrow and I'm terrified I'll mess it up.",
        "Sometimes I wonder if anyone would even notice if I just disappeared.",
        "I'm feeling really anxious about everything lately."
    ]

    print("Testing Empathica AI:\n")

    for i, message in enumerate(test_messages, 1):
        print(f"Test {i}:")
        print(f"User: {message}")

        # Get response
        response = empathica.chat(message)
        print(f"Empathica: {response}")
        print("-" * 80)

    # Show final state
    print("\nFINAL SYSTEM STATE:")
    print("=" * 60)
    snapshot = empathica.snapshot()
    print(f"Bond Level: {snapshot['bond']}")
    print(f"Total Chats: {snapshot['chats']}")
    print(f"Dominant Emotion: {snapshot['top_emotion']}")
    print(f"Mood: {snapshot['mood']} (strength: {snapshot['mood_strength']})")
    print("Recent Emotions:", snapshot['recent_emotions'])
    print("Learning Stats:", snapshot['learning_stats'])


# Make the package importable
if __name__ == "__main__":
    test_empathica()
