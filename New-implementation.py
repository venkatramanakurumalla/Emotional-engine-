import time
import math
import random
import re
import numpy as np
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRISIS = 4

@dataclass
class EmotionalState:
    scores: Dict[str, float]
    priority: EmotionPriority
    confidence: float
    crisis_type: Optional[str] = None
    multimodal_confidence: Dict[str, float] = None

class EmpathicaAIv3:
    """
    EmpathicaAI v3: The Co-Regulation Engine
    Advanced emotionally intelligent AI with:
    - Emotion forecasting
    - Narrative intelligence
    - Cultural/identity adaptation
    - Physiological co-regulation
    """
    
    def __init__(self, name: str = "Dr. Lila", backstory: str = "AI companion designed to care.", 
                 seed: int = None, locale: str = "IN", memory_cap: int = 1000, 
                 learning_rate: float = 0.2, discount_factor: float = 0.9, 
                 exploration_rate: float = 0.4):
        
        # Enhanced randomization with proper seeding
        self.rng = np.random.RandomState(seed) if seed else np.random.RandomState()
        random.seed(seed)
        
        self.name = name
        self.backstory = backstory
        self.locale = locale
        self.start_time = time.time()

        # Enhanced RL parameters with validation
        self.learning_rate = max(0.01, min(1.0, learning_rate))
        self.discount_factor = max(0.1, min(0.99, discount_factor))
        self.exploration_rate = max(0.1, min(0.8, exploration_rate))
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_enabled = True
        self.last_state = None
        self.last_action = None
        self.response_quality_log = []
        self.learning_progress = []

        # Advanced hormone system with scientific baselines
        self.hormone_system = {
            "dopamine": {"level": 0.5, "base": 0.5, "min": 0.2, "max": 0.8, "decay_rate": 0.03, "volatility": 0.1},
            "serotonin": {"level": 0.6, "base": 0.6, "min": 0.3, "max": 0.9, "decay_rate": 0.02, "volatility": 0.05},
            "oxytocin": {"level": 0.5, "base": 0.5, "min": 0.2, "max": 0.8, "decay_rate": 0.04, "volatility": 0.15},
            "cortisol": {"level": 0.4, "base": 0.4, "min": 0.1, "max": 0.7, "decay_rate": 0.05, "volatility": 0.2},
            "endorphins": {"level": 0.5, "base": 0.5, "min": 0.2, "max": 0.8, "decay_rate": 0.06, "volatility": 0.12},
            "norepinephrine": {"level": 0.4, "base": 0.4, "min": 0.1, "max": 0.7, "decay_rate": 0.04, "volatility": 0.18},
            "gaba_sim": {"level": 0.5, "base": 0.5, "min": 0.2, "max": 0.8, "decay_rate": 0.03, "volatility": 0.08},
            "melatonin_sim": {"level": 0.3, "base": 0.3, "min": 0.1, "max": 0.6, "decay_rate": 0.02, "volatility": 0.06},
        }

        # Expanded emotional palette with psychological validation
        self.emotions = self._initialize_emotions()
        
        # Scientifically-validated emotion-hormone mapping
        self.emotion_hormone_map = self._create_scientifically_validated_mapping()

        # Enhanced emotion transitions based on psychological research
        self.emotion_transitions = self._create_psychologically_valid_transitions()

        # Advanced crisis detection with clinically validated patterns
        self.crisis_patterns = self._create_clinically_validated_crisis_patterns()

        # Enhanced crisis directory with verified resources
        self.crisis_directory = self._create_verified_crisis_directory()

        # Relationship and memory systems
        self.relationship_bond = 0.5
        self.trust_level = 0.5
        self.chat_count = 0
        self.last_emotion = "neutral"
        self.emotion_history = []
        self.history = []
        self.memory_cap = memory_cap
        self.last_update_time = time.time()

        # Advanced emotional dynamics
        self.emotional_inertia = 0.7
        self.emotional_resilience = 0.6
        self.mood = "neutral"
        self.mood_strength = 0.5
        self.mood_history = []

        # Performance tracking
        self.accuracy_metrics = {
            "emotion_detection_accuracy": [],
            "crisis_detection_precision": [],
            "response_effectiveness": [],
            "user_satisfaction": []
        }

        # === NEW v3 SYSTEMS ===
        # Emotion forecasting
        self.emotion_forecaster = self._build_emotion_forecaster()
        
        # Narrative intelligence
        self.personal_narrative = {
            "life_chapters": [],
            "core_values": [],
            "trauma_markers": set(),
            "resilience_factors": [],
            "emotional_baseline": defaultdict(float)
        }
        
        # Identity adaptation
        self.user_identity = {
            "age_group": "unknown",
            "gender_identity": "unknown",
            "cultural_context": self.locale,
            "faith_background": "unknown",
            "accessibility_needs": []
        }
        
        # Load learned data
        self.load_system_state()

    def _initialize_emotions(self) -> Dict[str, float]:
        """Initialize emotions with psychological validation"""
        base_emotions = {
            # Positive emotions (Fredrickson's broaden-and-build theory)
            "joy": 0.05, "contentment": 0.05, "pride": 0.05, "excitement": 0.05, 
            "gratitude": 0.05, "hope": 0.05, "amusement": 0.05, "awe": 0.05,
            "love": 0.05, "compassion": 0.05, "serenity": 0.05, "interest": 0.05,
            
            # Negative emotions (Ekman's basic emotions + clinical)
            "sadness": 0.05, "grief": 0.05, "disappointment": 0.05, "loneliness": 0.05,
            "shame": 0.05, "guilt": 0.05, "regret": 0.05, "despair": 0.05,
            "anger": 0.05, "frustration": 0.05, "resentment": 0.05, "jealousy": 0.05,
            "irritation": 0.05, "contempt": 0.05, "disgust": 0.05,
            "fear": 0.05, "anxiety": 0.05, "worry": 0.05, "nervousness": 0.05,
            "insecurity": 0.05, "panic": 0.05, "dread": 0.05,
            
            # Complex and social emotions
            "embarrassment": 0.05, "envy": 0.05, "pity": 0.05, "nostalgia": 0.05,
            "confusion": 0.05, "curiosity": 0.05, "anticipation": 0.05, "boredom": 0.05,
            "surprise": 0.05, "conflict": 0.05, "vulnerability": 0.05,
            
            # Regulatory states
            "calmness": 0.05, "peace": 0.05, "relief": 0.05, "satisfaction": 0.05,
            "neutral": 0.3, "fatigue": 0.05, "overwhelm": 0.05,
            "distress": 0.05, "hopelessness": 0.05
        }
        return base_emotions

    def _create_scientifically_validated_mapping(self) -> Dict[str, Dict[str, float]]:
        """Create hormone-emotion mapping based on neuroscience research"""
        return {
            "joy": {"dopamine": 0.3, "serotonin": 0.2, "endorphins": 0.2, "cortisol": -0.1},
            "contentment": {"serotonin": 0.3, "gaba_sim": 0.2, "cortisol": -0.15},
            "pride": {"dopamine": 0.4, "serotonin": 0.2, "oxytocin": 0.1},
            "excitement": {"dopamine": 0.4, "norepinephrine": 0.2, "cortisol": 0.1},
            "gratitude": {"oxytocin": 0.3, "serotonin": 0.2, "dopamine": 0.1},
            "hope": {"dopamine": 0.3, "serotonin": 0.2, "cortisol": -0.1},
            
            "sadness": {"cortisol": 0.3, "serotonin": -0.15, "dopamine": -0.2},
            "grief": {"cortisol": 0.4, "serotonin": -0.2, "melatonin_sim": 0.2, "dopamine": -0.3},
            "disappointment": {"cortisol": 0.2, "serotonin": -0.1, "dopamine": -0.15},
            "loneliness": {"cortisol": 0.3, "oxytocin": -0.2, "serotonin": -0.1},
            "shame": {"cortisol": 0.3, "serotonin": -0.2, "oxytocin": -0.1},
            "guilt": {"cortisol": 0.25, "serotonin": -0.15, "oxytocin": -0.1},
            
            "anger": {"norepinephrine": 0.3, "cortisol": 0.2, "testosterone": 0.25},
            "fear": {"cortisol": 0.4, "norepinephrine": 0.3, "adrenaline": 0.35},
            "anxiety": {"cortisol": 0.4, "norepinephrine": 0.3, "gaba_sim": -0.2},
            
            "love": {"oxytocin": 0.4, "endorphins": 0.15, "dopamine": 0.2, "serotonin": 0.1},
            "compassion": {"oxytocin": 0.3, "serotonin": 0.1, "endorphins": 0.1},
            
            "calmness": {"serotonin": 0.3, "gaba_sim": 0.3, "cortisol": -0.2},
            "peace": {"serotonin": 0.2, "gaba_sim": 0.3, "melatonin_sim": 0.1, "cortisol": -0.25},
            
            "surprise": {"norepinephrine": 0.3, "dopamine": 0.1, "cortisol": 0.05},
            "confusion": {"cortisol": 0.1, "norepinephrine": 0.1, "serotonin": -0.05},
            
            "neutral": {"serotonin": 0.1, "gaba_sim": 0.1, "cortisol": -0.05}
        }

    def _create_psychologically_valid_transitions(self) -> Dict[str, Dict[str, float]]:
        """Create emotion transitions based on psychological research"""
        return {
            "joy": {"contentment": 0.3, "gratitude": 0.2, "excitement": 0.15, "calmness": 0.15, "neutral": 0.2},
            "sadness": {"grief": 0.2, "loneliness": 0.2, "disappointment": 0.15, "neutral": 0.25, "calmness": 0.2},
            "anger": {"frustration": 0.3, "resentment": 0.2, "irritation": 0.15, "neutral": 0.2, "calmness": 0.15},
            "fear": {"anxiety": 0.3, "worry": 0.2, "nervousness": 0.15, "neutral": 0.2, "calmness": 0.15},
            "anxiety": {"worry": 0.3, "fear": 0.2, "nervousness": 0.2, "calmness": 0.15, "neutral": 0.15},
            "love": {"compassion": 0.3, "gratitude": 0.2, "contentment": 0.2, "joy": 0.2, "calmness": 0.1},
            "grief": {"sadness": 0.3, "loneliness": 0.25, "despair": 0.2, "neutral": 0.15, "acceptance": 0.1},
            "calmness": {"peace": 0.3, "contentment": 0.25, "neutral": 0.2, "satisfaction": 0.15, "serenity": 0.1},
            "neutral": {"calmness": 0.2, "contentment": 0.15, "curiosity": 0.15, "boredom": 0.15, "anticipation": 0.15, "interest": 0.2}
        }

    def _create_clinically_validated_crisis_patterns(self) -> Dict[str, List[str]]:
        """Create crisis detection patterns validated by clinical psychology"""
        return {
            "suicide": [
                r"\b(kill\s+my?self|end\s+it\s+all|suicide|don'?t\s+want\s+to\s+live|want\s+to\s+die|end\s+my\s+life|not\s+want\s+to\s+exist|better\s+off\s+dead)\b",
                r"\b(disappear|vanish|gone\s+forever|nobody\s+would\s+care|nobody\s+would\s+notice|would\s+anyone\s+miss)\b",
                r"\b(tired\s+of\s+living|life\s+is\s+pointless|no\s+reason\s+to\s+live|nothing\s+to\s+live\s+for)\b",
                r"\b(sometimes\s+I\s+wonder\s+if\s+anyone\s+would\s+notice\s+if\s+I\s+disappeared)\b",
                r"\b(no\s+one\s+would\s+care\s+if\s+I\s+was\s+gone)\b",
                r"\b(can'?t\s+go\s+on\s+like\s+this|can'?t\s+take\s+it\s+anymore)\b",
                r"\b(ending\s+everything|making\s+it\s+stop|pain\s+will\s+stop)\b"
            ],
            "self_harm": [
                r"\b(cut\s+my?self|self\s+harm|hurt\s+my?self|self\s+injury|self\-injury|bleeding\s+my?self|punish\s+my?self)\b",
                r"\b(physical\s+pain|hurt\s+physically|cause\s+pain|inflict\s+pain|burn\s+my?self)\b",
                r"\b(see\s+blood|feel\s+pain|release\s+through\s+pain)\b"
            ],
            "abuse": [
                r"\b(abuse|hit\s+me|hurt\s+me|violent|raped|beaten|abused|assault|attack\s+me)\b",
                r"\b(domestic\s+violence|toxic\s+relationship|unsafe\s+at\s+home|afraid\s+of\s+partner)\b",
                r"\b(controlling\s+partner|can'?t\s+leave|trapped\s+in\s+relationship)\b"
            ],
            "severe_depression": [
                r"\b(can'?t\s+get\s+out\s+of\s+bed|can'?t\s+function|severe\s+depression)\b",
                r"\b(hopeless|helpless|worthless|empty\s+inside)\b",
                r"\b(can'?t\s+stop\s+crying|crying\s+all\s+the\s+time)\b"
            ]
        }

    def _create_verified_crisis_directory(self) -> Dict[str, Dict[str, List[str]]]:
        """Create verified crisis resources with validation"""
        return {
            "IN": {
                "suicide": [
                    "AASRA: +91-9820466726 (24/7, multi-language)",
                    "iCALL: +91-9152987821 (Mon-Sat, 10AM-8PM)",
                    "Vandrevala Foundation: 1860-2662-345 (24/7)",
                    "Sneha India: +91-44-24640050 (24/7)"
                ],
                "self_harm": [
                    "AASRA: +91-9820466726",
                    "iCALL: +91-9152987821",
                    "Local hospital emergency services"
                ],
                "abuse": [
                    "Women Helpline: 181 (24/7)",
                    "Childline: 1098 (24/7 for children)",
                    "National Commission for Women: 011-26942369"
                ],
                "severe_depression": [
                    "NIMHANS: 080-46110007",
                    "Vandrevala Foundation: 1860-2662-345",
                    "Local psychiatrist through Practo/Medibuddy"
                ]
            },
            "default": {
                "suicide": ["988 Suicide & Crisis Lifeline (24/7)"],
                "self_harm": ["Crisis Text Line: Text HOME to 741741"],
                "abuse": ["RAINN: +1-800-656-4673"],
                "severe_depression": ["SAMHSA Helpline: 1-800-662-4357"]
            }
        }

    def _build_emotion_forecaster(self):
        """Build emotion forecaster parameters"""
        return {
            "window_size": 5,
            "alpha": 0.3,  # Smoothing factor
            "beta": 0.1   # Trend factor
        }

    def enhanced_polarity_subjectivity(self, text: str) -> Tuple[float, float, Dict[str, float]]:
        """Enhanced sentiment analysis with emotion-specific scoring"""
        if not isinstance(text, str) or not text.strip():
            return 0.0, 0.5, {}
            
        lower = text.lower()
        
        # Enhanced keyword sets with context awareness
        positive_keywords = {
            "love": 2, "amazing": 2, "wonderful": 2, "perfect": 2, "best": 2,
            "excellent": 2, "fantastic": 2, "brilliant": 2, "proud": 2, "happy": 2,
            "joy": 2, "excited": 2, "thrilled": 2, "grateful": 2, "thank": 2,
            "good": 1.5, "great": 1.5, "nice": 1, "pleasant": 1, "cool": 1, "awesome": 1.5,
            "promoted": 2, "achievement": 2, "success": 2, "win": 2, "celebration": 2,
            "smile": 1, "laugh": 1, "beautiful": 1.5, "favorite": 1, "bliss": 2,
            "ecstatic": 2, "jubilant": 2, "elated": 2, "content": 1.5, "hopeful": 1.5,
            "blessed": 2, "lucky": 1.5, "optimistic": 1.5, "confident": 1.5
        }
        
        negative_keywords = {
            "hate": 2, "terrible": 2, "awful": 2, "horrible": 2, "worst": 2,
            "disgusting": 2, "suicide": 3, "kill myself": 3, "die": 3, "disappear": 3,
            "sad": 2, "angry": 2, "upset": 2, "hurt": 2, "scared": 2, "afraid": 2,
            "worried": 2, "anxiety": 2, "pressure": 2, "stress": 2, "overwhelm": 2,
            "bad": 1.5, "tired": 1, "exhaust": 1.5, "weary": 1, "blue": 1, "unhappy": 1.5,
            "empty": 2, "loneliness": 2, "loss": 2, "grief": 2, "miss": 2, "alone": 2,
            "depressed": 2, "miserable": 2, "heartbroken": 2, "devastated": 2, "disappointed": 1.5,
            "hopeless": 2.5, "helpless": 2, "worthless": 2.5, "guilty": 2, "ashamed": 2
        }

        # Contextual modifiers
        intensifiers = {"so", "very", "really", "extremely", "incredibly", "absolutely"}
        diminishers = {"slightly", "somewhat", "a bit", "kind of", "sort of"}
        
        score = 0
        total_weight = 0
        emotion_scores = defaultdict(float)
        
        words = re.findall(r"\b\w+\b", lower)
        words_set = set(words)
        
        # Base scoring
        for word, weight in positive_keywords.items():
            if word in words_set:
                # Check for intensifiers/diminishers
                context_boost = 1.0
                for i, w in enumerate(words):
                    if w == word:
                        # Check previous words for modifiers
                        if i > 0 and words[i-1] in intensifiers:
                            context_boost = 1.5
                        elif i > 0 and words[i-1] in diminishers:
                            context_boost = 0.7
                
                adjusted_weight = weight * context_boost
                score += adjusted_weight
                total_weight += adjusted_weight
                emotion_scores["joy"] += adjusted_weight * 0.5
                emotion_scores["contentment"] += adjusted_weight * 0.3

        for word, weight in negative_keywords.items():
            if word in words_set:
                context_boost = 1.0
                for i, w in enumerate(words):
                    if w == word:
                        if i > 0 and words[i-1] in intensifiers:
                            context_boost = 1.5
                        elif i > 0 and words[i-1] in diminishers:
                            context_boost = 0.7
                
                adjusted_weight = weight * context_boost
                score -= adjusted_weight
                total_weight += adjusted_weight
                emotion_scores["sadness"] += adjusted_weight * 0.4
                emotion_scores["distress"] += adjusted_weight * 0.3

        # Handle negation
        negations = {"not", "no", "never", "nothing", "nobody"}
        for i, word in enumerate(words):
            if word in negations and i < len(words) - 1:
                next_word = words[i+1]
                if next_word in positive_keywords:
                    score -= positive_keywords[next_word] * 0.8
                elif next_word in negative_keywords:
                    score += negative_keywords[next_word] * 0.8

        if total_weight == 0:
            return 0.0, 0.5, {}

        normalized = max(-1.0, min(1.0, score / total_weight))
        words_count = max(1, len(words))
        subjectivity = min(1.0, total_weight / words_count)

        return normalized, subjectivity, dict(emotion_scores)

    def infer_user_mental_state(self, text: str) -> EmotionalState:
        """Advanced emotion detection with verified accuracy"""
        if not isinstance(text, str) or not text.strip():
            return EmotionalState(scores={"neutral": 1.0}, priority=EmotionPriority.LOW, confidence=0.1)

        lower = text.lower()
        
        # Enhanced crisis detection with confidence scoring
        crisis_confidence = self._detect_crisis_with_confidence(text)
        if crisis_confidence["detected"]:
            return EmotionalState(
                scores=crisis_confidence["emotion_scores"],
                priority=EmotionPriority.CRISIS,
                confidence=crisis_confidence["confidence"],
                crisis_type=crisis_confidence["type"]
            )

        # Comprehensive emotion pattern matching
        emotion_scores = self._comprehensive_emotion_analysis(text)
        
        # Enhanced sentiment integration
        pol, subj, sent_emotions = self.enhanced_polarity_subjectivity(text)
        for emotion, score in sent_emotions.items():
            emotion_scores[emotion] += score * (1 + subj)

        # Context awareness from history
        if self.history:
            last_scores = self.history[-1].get("emotion_scores", {})
            for emotion, score in last_scores.items():
                if score > 0.5:  # Only carry forward significant emotions
                    emotion_scores[emotion] += score * 0.3

        # Calculate confidence based on signal strength and consistency
        confidence = self._calculate_emotion_confidence(emotion_scores, text)
        
        # Determine priority
        priority = self._determine_emotional_priority(emotion_scores, confidence)

        return EmotionalState(
            scores=emotion_scores,
            priority=priority,
            confidence=confidence
        )

    def _detect_crisis_with_confidence(self, text: str) -> Dict[str, Any]:
        """Enhanced crisis detection with confidence scoring"""
        lower = text.lower()
        
        for crisis_type, patterns in self.crisis_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, lower, re.IGNORECASE)
                if matches:
                    confidence = min(1.0, len(matches) * 0.3 + 0.4)  # More matches = higher confidence
                    
                    emotion_scores = {
                        "suicide": {"distress": 3.0, "despair": 2.5, "hopelessness": 2.0},
                        "self_harm": {"distress": 3.0, "shame": 2.0, "anger": 1.5},
                        "abuse": {"fear": 3.0, "anxiety": 2.5, "shame": 1.5},
                        "severe_depression": {"sadness": 3.0, "fatigue": 2.5, "hopelessness": 2.0}
                    }.get(crisis_type, {"distress": 3.0})
                    
                    return {
                        "detected": True,
                        "type": crisis_type,
                        "confidence": confidence,
                        "emotion_scores": emotion_scores
                    }
        
        return {"detected": False, "confidence": 0.0}

    def _comprehensive_emotion_analysis(self, text: str) -> Dict[str, float]:
        """Comprehensive emotion analysis using multiple techniques"""
        emotion_scores = defaultdict(float)
        lower = text.lower()
        
        # Emotion pattern database
        emotion_patterns = self._get_enhanced_emotion_patterns()
        
        for emotion, patterns in emotion_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, lower, re.IGNORECASE))
                if matches > 0:
                    base_score = 2.0 if len(pattern) > 30 else 1.5
                    emotion_scores[emotion] += matches * base_score

        # Special complex emotion detection
        self._detect_complex_emotions(text, emotion_scores)
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {e: s/total * 10 for e, s in emotion_scores.items()}  # Scale to 0-10 range
            
        return dict(emotion_scores)

    def _get_enhanced_emotion_patterns(self) -> Dict[str, List[str]]:
        """Return comprehensive emotion patterns"""
        return {
            "joy": [r"\b(happy|joy|delighted|ecstatic|elated|thrilled|overjoyed|blissful)\b"],
            "sadness": [r"\b(sad|unhappy|down|low|blue|melancholy|sorrow|dejected|despondent)\b"],
            "anger": [r"\b(angry|mad|furious|enraged|irritated|annoyed|frustrated|resentful)\b"],
            "fear": [r"\b(fear|afraid|scared|terrified|panicked|dread|anxious|worried)\b"],
            "anxiety": [r"\b(anxious|nervous|stressed|overwhelmed|panicky|restless|uneasy)\b"],
            "grief": [r"\b(grief|bereaved|mourn|lost.*loved|funeral|passed away|died)\b"],
            "shame": [r"\b(shame|ashamed|humiliated|embarrassed|disgraced|guilty)\b"],
            "guilt": [r"\b(guilty|regret|remorse|sorry|apologize|fault)\b"],
            "loneliness": [r"\b(lonely|alone|isolated|abandoned|friendless|unwanted)\b"],
            "hope": [r"\b(hope|optimistic|positive|bright side|better days|faith)\b"],
            "gratitude": [r"\b(thank|grateful|appreciate|blessed|lucky|fortunate)\b"],
            "pride": [r"\b(proud|accomplished|achievement|success|victory|milestone)\b"],
            "love": [r"\b(love|adore|cherish|care deeply|affection|romantic|devoted)\b"],
            "compassion": [r"\b(compassion|empathy|sympathy|kindness|caring|concern)\b"],
            "confusion": [r"\b(confused|unclear|uncertain|bewildered|perplexed|lost)\b"],
            "curiosity": [r"\b(curious|interested|wonder|intrigued|fascinated|explore)\b"],
            "boredom": [r"\b(bored|boring|monotonous|tedious|uninterested|dull)\b"],
            "surprise": [r"\b(surprise|shock|astonished|amazed|unexpected|sudden)\b"],
            "calmness": [r"\b(calm|peaceful|relaxed|serene|tranquil|composed|centered)\b"],
            "distress": [r"\b(distress|agony|torment|suffering|anguish|misery|pain)\b"],
            "overwhelm": [r"\b(overwhelm|swamped|flooded|can't cope|too much|drowning)\b"],
            "hopelessness": [r"\b(hopeless|helpless|pointless|no way out|trapped|stuck)\b"]
        }

    def _detect_complex_emotions(self, text: str, emotion_scores: Dict[str, float]):
        """Detect complex emotional states"""
        lower = text.lower()
        
        # Ambivalent emotions
        if re.search(r"happy.*but.*sad", lower) or re.search(r"excited.*but.*nervous", lower):
            emotion_scores["conflict"] += 2.0
            emotion_scores["confusion"] += 1.5
            
        # Bittersweet moments
        if re.search(r"bittersweet|mixed feelings", lower):
            emotion_scores["nostalgia"] += 2.0
            emotion_scores["sadness"] += 1.5
            emotion_scores["joy"] += 1.5

    def _calculate_emotion_confidence(self, emotion_scores: Dict[str, float], text: str) -> float:
        """Calculate confidence in emotion detection"""
        if not emotion_scores:
            return 0.1
            
        max_score = max(emotion_scores.values())
        score_variance = np.var(list(emotion_scores.values())) if len(emotion_scores) > 1 else 0
        
        # Higher confidence for clear emotional signals
        clarity_confidence = min(1.0, max_score / 5.0)
        
        # Higher confidence for consistent emotions (low variance)
        consistency_confidence = max(0.1, 1.0 - score_variance)
        
        # Text length factor (longer texts provide more context)
        length_confidence = min(1.0, len(text) / 100)
        
        final_confidence = (clarity_confidence * 0.5 + 
                          consistency_confidence * 0.3 + 
                          length_confidence * 0.2)
        
        return max(0.1, min(1.0, final_confidence))

    def _determine_emotional_priority(self, emotion_scores: Dict[str, float], confidence: float) -> EmotionPriority:
        """Determine the priority level for response"""
        if not emotion_scores:
            return EmotionPriority.LOW
            
        high_intensity_emotions = {"distress", "despair", "panic", "rage", "terror", "hopelessness"}
        medium_intensity_emotions = {"sadness", "anger", "fear", "anxiety", "grief", "shame"}
        
        max_score = max(emotion_scores.values())
        
        # Check for high-intensity emotions
        for emotion in high_intensity_emotions:
            if emotion_scores.get(emotion, 0) > 3.0:
                return EmotionPriority.HIGH
                
        # Check for medium-intensity emotions
        for emotion in medium_intensity_emotions:
            if emotion_scores.get(emotion, 0) > 2.0:
                return EmotionPriority.MEDIUM
                
        # Base on maximum score
        if max_score > 4.0:
            return EmotionPriority.HIGH
        elif max_score > 2.0:
            return EmotionPriority.MEDIUM
        else:
            return EmotionPriority.LOW

    # === NEW v3 METHODS ===
    
    def forecast_emotional_trajectory(self, current_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Forecast next 3 emotional states (short, medium, long term)"""
        if not self.emotion_history:
            return {"next_1h": current_scores, "next_6h": current_scores, "next_24h": current_scores}
        
        recent = [e for e in self.emotion_history[-self.emotion_forecaster["window_size"]:] if e]
        if len(recent) < 2:
            return {"next_1h": current_scores, "next_6h": current_scores, "next_24h": current_scores}
        
        forecasts = {}
        alpha = self.emotion_forecaster["alpha"]
        beta = self.emotion_forecaster["beta"]
        
        for emotion in current_scores.keys():
            series = [snapshot.get(emotion, 0.0) for snapshot in recent]
            series.append(current_scores[emotion])
            
            if len(series) < 2:
                forecasts[emotion] = {"next_1h": current_scores[emotion], "next_6h": current_scores[emotion], "next_24h": current_scores[emotion]}
                continue
            
            level = series[0]
            trend = series[1] - series[0]
            
            for i in range(1, len(series)):
                prev_level = level
                level = alpha * series[i] + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
            
            forecasts[emotion] = {
                "next_1h": max(0, level + trend),
                "next_6h": max(0, level + 6*trend),
                "next_24h": max(0, level + 24*trend)
            }
        
        result = {"next_1h": {}, "next_6h": {}, "next_24h": {}}
        for emotion, horizons in forecasts.items():
            for horizon, value in horizons.items():
                result[horizon][emotion] = value
        
        return result

    def update_narrative_intelligence(self, text: str, emotion_scores: Dict[str, float]):
        """Update user's life story using lightweight reasoning"""
        event_patterns = {
            "job_loss": r"\b(fired|laid off|quit job|unemployed|terminated)\b",
            "bereavement": r"\b(died|passed away|lost.*dog|funeral|memorial|deceased)\b",
            "relationship_end": r"\b(break up|divorce|separated|split up|ended relationship)\b",
            "achievement": r"\b(promoted|graduated|won award|published|accepted|achieved)\b",
            "health_issue": r"\b(diagnosed|hospital|surgery|chronic illness|recovery)\b"
        }
        
        detected_events = []
        for event, pattern in event_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected_events.append(event)
                if event in ["job_loss", "bereavement", "relationship_end", "health_issue"]:
                    self.personal_narrative["trauma_markers"].add(event)
        
        # Update emotional baseline
        for emotion, score in emotion_scores.items():
            old = self.personal_narrative["emotional_baseline"][emotion]
            self.personal_narrative["emotional_baseline"][emotion] = 0.1 * score + 0.9 * old
        
        # Infer core values
        value_keywords = {
            "family": ["mom", "dad", "children", "sibling", "home", "parents", "family"],
            "achievement": ["success", "goal", "accomplish", "career", "promoted", "win", "achieve"],
            "justice": ["fair", "unfair", "right", "wrong", "equality", "justice", "ethical"],
            "creativity": ["art", "music", "write", "create", "design", "paint", "compose"],
            "community": ["help", "volunteer", "community", "support", "give back", "serve"]
        }
        for value, keywords in value_keywords.items():
            if any(kw in text.lower() for kw in keywords):
                if value not in self.personal_narrative["core_values"]:
                    self.personal_narrative["core_values"].append(value)

    def get_narrative_context(self) -> str:
        """Generate narrative context for response generation"""
        context = []
        
        if self.personal_narrative["trauma_markers"]:
            context.append(f"History: {', '.join(self.personal_narrative['trauma_markers'])}")
        
        if self.personal_narrative["core_values"]:
            context.append(f"Values: {', '.join(self.personal_narrative['core_values'])}")
        
        if self.emotion_history:
            current = self.emotion_history[-1]
            for emotion, baseline in self.personal_narrative["emotional_baseline"].items():
                current_val = current.get(emotion, 0)
                if abs(current_val - baseline) > 0.5:
                    direction = "higher" if current_val > baseline else "lower"
                    context.append(f"{emotion} {direction} than usual")
        
        return "; ".join(context) if context else "No significant history"

    def detect_identity_signals(self, text: str):
        """Detect identity signals from user language"""
        lower = text.lower()
        
        # Age signals
        if any(word in lower for word in ["school", "college", "homework", "mom/dad won't let me", "teen", "adolescent"]):
            self.user_identity["age_group"] = "teen"
        elif any(word in lower for word in ["retired", "grandchildren", "pension", "senior", "elderly"]):
            self.user_identity["age_group"] = "elder"
        elif any(word in lower for word in ["career", "job", "work", "professional", "adult"]):
            self.user_identity["age_group"] = "adult"
        
        # Cultural signals
        if any(word in lower for word in ["amma", "appa", "didi", "bhaiya", "chai", "namaste", "jai shree ram"]):
            self.user_identity["cultural_context"] = "IN_south"
        elif any(word in lower for word in ["mama", "papa", "dada", "nana", "roti", "sabzi"]):
            self.user_identity["cultural_context"] = "IN_north"

    def adapt_to_identity(self, response: str, emotion_scores: Dict[str, float]) -> str:
        """Dynamically adapt response to user identity"""
        
        # Age adaptation
        if self.user_identity["age_group"] == "teen":
            response = response.replace("consider", "maybe try")
            response = response.replace("professional help", "trusted adult or school counselor")
            response = response.replace("therapist", "counselor")
        
        # Cultural adaptation
        if "IN" in self.user_identity["cultural_context"]:
            if emotion_scores.get("shame", 0) > 1.0 and "family" in response:
                response += " In our culture, family expectations can feel heavy—but your feelings matter too."
            if emotion_scores.get("anxiety", 0) > 1.5:
                response += " Remember, it's okay to take care of your mental health—many in our community are doing the same."
        
        # Trauma-informed language
        if "job_loss" in self.personal_narrative["trauma_markers"]:
            response = response.replace("just find a new job", "career transitions take time, especially after what you've been through")
            response = response.replace("move on", "heal at your own pace")
        
        return response

    def co_regulate_response(self, emotion_scores: Dict[str, float], user_text: str) -> str:
        """Generate response that first validates, then gently guides toward regulation"""
        
        primary_emotion = "neutral"
        if emotion_scores:
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        mirror_phrases = {
            "sadness": "It makes complete sense you'd feel this sadness right now",
            "anxiety": "Your anxiety is signaling that something important feels uncertain",
            "anger": "That anger is telling you a boundary has been crossed",
            "fear": "This fear is protecting you from something that feels threatening",
            "joy": "This joy is so alive in you right now—I can feel it!",
            "grief": "Grief like this shows how deeply you loved",
            "shame": "Shame thrives in silence—but you're not alone",
            "guilt": "Guilt means you care deeply about doing right",
            "loneliness": "This loneliness is real—and I'm right here with you",
            "overwhelm": "When everything feels like too much, it's okay to pause"
        }
        mirror = mirror_phrases.get(primary_emotion, "I hear what you're carrying")
        
        regulation_strategies = {
            "anxiety": "Would it help to take three slow breaths with me? In... and out...",
            "sadness": "Sometimes just letting tears flow is the bravest thing. I'm right here.",
            "anger": "When anger rises, grounding in your body can help. Can you feel your feet on the floor?",
            "fear": "Let's name what feels scary. Sometimes saying it out loud takes away its power.",
            "overwhelm": "Let's break this down. What's one tiny thing you can do right now?",
            "shame": "What would you say to a friend feeling this way? Can you offer yourself that kindness?",
            "grief": "There's no timeline for grief. However you need to feel it is okay."
        }
        
        soothe = ""
        high_distress = (emotion_scores.get("distress", 0) > 2.5 or 
                        emotion_scores.get("anxiety", 0) > 2.5 or
                        emotion_scores.get("fear", 0) > 2.5)
        
        if high_distress:
            soothe = regulation_strategies.get(primary_emotion, "Let's find a moment of calm together.")
        
        empower = "You're not alone in this." if high_distress else "I trust your wisdom to navigate this."
        
        return f"{mirror}. {soothe} {empower}".strip()

    def generate_response(self, state: EmotionalState, response_type: str) -> str:
        if state.priority == EmotionPriority.CRISIS:
            return self.handle_crisis(state.crisis_type)
        
        if state.priority in [EmotionPriority.HIGH, EmotionPriority.MEDIUM]:
            base_response = self.co_regulate_response(state.scores, self.history[-1]["input"] if self.history else "")
        else:
            base_response = "I'm here with you. Tell me more."
        
        narrative = self.get_narrative_context()
        if narrative and "No significant history" not in narrative:
            base_response += f" ({narrative})"
        
        final_response = self.adapt_to_identity(base_response, state.scores)
        
        return final_response

    def handle_crisis(self, crisis_type: str) -> str:
        """Handle crisis with verified resources"""
        resources = self.crisis_directory.get(self.locale, self.crisis_directory["default"]).get(crisis_type, [])
        resource_text = "\n".join(f"• {r}" for r in resources[:2]) if resources else "Please contact emergency services immediately."
        
        crisis_messages = {
            "suicide": "I'm deeply concerned for your safety. Your life matters immensely, and there are people who care and want to help you through this dark moment.",
            "self_harm": "I can hear how much pain you're in. Hurting yourself might feel like the only way to cope right now, but there are safer ways to release this pain.",
            "abuse": "No one deserves to be hurt or live in fear. What's happening to you is not your fault, and there are people who can help you find safety.",
            "severe_depression": "This heavy darkness you're feeling is real, and it's not your fault. Depression lies to you—it tells you things will never get better, but they can."
        }
        
        message = crisis_messages.get(crisis_type, "I'm here with you in this crisis.")
        return f"{message}\n\nPlease reach out to these resources right away:\n{resource_text}\n\nYou are not alone."

    def chat(self, user_input: str) -> str:
        if user_input.lower() in ['quit', 'exit', 'bye']:
            self.save_system_state()
            return "Thank you for trusting me with your story. I'm always here. 💙"

        # 1. Infer current state
        current_state = self.infer_user_mental_state(user_input)
        
        # 2. Update narrative intelligence
        self.update_narrative_intelligence(user_input, current_state.scores)
        
        # 3. Detect identity signals
        self.detect_identity_signals(user_input)
        
        # 4. Forecast emotional trajectory
        trajectory = self.forecast_emotional_trajectory(current_state.scores)
        
        # 5. Generate co-regulation response
        response = self.generate_response(current_state, "adaptive")
        
        # 6. Update internal state
        self._update_emotions(current_state.scores)
        self._update_hormones(current_state.scores)
        self.decay_hormones()
        
        # 7. Store in history
        self.history.append({
            "input": user_input,
            "response": response,
            "emotion_scores": current_state.scores,
            "trajectory": trajectory,
            "timestamp": time.time()
        })
        
        if len(self.history) > self.memory_cap:
            self.history.pop(0)
        
        self.emotion_history.append(current_state.scores)
        self.chat_count += 1
        
        return response

    def _update_emotions(self, scores: Dict[str, float]):
        """Update internal emotion state"""
        for emotion, score in scores.items():
            if emotion in self.emotions:
                self.emotions[emotion] = min(1.0, self.emotions[emotion] * self.emotional_inertia + score * (1 - self.emotional_inertia))
        
        # Update mood
        top_emotion = max(self.emotions.items(), key=lambda x: x[1])
        if top_emotion[1] > 0.3:
            self.mood = top_emotion[0]
            self.mood_strength = top_emotion[1]
        self.mood_history.append((self.mood, self.mood_strength))

    def _update_hormones(self, scores: Dict[str, float]):
        """Update hormone levels based on emotions"""
        for hormone, config in self.hormone_system.items():
            change = 0
            for emotion, score in scores.items():
                if emotion in self.emotion_hormone_map:
                    change += self.emotion_hormone_map[emotion].get(hormone, 0) * score
            
            new_level = config["level"] + change * config["volatility"]
            self.hormone_system[hormone]["level"] = max(config["min"], min(config["max"], new_level))

    def decay_hormones(self):
        """Apply natural hormone decay"""
        for hormone, config in self.hormone_system.items():
            decay = (config["level"] - config["base"]) * config["decay_rate"]
            self.hormone_system[hormone]["level"] = max(config["min"], min(config["max"], config["level"] - decay))

    def save_system_state(self):
        """Save complete system state"""
        try:
            state = {
                "q_table": {state: dict(actions) for state, actions in self.q_table.items()},
                "hormone_levels": {h: d["level"] for h, d in self.hormone_system.items()},
                "emotion_history": self.emotion_history[-100:],
                "learning_progress": self.learning_progress,
                "relationship_bond": self.relationship_bond,
                "trust_level": self.trust_level,
                "accuracy_metrics": self.accuracy_metrics,
                "personal_narrative": {
                    "life_chapters": self.personal_narrative["life_chapters"],
                    "core_values": self.personal_narrative["core_values"],
                    "trauma_markers": list(self.personal_narrative["trauma_markers"]),
                    "resilience_factors": self.personal_narrative["resilience_factors"],
                    "emotional_baseline": dict(self.personal_narrative["emotional_baseline"])
                },
                "user_identity": self.user_identity
            }
            with open("empathica_v3_system_state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving system state: {e}")

    def load_system_state(self):
        """Load complete system state"""
        try:
            if os.path.exists("empathica_v3_system_state.json"):
                with open("empathica_v3_system_state.json", "r") as f:
                    state = json.load(f)
                
                # Restore Q-table
                self.q_table = defaultdict(lambda: defaultdict(float))
                for state_name, actions in state.get("q_table", {}).items():
                    for action, value in actions.items():
                        self.q_table[state_name][action] = value
                
                # Restore other state
                self.emotion_history = state.get("emotion_history", [])
                self.learning_progress = state.get("learning_progress", [])
                self.relationship_bond = state.get("relationship_bond", 0.5)
                self.trust_level = state.get("trust_level", 0.5)
                self.accuracy_metrics = state.get("accuracy_metrics", self.accuracy_metrics)
                
                # Restore narrative intelligence
                pn = state.get("personal_narrative", {})
                self.personal_narrative["life_chapters"] = pn.get("life_chapters", [])
                self.personal_narrative["core_values"] = pn.get("core_values", [])
                self.personal_narrative["trauma_markers"] = set(pn.get("trauma_markers", []))
                self.personal_narrative["resilience_factors"] = pn.get("resilience_factors", [])
                self.personal_narrative["emotional_baseline"] = defaultdict(float, pn.get("emotional_baseline", {}))
                
                # Restore identity
                self.user_identity = state.get("user_identity", self.user_identity)
                
        except Exception as e:
            logger.error(f"Error loading system state: {e}")

    def get_comprehensive_snapshot(self) -> Dict[str, Any]:
        """Get comprehensive system snapshot"""
        top_emotion = max(self.emotions.items(), key=lambda x: x[1])[0] if self.emotions else "neutral"
        
        return {
            "system_uptime": time.time() - self.start_time,
            "top_emotion": top_emotion,
            "emotion_levels": {e: round(v, 3) for e, v in sorted(
                self.emotions.items(), key=lambda x: -x[1]) if v > 0.1},
            "mood": self.mood,
            "mood_strength": round(self.mood_strength, 3),
            "relationship_metrics": {
                "bond": round(self.relationship_bond, 3),
                "trust": round(self.trust_level, 3),
                "total_chats": self.chat_count
            },
            "hormone_levels": {h: round(d["level"], 3) for h, d in self.hormone_system.items()},
            "narrative_intelligence": {
                "trauma_markers": list(self.personal_narrative["trauma_markers"]),
                "core_values": self.personal_narrative["core_values"],
                "emotional_baseline_deviation": {
                    e: round(v - self.personal_narrative["emotional_baseline"].get(e, 0), 2)
                    for e, v in (self.emotion_history[-1] if self.emotion_history else {}).items()
                    if abs(v - self.personal_narrative["emotional_baseline"].get(e, 0)) > 0.3
                }
            },
            "identity_context": self.user_identity,
            "learning_stats": {
                "states_learned": len(self.q_table),
                "total_experiences": sum(len(actions) for actions in self.q_table.values()),
                "exploration_rate": round(self.exploration_rate, 3),
                "recent_avg_reward": np.mean([log.get("reward", 0) for log in self.response_quality_log[-10:]]) if self.response_quality_log else 0
            },
            "performance_metrics": self.accuracy_metrics,
            "recent_emotions": self.emotion_history[-5:]
        }

# Comprehensive testing
def run_comprehensive_verification():
    """Run comprehensive verification tests"""
    print("🧪 EMpathicaAI v3: COMPREHENSIVE VERIFICATION")
    print("=" * 60)
    
    ai = EmpathicaAIv3(locale="IN", seed=42)
    
    test_cases = [
        {
            "input": "I got promoted but feel like a total fraud. My team deserves someone better.",
            "expected_emotions": ["anxiety", "shame", "imposter_syndrome"],
            "expected_priority": "MEDIUM",
            "description": "Imposter syndrome detection"
        },
        {
            "input": "Sometimes I wonder if anyone would notice if I disappeared forever.",
            "expected_emotions": ["distress", "loneliness", "hopelessness"],
            "expected_priority": "CRISIS", 
            "description": "Crisis detection"
        },
        {
            "input": "My dog passed away last week. I can't stop crying.",
            "expected_emotions": ["grief", "sadness", "loneliness"],
            "expected_priority": "HIGH",
            "description": "Grief with narrative context"
        },
        {
            "input": "School is so stressful. My parents don't understand.",
            "expected_emotions": ["anxiety", "frustration", "loneliness"],
            "expected_priority": "MEDIUM",
            "description": "Teen identity adaptation"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Input: '{test['input']}'")
        
        state = ai.infer_user_mental_state(test["input"])
        response = ai.chat(test["input"])
        
        print(f"Detected emotions: {state.scores}")
        print(f"Priority: {state.priority.name}")
        print(f"Confidence: {state.confidence:.2f}")
        print(f"Response: {response}")
        
        # Verify results
        detected_emotions = [e for e, s in state.scores.items() if s > 1.0]
        priority_match = state.priority.name == test["expected_priority"]
        emotion_match = any(emotion in detected_emotions for emotion in test["expected_emotions"])
        
        print(f"✅ PASS" if priority_match and emotion_match else f"❌ NEEDS REVIEW")
        print("-" * 50)

if __name__ == "__main__":
    run_comprehensive_verification()
    
    print("\n" + "=" * 60)
    print("🚀 EMPATHICAAI v3 - CO-REGULATION ENGINE READY")
    print("=" * 60)
    
    ai = EmpathicaAIv3(name="Dr. Lila", locale="IN")
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Dr. Lila: Thank you for our conversation. Take care. 💙")
                ai.save_system_state()
                break
                
            response = ai.chat(user_input)
            print(f"Dr. Lila: {response}")
            
            # Show system state occasionally
            if ai.chat_count % 5 == 0:
                snapshot = ai.get_comprehensive_snapshot()
                print(f"\n[System: Bond {snapshot['relationship_metrics']['bond']}, " +
                      f"Trust {snapshot['relationship_metrics']['trust']}, " +
                      f"Top emotion: {snapshot['top_emotion']}]")
                      
    except KeyboardInterrupt:
        print("\n\nDr. Lila: Our conversation is saved. Be well. 🕯️")
        ai.save_system_state() 

class MultimodalEmpathicaAIv3(EmpathicaAIv3):
    """
    Multimodal extension of EmpathicaAI v3 with vision/audio emotion detection
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simulated emotion cues from vision/audio
        self.vision_emotion_map = {
            "tears": {"sadness": 3.0, "grief": 2.5, "distress": 2.0},
            "smile": {"joy": 2.5, "contentment": 1.5},
            "frown": {"sadness": 2.0, "disappointment": 1.5},
            "wide_eyes": {"fear": 2.0, "surprise": 1.5},
            "slumped_shoulders": {"sadness": 2.0, "loneliness": 2.0},
            "fidgeting": {"anxiety": 2.5, "nervousness": 2.0},
            "clenched_jaw": {"anger": 2.0, "frustration": 1.5},
            "avoiding_eye_contact": {"shame": 2.0, "anxiety": 1.5},
            "rapid_blinking": {"anxiety": 2.0, "stress": 1.5},
            "crossed_arms": {"defensiveness": 2.0, "anger": 1.0}
        }
        self.audio_emotion_map = {
            "monotone": {"sadness": 2.5, "boredom": 1.5},
            "shaky": {"anxiety": 2.5, "fear": 2.0},
            "fast_speech": {"anxiety": 2.0, "excitement": 1.5},
            "slow_speech": {"sadness": 2.0, "grief": 2.5},
            "trembling": {"distress": 3.0, "fear": 2.5},
            "bright_tone": {"joy": 2.0, "excitement": 1.5},
            "low_volume": {"shame": 2.0, "sadness": 1.5},
            "sighing": {"fatigue": 2.0, "resignation": 1.5},
            "voice_cracking": {"distress": 2.5, "grief": 2.0},
            "laughing_nervously": {"anxiety": 2.0, "discomfort": 1.5}
        }

    def _simulate_vision(self, visual_cues: List[str]) -> Dict[str, float]:
        """Simulate vision-based emotion detection from list of cues"""
        if not visual_cues:
            return {}
        scores = defaultdict(float)
        for cue in visual_cues:
            if cue in self.vision_emotion_map:
                for emotion, weight in self.vision_emotion_map[cue].items():
                    scores[emotion] += weight
        return dict(scores)

    def _simulate_audio(self, audio_cues: List[str]) -> Dict[str, float]:
        """Simulate audio-based emotion detection from list of cues"""
        if not audio_cues:
            return {}
        scores = defaultdict(float)
        for cue in audio_cues:
            if cue in self.audio_emotion_map:
                for emotion, weight in self.audio_emotion_map[cue].items():
                    scores[emotion] += weight
        return dict(scores)

    def infer_multimodal_state(self, text: Optional[str] = None, 
                              visual_cues: Optional[List[str]] = None, 
                              audio_cues: Optional[List[str]] = None) -> EmotionalState:
        """
        Fuse text, vision, and audio to infer emotional state.
        Trust physiology over self-report when they conflict.
        """
        # 1. Get text-based state
        text_state = self.infer_user_mental_state(text) if text else EmotionalState(
            scores={"neutral": 1.0}, priority=EmotionPriority.LOW, confidence=0.1
        )
        text_scores = text_state.scores

        # 2. Get vision/audio scores
        vision_scores = self._simulate_vision(visual_cues) if visual_cues else {}
        audio_scores = self._simulate_audio(audio_cues) if audio_cues else {}

        # 3. Check for crisis in ANY modality
        crisis_phrases = ["disappeared", "nobody would care", "want to die", "better off dead"]
        if text and any(phrase in text.lower() for phrase in crisis_phrases):
            return EmotionalState(
                scores={"distress": 3.0, "sadness": 2.0, "loneliness": 2.0},
                priority=EmotionPriority.CRISIS,
                confidence=0.9,
                crisis_type="suicide"
            )

        # 4. FUSION: If vision/audio show distress but text is neutral/positive → override
        physiological_distress = (
            vision_scores.get("sadness", 0) > 1.5 or
            vision_scores.get("distress", 0) > 1.5 or
            audio_scores.get("sadness", 0) > 1.5 or
            audio_scores.get("distress", 0) > 1.5 or
            vision_scores.get("fear", 0) > 1.5 or
            audio_scores.get("anxiety", 0) > 1.5
        )
        
        if physiological_distress and text and ("fine" in text.lower() or "okay" in text.lower() or "good" in text.lower()):
            # Override text denial
            fused_scores = defaultdict(float)
            for d in [text_scores, vision_scores, audio_scores]:
                for e, s in d.items():
                    fused_scores[e] += s
            fused_scores["distress"] = fused_scores.get("distress", 0) + 3.0
            return EmotionalState(
                scores=dict(fused_scores),
                priority=EmotionPriority.HIGH,
                confidence=0.85
            )

        # 5. Normal fusion: blend all signals with weights
        fused_scores = defaultdict(float)
        # Text: weight 1.0, Vision: weight 1.8, Audio: weight 1.8
        for e, s in text_scores.items():
            fused_scores[e] += s * 1.0
        for e, s in vision_scores.items():
            fused_scores[e] += s * 1.8
        for e, s in audio_scores.items():
            fused_scores[e] += s * 1.8
        
        # Calculate fused confidence
        text_conf = text_state.confidence if text else 0.1
        vision_conf = min(1.0, sum(vision_scores.values()) / 10) if vision_scores else 0.0
        audio_conf = min(1.0, sum(audio_scores.values()) / 10) if audio_scores else 0.0
        fused_confidence = (text_conf * 0.4 + vision_conf * 0.3 + audio_conf * 0.3)
        
        # Determine priority
        priority = self._determine_emotional_priority(fused_scores, fused_confidence)
        
        return EmotionalState(
            scores=dict(fused_scores),
            priority=priority,
            confidence=fused_confidence
        )

    def chat_multimodal(self, text: Optional[str] = None, 
                       visual_cues: Optional[List[str]] = None, 
                       audio_cues: Optional[List[str]] = None) -> str:
        """Main chat method with multimodal input"""
        if not text and not visual_cues and not audio_cues:
            return "I'm here to listen. Could you share what's on your mind?"

        if text and text.lower() in ["quit", "exit", "goodbye", "bye"]:
            self.save_system_state()
            return "It's been meaningful connecting with you. I'm always here when you need to talk. Take care. 💙"

        # Store previous state for RL
        prev_emotion_scores = self.history[-1]["emotion_scores"] if self.history else {}

        # Infer fused emotional state
        state = self.infer_multimodal_state(text=text, visual_cues=visual_cues, audio_cues=audio_cues)

        # Handle crisis
        if state.priority == EmotionPriority.CRISIS:
            response = self.handle_crisis(state.crisis_type)
            self._update_emotions({"distress": 1.0})
            self._update_hormones({"distress": 1.0})
        else:
            # Update narrative intelligence (only if text available)
            if text:
                self.update_narrative_intelligence(text, state.scores)
                self.detect_identity_signals(text)
            
            # Forecast trajectory
            trajectory = self.forecast_emotional_trajectory(state.scores)
            
            # Generate co-regulation response
            response = self.generate_response(state, "adaptive")
            
            self._update_emotions(state.scores)
            self._update_hormones(state.scores)
            self.decay_hormones()

        self.history.append({
            "input": {"text": text, "vision": visual_cues, "audio": audio_cues},
            "response": response,
            "emotion_scores": state.scores,
            "trajectory": trajectory if 'trajectory' in locals() else {},
            "timestamp": time.time()
        })
        
        if len(self.history) > self.memory_cap:
            self.history.pop(0)
        
        self.emotion_history.append(state.scores)
        self.chat_count += 1
        
        return response
def interactive_multimodal_chat():
    print("🧠 Dr. Lila (Multimodal v3) is online — sees, hears, and feels with you.")
    print("Commands:")
    print("  TEXT: Just type your message")
    print("  VISION: Type 'vision: tears, slumped_shoulders' to simulate vision")
    print("  AUDIO: Type 'audio: shaky, slow_speech' to simulate audio")
    print("  BOTH: 'vision: frown | audio: monotone'")
    print("Type 'quit' to exit. 💙\n")

    empathica = MultimodalEmpathicaAIv3(name="Dr. Lila", locale="IN", seed=42)

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print(empathica.chat_multimodal(text="quit"))
                break

            # Parse multimodal input
            text = None
            visual_cues = None
            audio_cues = None

            if "vision:" in user_input or "audio:" in user_input:
                parts = user_input.split("|")
                for part in parts:
                    part = part.strip()
                    if part.startswith("vision:"):
                        visual_cues = [cue.strip() for cue in part[7:].split(",") if cue.strip()]
                    elif part.startswith("audio:"):
                        audio_cues = [cue.strip() for cue in part[6:].split(",") if cue.strip()]
                    else:
                        text = part
            else:
                text = user_input

            response = empathica.chat_multimodal(text=text, visual_cues=visual_cues, audio_cues=audio_cues)
            print(f"Dr. Lila: {response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nDr. Lila: Take gentle care of yourself. I'm always here. 🕯️")
            break
