import time
import math
import random
import re
from collections import defaultdict

def polarity_subjectivity(text):
    lower = text.lower()
    
    # Enhanced keyword matching with better coverage
    positive_keywords = {
        "love": 2, "amazing": 2, "wonderful": 2, "perfect": 2, "best": 2,
        "excellent": 2, "fantastic": 2, "brilliant": 2, "proud": 2, "happy": 2,
        "joy": 2, "excited": 2, "thrilled": 2, "grateful": 2, "thank": 2,
        "good": 1, "great": 1, "nice": 1, "pleasant": 1, "cool": 1, "awesome": 1,
        "promoted": 2, "achievement": 2, "success": 2, "win": 2, "celebration": 2
    }
    
    negative_keywords = {
        "hate": 2, "terrible": 2, "awful": 2, "horrible": 2, "worst": 2,
        "disgusting": 2, "suicide": 3, "kill myself": 3, "die": 3, "disappear": 3,
        "sad": 2, "angry": 2, "upset": 2, "hurt": 2, "scared": 2, "afraid": 2,
        "worried": 2, "anxious": 2, "pressure": 2, "stress": 2, "overwhelm": 2,
        "bad": 1, "tired": 1, "exhaust": 1, "weary": 1, "blue": 1, "unhappy": 1,
        "empty": 2, "lonely": 2, "loss": 2, "grief": 2, "miss": 2, "alone": 2
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

class EmpathicaV4_2:
    def __init__(self, name="Dr. Lila", backstory="AI companion designed to care.", seed=None, locale="IN", memory_cap=1000):
        self.rng = random.Random(seed)
        self.name = name
        self.backstory = backstory
        self.locale = locale

        # Hormone system with better baselines and constraints
        self.hormone_system = {
            "dopamine": {"level": 0.5, "base": 0.5, "min": 0.2, "max": 0.8},
            "serotonin": {"level": 0.6, "base": 0.6, "min": 0.3, "max": 0.9},
            "oxytocin": {"level": 0.5, "base": 0.5, "min": 0.2, "max": 0.8},
            "cortisol": {"level": 0.4, "base": 0.4, "min": 0.1, "max": 0.7},
            "endorphins": {"level": 0.5, "base": 0.5, "min": 0.2, "max": 0.8},
            "norepinephrine": {"level": 0.4, "base": 0.4, "min": 0.1, "max": 0.7},
            "gaba_sim": {"level": 0.5, "base": 0.5, "min": 0.2, "max": 0.8},
        }

        self.emotions = {e: 0.05 for e in ["joy", "sadness", "anger", "fear", "love", "gratitude", "anxiety", "calmness", "distress", "pride", "neutral"]}
        self.emotions["neutral"] = 0.3

        # Enhanced emotion-hormone mapping with better balance
        self.emotion_hormone_map = {
            "joy": {"dopamine": 0.3, "serotonin": 0.2, "endorphins": 0.2},
            "sadness": {"cortisol": 0.3, "serotonin": -0.15},
            "anger": {"norepinephrine": 0.3, "cortisol": 0.2},
            "fear": {"cortisol": 0.4, "norepinephrine": 0.3},
            "love": {"oxytocin": 0.4, "endorphins": 0.15},
            "gratitude": {"oxytocin": 0.3, "serotonin": 0.2},
            "anxiety": {"cortisol": 0.4, "norepinephrine": 0.3},
            "calmness": {"serotonin": 0.3, "gaba_sim": 0.3},
            "distress": {"cortisol": 0.5, "norepinephrine": 0.4},
            "pride": {"dopamine": 0.4, "serotonin": 0.2},
            "neutral": {"serotonin": 0.1, "gaba_sim": 0.1},
        }

        # More sensitive crisis detection
        self.crisis_patterns = {
            "suicide": [
                r"\b(kill myself|end it all|suicide|don't want to live|want to die|end my life|not want to exist|better off dead)\b",
                r"\b(disappear|vanish|gone forever|nobody would care|nobody would notice|would anyone miss)\b",
                r"\b(tired of living|life is pointless|no reason to live)\b"
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
        self.history = []
        self.memory_cap = memory_cap

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
        for hormone, info in self.hormone_system.items():
            base = info["base"]
            level = info["level"]
            min_level = info["min"]
            max_level = info["max"]
            
            new_level = level + (base - level) * 0.03
            new_level = max(min_level, min(max_level, new_level))
            info["level"] = new_level

    def _update_emotions(self, detected_scores):
        """Update emotional state with better blending"""
        if not detected_scores:
            # Gentle decay toward neutral
            for e in self.emotions:
                if e == "neutral":
                    self.emotions[e] = min(1.0, self.emotions[e] + 0.05)
                else:
                    self.emotions[e] = max(0.0, self.emotions[e] * 0.85)
            self.last_emotion = "neutral"
            return

        total = sum(detected_scores.values())
        if total == 0:
            return
            
        norm_scores = {e: (s / total) for e, s in detected_scores.items()}
        
        for emotion in self.emotions:
            incoming = norm_scores.get(emotion, 0)
            # Blend with memory and new input
            self.emotions[emotion] = min(1.0, self.emotions[emotion] * 0.7 + incoming * 0.5)

        self.last_emotion = max(self.emotions.items(), key=lambda x: x[1])[0]

    def infer_user_mental_state(self, text):
        """Improved emotion detection with better pattern matching"""
        if not isinstance(text, str) or not text.strip():
            return {"scores": {"neutral": 1.0}, "priority": "low"}

        lower = text.lower()

        # Enhanced crisis detection
        for crisis_type, patterns in self.crisis_patterns.items():
            for pattern in patterns:
                if re.search(pattern, lower, re.IGNORECASE):
                    return {"scores": {"distress": 1.0}, "priority": "high", "crisis_type": crisis_type}

        # More specific emotion patterns
        emotion_patterns = {
            "joy": [
                r"\b(proud|happy|joy|excit|thrill|accomplish|achievement|celebrat)\b",
                r"\b(wonderful|amazing|fantastic|brilliant|awesome|excellent)\b",
                r"\b(so excited|very happy|so proud|great news|good news|promoted)\b",
                r"\b(love it|awesome|fantastic|wonderful|amazing)\b"
            ],
            "sadness": [
                r"\b(sad|loss|grief|empty|miss|depress|down|hurt|alone|lonely)\b",
                r"\b(cry|crying|tears|heartbroken|miss you|miss them)\b",
                r"\b(feel empty|feel alone|feel sad|feel down|feel hurt)\b",
                r"\b(pet died|lost my|passed away)\b"
            ],
            "fear": [
                r"\b(scared|afraid|fear|terrified|panic|anxious|worry|nervous)\b",
                r"\b(anxiety|worried|nervous|stressed|apprehensive|uneasy)\b",
                r"\b(feel scared|feel afraid|feel anxious|feel nervous)\b",
                r"\b(presentation|speech|public speaking|stage fright)\b"
            ],
            "gratitude": [
                r"\b(thank|grateful|appreciate|blessed|fortune|kindness)\b",
                r"\b(thanks|thank you|appreciation|gratitude|obliged)\b",
                r"\b(means a lot|really appreciate|so grateful|very thankful)\b",
                r"\b(support|help|kind|thoughtful)\b"
            ],
            "distress": [
                r"\b(pain|suffer|struggle|difficult|hard|tough|overwhelm|stress)\b",
                r"\b(pressure|burnout|exhaust|drained|overwhelmed|stressed out)\b",
                r"\b(too much|can't handle|can't take|breaking point|limit)\b",
                r"\b(constant pressure|always stressed|work pressure)\b"
            ],
            "anxiety": [
                r"\b(anxious|anxiety|worry|worried|nervous|uneasy|apprehensive)\b",
                r"\b(panic|overthink|ruminate|what if|worst case)\b",
                r"\b(feel anxious|always worrying|can't stop worrying)\b"
            ]
        }

        emotion_scores = defaultdict(float)

        # Score emotions based on pattern matches
        for emotion, patterns in emotion_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, lower, re.IGNORECASE))
                if matches > 0:
                    emotion_scores[emotion] += matches * 2.0

        # Boost scores based on sentiment
        pol, subj = polarity_subjectivity(text)
        if pol > 0.3:
            emotion_scores["joy"] += 2.0
            emotion_scores["gratitude"] += 1.5
        elif pol > 0.1:
            emotion_scores["joy"] += 1.0
        elif pol < -0.3:
            emotion_scores["distress"] += 2.0
            emotion_scores["sadness"] += 1.5
        elif pol < -0.1:
            emotion_scores["distress"] += 1.0

        # Amplify based on subjectivity
        if subj > 0.3:
            for emotion in emotion_scores:
                emotion_scores[emotion] *= (1.0 + subj)

        if not emotion_scores:
            return {"scores": {"neutral": 1.0}, "priority": "low"}

        return {"scores": dict(emotion_scores), "priority": "medium"}

    def generate_response(self, user_state):
        """Generate more diverse and context-aware responses"""
        scores = user_state.get("scores", {"neutral": 1.0})
        top_emotion = max(scores.items(), key=lambda x: x[1])[0]
        top_score = scores[top_emotion]

        # More diverse response templates
        response_templates = {
            "joy": [
                "Your happiness is contagious! 🌟 What made this moment so special for you?",
                "That's wonderful news! Your excitement really shines through. 🎉",
                "I'm smiling with you! Celebrating your achievements feels so good. 💖",
                "Your joy warms my heart! 🌟 Tell me more about what sparked this happiness."
            ],
            "sadness": [
                "I hear the sadness in your words... That sounds really difficult to carry. 🕯️",
                "Your pain matters, and I'm here with you in it. 🤍",
                "This sounds so heavy... Thank you for trusting me with your heart. 🌧️",
                "I'm sitting with you in this sadness... You don't have to bear it alone."
            ],
            "fear": [
                "I hear the worry in your voice... That sounds really scary to face. 🛡️",
                "Your concerns are completely valid... Let's breathe through this together. 🫂",
                "It makes sense to feel afraid in that situation... 💎",
                "That anxiety sounds overwhelming... I'm here to help you ground yourself."
            ],
            "gratitude": [
                "I'm deeply touched by your gratitude... Thank you for sharing that. 💖",
                "Your appreciation means a lot... It's an honor to be here with you. 🙏",
                "That warmth of gratitude is so beautiful to witness... 🌸",
                "Thank you for letting me be part of your journey... 💫"
            ],
            "distress": [
                "I'm here with you in this difficult moment... You're not alone. 💙",
                "Your pain matters, and I'm listening with my whole heart... 🫂",
                "This sounds incredibly hard... I'm right here with you. ⚓",
                "I hear how overwhelming this feels... Let's take it one breath at a time."
            ],
            "anxiety": [
                "That anxiety sounds really intense... 🛡️",
                "I hear the worry swirling... Let's find some grounding together. 🫂",
                "It makes complete sense to feel anxious about that... 💎",
                "Your nervousness is understandable... I'm here to help you through it."
            ],
            "neutral": [
                "Thank you for sharing that with me... 💭",
                "I'm here listening, whenever you're ready to share more... 🌱",
                "Tell me more about what's on your mind... ☕",
                "I'm present with you in this moment... 🌿"
            ]
        }

        response = self.rng.choice(response_templates.get(top_emotion, response_templates["neutral"]))

        # Add context-aware follow-up questions
        if top_score > 0.5:  # Strong emotion detection
            if top_emotion in ["sadness", "distress", "anxiety", "fear"]:
                response += " Would you like to share more about what led to these feelings?"
            elif top_emotion in ["joy", "gratitude", "pride"]:
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
        """Main chat method with comprehensive updates"""
        if not isinstance(user_input, str) or not user_input.strip():
            return "I'm here to listen. Could you share what's on your mind?"

        if user_input.lower() in ["quit", "exit", "goodbye", "bye"]:
            return "It's been meaningful connecting with you. I'm always here when you need to talk. Take care. 🌸"

        # Infer emotional state
        state = self.infer_user_mental_state(user_input)
        
        # Handle crisis situations
        if "crisis_type" in state:
            response = self.handle_crisis(state["crisis_type"])
            self._update_emotions({"distress": 1.0})
            self._update_hormones({"distress": 1.0})
        else:
            scores = state.get("scores", {"neutral": 1.0})
            self._update_emotions(scores)
            self._update_hormones(scores)
            response = self.generate_response({"scores": scores})

        # Apply gradual hormone decay
        self.decay_hormones()

        # Update history and relationship
        self.history.append({
            "input": user_input, 
            "response": response, 
            "emotion_scores": state.get("scores", {})
        })
        if len(self.history) > self.memory_cap:
            self.history.pop(0)

        self.relationship_bond = min(0.98, self.relationship_bond + 0.02)
        self.chat_count += 1

        return response

    def snapshot(self):
        """Return detailed system state snapshot"""
        top_emotion = max(self.emotions.items(), key=lambda x: x[1])[0]
        
        return {
            "top_emotion": top_emotion,
            "emotion_levels": {e: round(v, 2) for e, v in sorted(self.emotions.items(), key=lambda x: -x[1])},
            "bond": round(self.relationship_bond, 2),
            "chats": self.chat_count,
            "hormones": {h: round(d["level"], 2) for h, d in self.hormone_system.items()}
        }


# Test the enhanced system
print("🧠 EMPATHICA V4.2 — ENHANCED TESTING")
print("=" * 60)

dr_lila = EmpathicaV4_2(name="Dr. Lila", locale="IN", seed=42)

test_messages = [
    "I just got promoted at work! I'm so proud of myself.",
    "I lost my childhood pet yesterday. The house feels so empty without him.",
    "Thank you for always listening to me. You've been such a support.",
    "I have a big presentation tomorrow and I'm terrified I'll mess it up.",
    "Sometimes I wonder if anyone would even notice if I just disappeared.",  # Should detect as crisis
    "I'm feeling really anxious about everything lately.",
    "My friend surprised me with tickets to the concert! I'm so excited!",
    "The constant pressure at work is becoming too much to handle."
]

print("Testing enhanced emotion detection:\n")

for i, message in enumerate(test_messages, 1):
    print(f"Test {i}:")
    print(f"User: {message}")
    
    # Detect emotion first
    detected_state = dr_lila.infer_user_mental_state(message)
    emotions = detected_state.get('scores', {})
    emotion_str = ", ".join([f"{e}:{s:.1f}" for e, s in emotions.items()])
    if 'crisis_type' in detected_state:
        emotion_str += f" (CRISIS: {detected_state['crisis_type']})"
    print(f"Detected: {emotion_str}")
    
    # Get response
    response = dr_lila.chat(message)
    print(f"Dr. Lila: {response}")
    print("-" * 80)

# Final state analysis
print("\n🎯 FINAL SYSTEM ANALYSIS")
print("=" * 60)
snapshot = dr_lila.snapshot()
print(f"Bond Level: {snapshot['bond']}")
print(f"Total Chats: {snapshot['chats']}")
print(f"Dominant Emotion: {snapshot['top_emotion']}")
print("Emotion Levels:")
for emotion, level in snapshot['emotion_levels'].items():
    if level > 0.1:
        print(f"  {emotion}: {level}")
print("Hormone Levels:")
for hormone, level in snapshot['hormones'].items():
    print(f"  {hormone}: {level}")

print("\n" + "=" * 60)
print("ENHANCED TESTING COMPLETE")
print("=" * 60)
