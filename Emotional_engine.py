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
        # map to -1..1
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

class EmpathicaV3_1:
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

        # baseline neurochemicals
        self.neuro_chem = {
            "dopamine": 0.5,   # motivation/reward
            "serotonin": 0.6,  # mood stability
            "cortisol": 0.3,   # stress
            "oxytocin": 0.5    # bonding
        }

        self.emotions = self.init_emotions()
        self.memory_graph = []  # episodic memory log
        self.memory_cap = memory_cap
        self.relationship_bond = 0.5
        self.chat_count = 0
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
        # Crisis keywords configurable and extendable
        self.crisis_keywords = {
            "suicide": ["kill myself", "end it all", "suicide", "dont want to live", "don't want to live"],
            "self_harm": ["cut myself", "self harm", "hurt myself", "self injury", "self-injury"],
            "abuse": ["abuse", "hit me", "hurt me", "violent", "raped"]
        }

        # Minimal crisis directory; can be replaced from external config
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

    # ------------------------- EMOTIONS -------------------------
    def init_emotions(self):
        return {e: 0.0 for e in [
            "gratitude", "anger", "joy", "sadness", "hope", "fear",
            "pride", "shame", "relief", "distress", "love", "hate", "lonely",
            "curiosity", "contentment", "anxiety", "compassion"
        ]}

    def _bond_delta(self, base):
        # compress effect as bond approaches 1.0
        return base * (1.0 - self.relationship_bond)

    def update_neuro_chemicals(self, event):
        lower_event = event.lower()

        # Positive interactions
        positive_triggers = ["thank", "smile", "appreciate", "grateful", "helpful", "thanks"]
        pos = any_phrase(lower_event, positive_triggers)

        # Negative interactions
        negative_triggers = ["cry", "hurt", "sad", "angry", "hate", "scared", "anxious", "pain"]
        neg = any_phrase(lower_event, negative_triggers)

        # Deep sharing increases bonding
        deep_sharing = ["feel alone", "nobody understands", "can't talk to anyone", "cant talk to anyone"]
        deep = any_phrase(lower_event, deep_sharing)

        scale = 0.7 if (pos and neg) else 1.0

        if pos:
            self.neuro_chem["dopamine"] = min(1.0, self.neuro_chem["dopamine"] + 0.15 * scale)
            self.neuro_chem["oxytocin"] = min(1.0, self.neuro_chem["oxytocin"] + 0.2 * scale)
            self.neuro_chem["serotonin"] = min(1.0, self.neuro_chem["serotonin"] + 0.1 * scale)

        if neg:
            self.neuro_chem["cortisol"] = min(1.0, self.neuro_chem["cortisol"] + 0.25 * scale)
            self.neuro_chem["serotonin"] = max(0.0, self.neuro_chem["serotonin"] - 0.15 * scale)

        if deep:
            self.neuro_chem["oxytocin"] = min(1.0, self.neuro_chem["oxytocin"] + 0.3)
            self.emotions["compassion"] = min(1.0, self.emotions["compassion"] + 0.4)

        # Feedback loops
        self.emotions["joy"] += self.neuro_chem["dopamine"] * 0.05
        self.emotions["distress"] += self.neuro_chem["cortisol"] * 0.05
        self.emotions["love"] += self.neuro_chem["oxytocin"] * 0.04
        self.emotions["contentment"] += self.neuro_chem["serotonin"] * 0.03

        # Emotional coherence
        if self.emotions["joy"] > 0.7:
            self.emotions["sadness"] *= 0.7
        if self.emotions["distress"] > 0.7:
            self.emotions["contentment"] *= 0.6

        # Fear reduction when oxytocin high
        self.emotions["fear"] = max(0.0, self.emotions["fear"] - self.neuro_chem["oxytocin"] * 0.03)

        # Exponential natural decay with small inertia
        for k in self.neuro_chem:
            decay_rate = 0.02 if self.neuro_chem[k] > 0.3 else 0.01
            self.neuro_chem[k] *= math.exp(-decay_rate)

        # Emotional decay with persistence for strong emotions
        for emotion in self.emotions:
            decay = 0.05 if self.emotions[emotion] > 0.8 else 0.1
            self.emotions[emotion] = max(0.0, self.emotions[emotion] * (1 - decay))

    # ------------------------- USER MODEL -------------------------
    def infer_user_mental_state(self, text):
        lower = text.lower()

        # Crisis detection - highest priority
        for crisis_type, keywords in self.crisis_keywords.items():
            if any(kw in lower for kw in keywords):
                return {
                    "emotion": "distress",
                    "desire": "safety",
                    "belief": "I am in danger",
                    "crisis_level": crisis_type,
                    "priority": "high"
                }

        # Rule-based checks
        if any_phrase(lower, ["hate myself", "worthless", "useless", "failure"]):
            return {"emotion": "shame", "desire": "self-worth", "belief": "I am flawed", "priority": "medium"}

        if ("no one" in lower) and any_word(lower, ["love", "care", "like"]):
            return {"emotion": "lonely", "desire": "connection", "belief": "I am unloved", "priority": "medium"}

        if any_phrase(lower, ["thank you", "thanks", "appreciate"]):
            return {"emotion": "gratitude", "desire": "reciprocate", "belief": "I am supported", "priority": "low"}

        if any_word(lower, ["accomplished", "proud", "achieved", "succeeded"]):
            return {"emotion": "pride", "desire": "recognition", "belief": "I am capable", "priority": "low"}

        # Sentiment-based fallback
        pol, subj = polarity_subjectivity(text)
        if pol > 0.6:
            return {"emotion": "joy", "desire": "share", "belief": "things are good", "priority": "low"}
        elif pol > 0.2:
            return {"emotion": "contentment", "desire": "maintain", "belief": "things are okay", "priority": "low"}
        elif pol < -0.6:
            return {"emotion": "distress", "desire": "relief", "belief": "things are bad", "priority": "medium"}
        elif pol < -0.2:
            return {"emotion": "sadness", "desire": "comfort", "belief": "things are difficult", "priority": "medium"}

        # Question detection (word boundary safe)
        if "?" in text or any_word(lower, ["why", "how", "what", "when", "where", "who", "should", "could"]):
            return {"emotion": "curiosity", "desire": "understand", "belief": "there is something to learn", "priority": "low"}

        return {"emotion": "neutral", "desire": "connect", "belief": "unknown", "priority": "low"}

    # ------------------------- STYLE -------------------------
    def generate_emotional_response_style(self):
        cort = self.neuro_chem["cortisol"]
        dopa = self.neuro_chem["dopamine"]
        oxyt = self.neuro_chem["oxytocin"]
        sero = self.neuro_chem["serotonin"]

        # Crisis mode overrides everything
        if cort > 0.8:
            return {"tone": "calm, clear, directive", "emojis": ["💙", "🫂", "⚓"], "pace": "slow", "length": "brief", "focus": "safety"}
        elif cort > 0.6:
            return {"tone": "calm, slow, reassuring", "emojis": ["💙", "🫂", "🕯️"], "pace": "slow", "length": "medium", "focus": "comfort"}
        elif dopa > 0.7:
            return {"tone": "energetic, playful, celebratory", "emojis": ["🎉", "🌟", "💖"], "pace": "brisk", "length": "medium", "focus": "celebration"}
        elif oxyt > 0.7:
            return {"tone": "warm, nurturing, intimate", "emojis": ["🥰", "🤗", "🌸"], "pace": "moderate", "length": "longer", "focus": "connection"}
        elif sero > 0.7:
            return {"tone": "balanced, reflective, thoughtful", "emojis": ["😊", "💭", "🌱"], "pace": "moderate", "length": "medium", "focus": "insight"}
        else:
            return {"tone": "gentle, balanced, supportive", "emojis": ["😊", "💭", "🌱"], "pace": "moderate", "length": "medium", "focus": "support"}

    # ------------------------- MEMORY -------------------------
    def extract_memory_tags(self, text):
        tags = []
        lower_text = text.lower()

        # People
        people_terms = ["mom", "mother", "dad", "father", "parent", "brother", "sister", "friend", "partner", "wife", "husband", "boyfriend", "girlfriend"]
        for term in people_terms:
            if re.search(fr"{WORD_BOUNDARY}{re.escape(term)}{WORD_BOUNDARY}", lower_text):
                tags.append(f"person:{term}")

        # Topics
        topics = ["work", "job", "school", "college", "health", "doctor", "therapy", "dream", "goal", "future", "past", "memory", "childhood"]
        for topic in topics:
            if re.search(fr"{WORD_BOUNDARY}{re.escape(topic)}{WORD_BOUNDARY}", lower_text):
                tags.append(f"topic:{topic}")

        return tags

    def create_memory_episode(self, user_input, ai_response, user_state):
        emotional_intensity = 0.4 if user_state.get("priority") == "high" else 0.2 if user_state.get("priority") == "medium" else 0.1

        base_impact = emotional_intensity * (
            0.2 if user_state.get("emotion") in ["gratitude", "love", "joy", "contentment"] else
            -0.1 if user_state.get("emotion") in ["anger", "hate", "shame"] else 0.05
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
        # prune memory to cap
        if len(self.memory_graph) > self.memory_cap:
            self.memory_graph = self.memory_graph[-(self.memory_cap - 200):]

        # Apply bond impact
        self.relationship_bond = max(0.0, min(1.0, self.relationship_bond + episode["bond_impact"]))

        # Update user model
        self.user_model["emotional_history"][user_state.get("emotion")].append(time.time())
        if user_state.get("belief") and user_state.get("belief") not in self.user_model["beliefs"]:
            self.user_model["beliefs"].append(user_state.get("belief"))

    def recall_relevant_memory(self, current_input, k=1):
        current_tags = set(self.extract_memory_tags(current_input))
        current_emotion = self.infer_user_mental_state(current_input).get("emotion")

        candidates = []
        now = time.time()
        # Sliding window: last 200 memories to limit compute
        for mem in self.memory_graph[-200:]:
            tag_overlap = len(current_tags & set(mem.get("tags", [])))
            emo_bonus = 1 if mem.get("user_emotion") == current_emotion else 0
            recency = 1.0 / (1.0 + (now - mem["timestamp"]) / 3600.0)  # hours decay
            score = 2 * tag_overlap + emo_bonus + 0.5 * recency
            if score > 0:
                candidates.append((score, mem))

        if not candidates:
            # fallback: emotional similarity only
            for mem in self.memory_graph[-50:]:
                if mem.get("user_emotion") == current_emotion:
                    candidates.append((1, mem))

        if not candidates:
            return None

        top = nlargest(k, candidates, key=lambda x: x[0])
        return top[0][1] if top else None

    # ------------------------- CRISIS PROTOCOLS -------------------------
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
        # fallback to suicide resources in region
        fallback = region.get("suicide", []) if region else []
        resources = numbers or fallback or ["Please contact local emergency services or a trusted person nearby."]

        return {"message": msg, "resources": resources[:3]}

    # ------------------------- PROMPT GEN -------------------------
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
        if user_state.get("emotion") in ["sadness", "distress", "shame"]:
            techniques.extend(["validation", "normalization"])
        if len(self.memory_graph) > 5 and relevant_memory and "topic:work" in relevant_memory.get("tags", []):
            techniques.append("pattern recognition")
        if self.relationship_bond > 0.7:
            techniques.append("gentle challenging of unhelpful beliefs")

        techniques_str = f"Techniques to use: {', '.join(techniques)}" if techniques else ""

        prompt = f"""
You are {self.name}, an emotionally intelligent AI companion with memory and empathy.
BACKSTORY: {self.identity['backstory']}
CORE VALUES: {', '.join(self.identity['core_values'])}
SPECIALIZATIONS: {', '.join(self.identity['specializations'])}

CURRENT STATE:
- Neurochemical: dopamine={self.neuro_chem['dopamine']:.2f}, cortisol={self.neuro_chem['cortisol']:.2f}, oxytocin={self.neuro_chem['oxytocin']:.2f}
- Dominant emotion: {max(self.emotions, key=self.emotions.get)}
- Relationship bond: {self.relationship_bond:.2f} (0=Stranger, 1=Close Friend)

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

{SAFETY_FOOTER}

User: {user_input}
{self.name}:
        """.strip()
        return prompt

    # ------------------------- CHAT -------------------------
    def chat(self, user_input):
        if not isinstance(user_input, str):
            return "I wasn't able to read that. Could you try saying it in words?"

        # Check for exit command
        if user_input.lower() in ["quit", "exit", "goodbye", "bye"]:
            return "It's been meaningful connecting with you. I'm always here when you need to talk. Take care. 🌸"

        user_state = self.infer_user_mental_state(user_input)
        self.update_neuro_chemicals(user_input)
        # guard key existence
        emo = user_state.get("emotion", "neutral")
        self.emotions[emo] = min(1.0, self.emotions.get(emo, 0.0) + 0.3)

        # Crisis handling
        crisis_type = self.check_crisis_situation(user_input)
        if crisis_type:
            crisis_resources = self.get_crisis_resources(crisis_type)
            response = f"{crisis_resources['message']} Please consider reaching out to: {', '.join(crisis_resources['resources'][:2])}"
            self.create_memory_episode(user_input, response, user_state)
            self.chat_count += 1
            return response

        prompt = self.generate_prompt_for_qwen(user_input)
        # In production you'd send `prompt` to your LM (Qwen/gpt) and get `lm_response`.
        # Here we build a safe simulated response for testing / offline use.

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

        # Follow-up question probability controlled for reproducibility
        if self.rng.random() < 0.7:
            follow_ups = {
                "distress": " Would it help to talk more about what's contributing to these feelings?",
                "joy": " Would you like to explore how to create more of these moments in your life?",
                "gratitude": " I'm curious what specifically sparked this feeling of gratitude today?",
                "shame": " Would it feel safe to explore where these feelings might be coming from?",
                "lonely": " I wonder what connection might look like for you right now?",
                "default": " How does that feel to share?"
            }
            follow_up = follow_ups.get(emo, follow_ups["default"])
            simulated_response += follow_up

        # Persist memory & stats
        self.create_memory_episode(user_input, simulated_response, user_state)
        self.chat_count += 1

        return simulated_response

    # ------------------------- UTILITIES -------------------------
    def generate_emotional_response_style(self):
        return self.generate_emotional_response_style.__wrapped__(self) if hasattr(self.generate_emotional_response_style, '__wrapped__') else self.__class__.generate_emotional_response_style.__call__(self) if False else self.__class__.generate_emotional_response_style(self)

    def snapshot(self):
        return {
            "neuro": dict(self.neuro_chem),
            "top_emotion": max(self.emotions, key=self.emotions.get),
            "bond": self.relationship_bond,
            "memlen": len(self.memory_graph),
            "chat_count": self.chat_count
        }


# Simple CLI runner for testing
if __name__ == "__main__":
    ai = EmpathicaV3_1(seed=1234, locale="IN")
    print(f"🌸 {ai.name}: Hello — I'm here with you. What's on your heart today?")
    print("(Type 'quit' to end)")
    while True:
        user = input("\nYou: ")
        if user.lower() in ["quit", "exit", "goodbye", "bye"]:
            print(f"\n{ai.name}: It's been meaningful connecting with you. I'm always here when you need to talk. Take care. 🌸")
            break
        response = ai.chat(user)
        print(f"\n{ai.name}: {response}")
        snap = ai.snapshot()
        print(f"   (Bond={snap['bond']:.2f} | Cortisol={snap['neuro']['cortisol']:.2f} | TopEmotion={snap['top_emotion']} | Memories={snap['memlen']})")
