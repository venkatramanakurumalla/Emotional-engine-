# Empathica V3.1: A Context-Aware Emotional Engine

A sophisticated Python framework designed to power emotionally intelligent AI companions. This engine models a layered system of internal states, including simulated neurochemicals, dynamic emotions, and episodic memory, to generate deeply empathetic and contextually aware responses.

Empathica V3.1 is not an LLM itself, but a powerful pre-processing and state management system that creates highly detailed prompts, enabling a generative model (like Qwen or others) to produce nuanced, human-like dialogue.

## ✨ Key Features

* **Simulated Neurochemical Model**: Tracks and updates internal "neurochemicals" like dopamine, cortisol, and oxytocin to influence the AI's mood and tone in real-time.
* **Dynamic Emotional State Engine**: A system of interconnected emotions that respond to user input and internal neurochemical changes, providing the AI with a rich emotional landscape.
* **User Mental State Inference**: Analyzes user input to infer a current emotional state, underlying desires, and beliefs, which are then used to tailor the AI's response.
* **Context-Aware Memory Graph**: An episodic memory system that stores key interactions, tags them for relevance, and intelligently recalls them to build a continuous, personalized conversational history with the user.
* **Dynamic Response Styling**: Generates a detailed style guide for the connected LLM, specifying tone, pace, length, and even appropriate emojis based on the AI's and user's current emotional states.
* **Critical Crisis Protocol**: Automatically detects and responds to high-risk keywords (e.g., suicide, self-harm) by overriding all other logic to provide immediate, safety-oriented messaging and localized resources.
* **Extensible & Modular Design**: The codebase is designed to be easily integrated with any large language model and allows for independent development of its core components.

## 🧠 How It Works

The engine operates in a continuous loop for each user input:

1.  **Analyze User Input**: The `infer_user_mental_state` function analyzes the user's message for keywords and sentiment to determine their emotional state and priority level.
2.  **Update Internal State**: The `update_neuro_chemicals` function modifies the AI's internal neurochemical levels based on the interaction. This, in turn, influences the AI's emotional state.
3.  **Recall & Store Memory**: The `recall_relevant_memory` function searches the memory graph for past conversations related to the current topic. The `create_memory_episode` function then logs the entire interaction for future recall, updating the relationship bond in the process.
4.  **Generate LLM Prompt**: All the gathered context—AI's internal state, user's state, relevant memories, and a detailed response style—is compiled into a single, comprehensive prompt for an external LLM.
5.  **Simulate Response**: In this standalone script, a simulated response is generated based on the inferred user emotion. In a production environment, this is where the prompt would be sent to an LLM API to get the final response.

## 🚀 Getting Started

### Prerequisites

* Python 3.6+
* The `TextBlob` library is recommended for more advanced sentiment analysis, but the code includes a safe fallback if it's not installed.

```bash
pip install textblob
Safety and Ethical Considerations
This project prioritizes user safety with its built-in check_crisis_situation and get_crisis_resources functions. The SAFETY_FOOTER in the prompt template serves as a constant reminder for any connected LLM to adhere to strict safety guidelines. While this is a foundational step, it is not a substitute for professional mental health support.
