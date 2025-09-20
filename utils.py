# utils.py
import json
import os

def load_questions(filename: str = "question_bank.json"):
    """
    Load questions from the given JSON file.
    Default: 'question_bank.json'. Pass 'aptitude_bank.json' to load aptitude questions.
    Returns a safe default structure if the file does not exist or is invalid JSON.
    """
    # Normalized filename
    filename = filename or "question_bank.json"

    if not os.path.exists(filename):
        # Return a sensible default structure depending on which file is requested
        if filename == "aptitude_bank.json":
            return {
                "stem": {"aptitude": []},
                "abm": {"aptitude": []},
                "humss": {"aptitude": []},
                "tvl": {"aptitude": []},
                "techvoc": {"aptitude": []},
                "gas": {"aptitude": []}
            }
        else:
            # default for question_bank.json (knowledge / interest / personality / goal)
            return {
                "stem": {"knowledge": [], "interest": [], "personality": [], "goal": []},
                "abm": {"knowledge": [], "interest": [], "personality": [], "goal": []},
                "humss": {"knowledge": [], "interest": [], "personality": [], "goal": []},
                "tvl": {"knowledge": [], "interest": [], "personality": [], "goal": []},
                "techvoc": {"knowledge": [], "interest": [], "personality": [], "goal": []},
                "gas": {"knowledge": [], "interest": [], "personality": [], "goal": []}
            }

    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        # Invalid JSON -> return same sensible default as above
        if filename == "aptitude_bank.json":
            return {
                "stem": {"aptitude": []},
                "abm": {"aptitude": []},
                "humss": {"aptitude": []},
                "tvl": {"aptitude": []},
                "techvoc": {"aptitude": []},
                "gas": {"aptitude": []}
            }
        else:
            return {
                "stem": {"knowledge": [], "interest": [], "personality": [], "goal": []},
                "abm": {"knowledge": [], "interest": [], "personality": [], "goal": []},
                "humss": {"knowledge": [], "interest": [], "personality": [], "goal": []},
                "tvl": {"knowledge": [], "interest": [], "personality": [], "goal": []},
                "techvoc": {"knowledge": [], "interest": [], "personality": [], "goal": []},
                "gas": {"knowledge": [], "interest": [], "personality": [], "goal": []}
            }

def save_questions(data, filename: str = "question_bank.json"):
    """
    Save questions to the given JSON file (defaults to question_bank.json).
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_training_data():
    """
    Load training_data.json if exists, otherwise return empty list.
    """
    path = "training_data.json"
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return []

def save_training_data(data):
    """
    Save training_data.json.
    """
    with open("training_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
