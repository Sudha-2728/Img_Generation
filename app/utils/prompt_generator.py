import requests

OLLAMA_URL = "http://localhost:11434"


def get_ollama_model_name():
    """Fetch the first available model from the local Ollama server."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=20)
        response.raise_for_status()
        data = response.json()

        models = data.get("models", [])
        if not models:
            raise ValueError("No Ollama models were found. Please check 'ollama list'.")

        model_names = [model.get("name") for model in models if model.get("name")]
        if not model_names:
            raise ValueError("No model names were returned by Ollama.")

        return model_names[0]

    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Please make sure Ollama is running on your computer."
        ) from exc


def build_image_prompt(user_idea, style, mood, lighting, color_style, additional_requirements=""):
    """Create a clear and detailed prompt for the image-generation model."""
    extra = additional_requirements.strip()

    full_prompt = (
        "You are a prompt optimizer for AI image generation. "
        "Create a single, highly descriptive image prompt from the following inputs. "
        "Do not explain the process. Only return the final prompt.\n\n"
        f"User idea: {user_idea}\n"
        f"Image style: {style}\n"
        f"Mood: {mood}\n"
        f"Lighting: {lighting}\n"
        f"Color style: {color_style}\n"
        f"Additional requirements: {extra if extra else 'None'}\n\n"
        "Include the main subject, character or object details, environment, background, composition, "
        "camera angle, textures, lighting mood, and overall visual quality. Keep it concise but rich in detail."
    )

    return full_prompt


def generate_image_prompt(user_idea, style, mood, lighting, color_style, additional_requirements=""):
    """Send the prompt request to Ollama and return the generated prompt text."""
    model_name = get_ollama_model_name()
    prompt = build_image_prompt(
        user_idea=user_idea,
        style=style,
        mood=mood,
        lighting=lighting,
        color_style=color_style,
        additional_requirements=additional_requirements,
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()
    return result.get("response", "").strip()
