# Configuration for the application

from dotenv import load_dotenv

# Load environment variables from a .env file if present (e.g. OPENROUTER_API_KEY).
load_dotenv()

# OpenRouter (OpenAI-compatible) settings.
# Model ids use OpenRouter's "provider/model" slug format. See https://openrouter.ai/models
MODEL_NAME = "google/gemini-3.1-flash-lite"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

TEMPERATURE = 1.0
MAX_TOKENS = 4000

USE_NAVIGATOR = False
