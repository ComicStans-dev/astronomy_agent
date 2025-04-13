import google.generativeai as genai
import os
import logging
# import argparse # Remove argparse, no longer needed for generic prompts
import datetime
import time
from abc import ABC, abstractmethod
import requests # Add requests for weather API
import json # Add json for loading equipment specs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Attempt to get keys from environment first, then use placeholders
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB4XasCrezZm4x1gdiT8CLneCyw3pcbHbQ")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "ca20621df8fd31cbaf40e20a98ce3039")

# Location Configuration
LOCATION_NAME = "Beaverton, Oregon"
LATITUDE = 45.514595
LONGITUDE = -122.847565

# --- LLM Provider Interface ---
class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generates a response from the LLM."""
        pass

    @abstractmethod
    def get_usage_info(self) -> dict:
        """Returns usage information for the last call."""
        pass

# --- Gemini Implementation ---
class GeminiProvider(LLMProvider):
    """Implementation for the Google Gemini provider."""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        if not api_key:
             logging.error("GeminiProvider requires a valid API key (received empty or None).")
             raise ValueError("Invalid or missing Gemini API key provided to GeminiProvider.")
        try:
            genai.configure(api_key=self.api_key)
            # Use a specific, reliable model version
            self.model = genai.GenerativeModel('gemini-1.5-flash') # Or gemini-1.5-pro if preferred
            self._last_usage = {}
            logging.info("Gemini API configured successfully with model: gemini-1.5-flash")
        except Exception as e:
            logging.error(f"Failed to configure Gemini API: {e}")
            raise ConnectionError(f"Failed to configure Gemini API: {e}")

    def generate_response(self, prompt: str) -> str:
        """Sends the prompt to Gemini and returns the response."""
        self._last_usage = {} # Reset usage info
        try:
            start_time = time.time()
            response = self.model.generate_content(prompt)
            end_time = time.time()

            # Basic usage info
            response_text = ""
            # Handle cases where response might be structured differently
            if response.parts:
                response_text = response.text # Preferred method
            elif response.candidates and response.candidates[0].content.parts:
                 response_text = " ".join(part.text for part in response.candidates[0].content.parts)

            self._last_usage = {
                "timestamp": datetime.datetime.now().isoformat(),
                "provider": "gemini",
                "model": "gemini-1.5-flash", # Match model used
                "prompt_length": len(prompt),
                "response_length": len(response_text),
                "latency_seconds": round(end_time - start_time, 2),
                # Attempt to get token counts if available in usage_metadata
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', None),
                "candidates_tokens": getattr(response.usage_metadata, 'candidates_token_count', None),
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', None),
            }
            logging.info(f"Gemini response received. Latency: {self._last_usage['latency_seconds']}s")

            # Check for blocked content
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                logging.error(f"Prompt blocked due to: {block_reason}")
                self._last_usage["error"] = f"Blocked: {block_reason}"
                # Raise a specific error for blocked prompts
                raise ValueError(f"Prompt blocked by API: {block_reason}")
            
            # Check for potentially empty but not blocked responses
            if not response_text and not (hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason):
                logging.warning("Gemini response appears empty.")
                finish_reason = "Unknown"
                if response.candidates:
                    finish_reason = str(response.candidates[0].finish_reason)
                warning_msg = f"Warning: Received empty response from Gemini. Finish Reason: {finish_reason}."
                self._last_usage["warning"] = warning_msg
                # Return warning but don't raise error unless blocked
                return warning_msg

            return response_text
        except (genai.types.generation_types.StopCandidateException,
                genai.types.generation_types.BlockedPromptException,
                genai.types.generation_types.InvalidArgumentException) as gen_ex:
             # Catch specific Gemini SDK errors
             logging.error(f"Gemini generation error: {gen_ex}")
             self._last_usage["error"] = str(gen_ex)
             # Re-raise the specific Gemini exception for upstream handling
             raise ValueError(f"Gemini API Error: {gen_ex}") from gen_ex
        except Exception as e:
            # Catch unexpected errors during API call
            logging.error(f"Unexpected error calling Gemini API: {e}", exc_info=True)
            if "rate limit" in str(e).lower():
                logging.warning("Potential rate limit hit. Consider adding delays or backoff.")
            self._last_usage["error"] = str(e)
            # Raise a ConnectionError for consistency on call failures
            raise ConnectionError(f"Failed to get response from Gemini: {e}") from e

    def get_usage_info(self) -> dict:
        """Returns usage information for the last Gemini call."""
        return self._last_usage

# --- Weather Fetching ---
def get_weather_data(api_key: str, lat: float, lon: float) -> dict:
    """
    Fetches weather data for the specified location using OpenWeatherMap API.
    Requires 'requests' library: pip install requests
    Uses the provided API key.
    """
    logging.info(f"Attempting to fetch weather for Lat: {lat}, Lon: {lon}")
    if not api_key:
        logging.warning("Weather API key not provided to get_weather_data function.")
        # Return error data if API key is missing
        return {
            "description": "Weather data unavailable (API key missing in function call)",
            "cloud_cover_percent": -1,
            "seeing_conditions": "Unknown",
            "temperature_c": -999,
            "humidity_percent": -1,
            "error": "API key missing"
        }

    # Use the OpenWeatherMap Current Weather endpoint
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric" # Get temperature in Celsius
    }

    try:
        response = requests.get(base_url, params=params, timeout=10) # 10 second timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Extract relevant information
        description = data.get('weather', [{}])[0].get('description', 'N/A')
        cloud_cover = data.get('clouds', {}).get('all', -1) # Cloudiness percentage
        temperature = data.get('main', {}).get('temp', -999)
        humidity = data.get('main', {}).get('humidity', -1)

        # Basic seeing condition inference (very simplified)
        seeing = "Unknown"
        if cloud_cover != -1:
            if cloud_cover <= 20:
                seeing = "Good"
            elif cloud_cover <= 60:
                seeing = "Average"
            else:
                seeing = "Poor"

        weather_info = {
            "description": description,
            "cloud_cover_percent": cloud_cover,
            "seeing_conditions": seeing,
            "temperature_c": temperature,
            "humidity_percent": humidity
        }
        logging.info(f"Weather data fetched successfully: {description}, Clouds: {cloud_cover}%" )
        return weather_info

    except requests.exceptions.Timeout:
        logging.error("Weather API request timed out.")
        return {"description": "Error: Weather API timeout", "error": "Timeout", "cloud_cover_percent": -1, "seeing_conditions": "Unknown"}
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err} - Status Code: {response.status_code}")
        # Handle specific errors like 401 Unauthorized (invalid key) or 429 Too Many Requests
        error_detail = f"HTTP Error: {response.status_code}"
        if response.status_code == 401:
            error_detail = "HTTP Error: 401 Unauthorized (Invalid API Key?)"
        elif response.status_code == 429:
            error_detail = "HTTP Error: 429 Too Many Requests (Rate Limit?)"
        return {"description": error_detail, "error": str(http_err), "cloud_cover_percent": -1, "seeing_conditions": "Unknown"}
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Error fetching weather data: {req_err}")
        return {"description": f"Error fetching weather: {req_err}", "error": str(req_err), "cloud_cover_percent": -1, "seeing_conditions": "Unknown"}
    except Exception as e:
        # Catch other potential errors (e.g., JSON parsing)
        logging.error(f"Error processing weather data: {e}", exc_info=True)
        return {"description": f"Error processing weather data: {e}", "error": str(e), "cloud_cover_percent": -1, "seeing_conditions": "Unknown"}

# --- Equipment Specs Loading ---
def load_equipment_specs(filepath="Astronomy Agent/equipment_specs.json") -> dict:
    """Loads equipment specifications from a JSON file."""
    try:
        with open(filepath, 'r') as file:
            specs = json.load(file)
        logging.info(f"Equipment specifications loaded successfully from {filepath}.")
        return specs
    except FileNotFoundError:
        logging.error(f"Equipment specifications file not found at {filepath}. Returning empty specs.")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing equipment JSON file {filepath}: {e}. Returning empty specs.")
        return {}
    except Exception as e:
        logging.error(f"An unexpected error occurred loading {filepath}: {e}. Returning empty specs.")
        return {}

# --- Astronomy Prompt Generation ---
def create_astronomy_prompt(location_name: str, lat: float, lon: float, date: str, weather: dict, equipment_specs: dict) -> str:
    """Creates the detailed prompt for the LLM, including equipment specs."""
    # Safely get weather details
    weather_str = f"Cloud Cover: {weather.get('cloud_cover_percent', 'N/A')}%, Seeing: {weather.get('seeing_conditions', 'N/A')}, Temp: {weather.get('temperature_c', 'N/A')}C, Humidity: {weather.get('humidity_percent', 'N/A')}%, Description: {weather.get('description', 'N/A')}"
    if weather.get("error"):
        weather_str = f"Weather data retrieval failed: {weather.get('description', 'Unknown error')}"

    # Format equipment specs for the prompt
    specs_str = "No equipment specifications provided." # Default message
    if equipment_specs:
        try:
            specs_str = json.dumps(equipment_specs, indent=2)
        except Exception as e:
            logging.error(f"Failed to convert equipment specs to JSON string: {e}")
            specs_str = "Error formatting equipment specifications."

    # Main prompt template including equipment
    prompt = f"""
You are an expert astronomy assistant providing advice for astrophotography.

Location: {location_name} (Latitude: {lat:.4f}, Longitude: {lon:.4f})
Date: {date}
Weather Forecast: {weather_str}

User's Equipment:
```json
{specs_str}
```

Based on the location, date, weather, and the user's equipment, please identify the top 3-5 deep-sky objects (galaxies, nebulae) AND any prominent planets that are well-positioned for imaging tonight.

Critically evaluate the weather forecast and consider the equipment:
- If conditions are poor (e.g., high cloud cover > 60-70%, Poor seeing), state this and suggest alternatives like focusing only on the brightest targets suitable for the equipment (if any are visible), processing old data, planning, or lunar/planetary imaging.
- If conditions are marginal, advise focusing on brighter objects or using shorter exposures, considering the camera's sensitivity and telescope's focal length.
- If conditions are good, recommend a mix of targets, prioritizing those well-suited to the provided telescope's focal length ({equipment_specs.get('telescope', {}).get('focal_length_mm', 'N/A')}mm) and camera's pixel scale (calculate if possible: pixel scale ≈ 206.265 * {equipment_specs.get('imaging_camera', {}).get('pixel_size_microns', 'N/A')} / {equipment_specs.get('telescope', {}).get('focal_length_mm', 'N/A')} arcsec/pixel).

For EACH recommended object:
1.  **Identification:** Common name and designation.
2.  **Visibility Window:** Altitude/azimuth range during optimal window (~9 PM - 3 AM local), culmination/rise/set info.
3.  **Framing:** Assess how well the object fits the Field of View (FOV) provided by the user's telescope ({equipment_specs.get('telescope', {}).get('focal_length_mm', 'N/A')}mm) and camera ({equipment_specs.get('imaging_camera', {}).get('sensor', 'N/A')} sensor, {equipment_specs.get('imaging_camera', {}).get('pixel_size_microns', 'N/A')}μm pixels). Suggest if it's a good fit, too large, or too small.
4.  **Filters & Physics:** Recommend specific filters (L/RGB, Ha/OIII/SII, Dual/Multi-band) considering the object type, {location_name}'s light pollution, and the user's color camera ({equipment_specs.get('imaging_camera', {}).get('model', 'N/A')}). Explain the physics justifying the choice (emission lines vs broadband, contrast, color balance with OSC).
5.  **Beginner Tip:** Practical, actionable tip (e.g., starting exposure based on f/{equipment_specs.get('telescope', {}).get('focal_ratio', 'N/A')} ratio, calibration frames, focusing check).
6.  **Advanced Insight:** Technical insight linking physics/optics/materials to the user's setup (e.g., Resolution vs. pixel scale/seeing, SNR optimization for the specific camera sensor, QE impact, thermal noise in the {equipment_specs.get('imaging_camera', {}).get('model', 'N/A')}, aberration correction relevant to the f/{equipment_specs.get('telescope', {}).get('focal_ratio', 'N/A')} scope).

Target Audience: PhD-level physics/optics understanding, but new to practical amateur astronomy. Bridge theory and practice using their specific gear.
"""
    return prompt.strip()

# --- Main Execution ---
def run_astronomy_assistant():
    """Main function to run the astronomy assistant."""
    print("--- Starting Astronomy Assistant ---")

    # === API Key Verification ===
    # Access the global constants defined at the top
    global_gemini_key = GEMINI_API_KEY
    global_weather_key = WEATHER_API_KEY

    # Verify Gemini API Key
    # Check if the key exists and is not None or empty
    if not global_gemini_key:
        print("Error: Gemini API Key is missing (not found in env or default value).")
        logging.error("Exiting: Gemini API Key not configured correctly.")
        return # Exit if no valid key
    else:
        print(f"Using Gemini API Key starting with: {global_gemini_key[:4]}...")

    # Verify Weather API Key (using the global one)
    # Check if the key exists and is not None or empty
    if not global_weather_key:
        print("Error: Weather API Key is missing (not found in env or default value).")
        logging.error("Exiting: Weather API Key not configured correctly.")
        return # Exit if no valid weather key
    else:
        print(f"Using OpenWeatherMap API Key starting with: {global_weather_key[:4]}...")
        logging.info(f"Using OpenWeatherMap API Key prefix: {global_weather_key[:4]}")
    # =========================

    # === Load Equipment Specs ===
    equipment_specs = load_equipment_specs() # Load from default path
    if not equipment_specs:
        print("Warning: Could not load equipment specifications. Proceeding without them.")
        logging.warning("Equipment specs file not found or invalid. Prompt will not include equipment details.")
    # ==========================

    llm = None # Initialize llm to None
    try:
        # Now, use the verified keys when initializing providers and fetching data
        logging.info("Initializing LLM provider: Gemini")
        llm = GeminiProvider(global_gemini_key) # Use the verified key

        current_date = datetime.date.today().isoformat()
        logging.info(f"Getting weather for {LOCATION_NAME} ({LATITUDE:.4f}, {LONGITUDE:.4f}) on {current_date}")
        weather_data = get_weather_data(global_weather_key, LATITUDE, LONGITUDE) # Use the verified key
        logging.info(f"Weather data received: {weather_data}")

        # Check if weather fetch failed and provide feedback
        if weather_data.get("error"):
             weather_error_msg = weather_data.get("description", "Unknown weather error")
             print(f"\nWarning: Could not retrieve valid weather data ({weather_error_msg}).")
             print("Proceeding with error information for the prompt, recommendations may be limited.")
             logging.warning(f"Failed to retrieve weather: {weather_error_msg}. Prompt will use error info.")
             # Ensure structure for prompt generation even if error occurred
             weather_data = {
                 "description": weather_error_msg,
                 "cloud_cover_percent": -1,
                 "seeing_conditions": "Unknown",
                 "temperature_c": -999,
                 "humidity_percent": -1,
                 "wind_speed_mps": -1,
                 "error": weather_data.get("error") # Preserve the error type
             }

        # Create the prompt, now including equipment specs
        logging.info("Generating astronomy prompt including equipment specs...")
        astro_prompt = create_astronomy_prompt(LOCATION_NAME, LATITUDE, LONGITUDE, current_date, weather_data, equipment_specs)
        # Ensure this multi-line comment is correctly formatted and doesn't cause syntax errors
        # print(f"\n--- DEBUG: Generated Prompt ---\n{astro_prompt}\n-----------------------------\n")

        logging.info("Sending prompt to Gemini for astronomy advice...")
        response_text = llm.generate_response(astro_prompt)

        # Print the results
        print("\n--- Astronomy Assistant Advice ---")
        print(f"Date: {current_date}")
        print(f"Location: {LOCATION_NAME} ({LATITUDE:.4f}, {LONGITUDE:.4f})") # Format coords
        # Display weather info used in prompt (might include error state)
        temp_c = weather_data.get('temperature_c', 'N/A')
        cloud = weather_data.get('cloud_cover_percent', 'N/A')
        seeing = weather_data.get('seeing_conditions', 'N/A')
        desc = weather_data.get('description', 'N/A')
        weather_summary = f"Desc: {desc}, Cloud: {cloud}%, Seeing: {seeing}, Temp: {temp_c}°C"
        if weather_data.get("error"):
             weather_summary = f"Weather Status: Error - {weather_data.get('description', 'Unknown Error')}"
        print(f"Weather Used: {weather_summary}")
        print("---")
        print(response_text)
        print("---------------------------------\n")

        # Log usage info after successful generation
        if llm:
            usage_info = llm.get_usage_info()
            logging.info(f"LLM Usage Info: {usage_info}")
            # print(f"LLM Usage: Tokens={usage_info.get('total_tokens', 'N/A')}, Latency={usage_info.get('latency_seconds', 'N/A')}s")

    except ValueError as ve:
        # Catch specific errors like API key issues or blocked prompts from GeminiProvider
        logging.error(f"Configuration or Value Error: {ve}")
        print(f"\nError: {ve}")
    except ConnectionError as ce:
        # Catch connection issues during API calls (Gemini or Weather)
        logging.error(f"Connection Error: {ce}")
        print(f"\nError: Could not connect to a required service - {ce}")
    except Exception as e:
        # Catch any other unexpected errors during the main flow
        logging.error(f"Astronomy assistant failed unexpectedly: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # Optional: Log usage even if an error occurred after the LLM call but before logging
        if llm and llm.get_usage_info():
            usage_info = llm.get_usage_info()
            if usage_info.get("error") or usage_info.get("warning"):
                logging.warning(f"LLM Usage Info (potentially partial due to error): {usage_info}")

    print("--- Astronomy Assistant Finished ---")

if __name__ == "__main__":
    # ... (requests check remains the same)
    try:
        import requests
    except ImportError:
        print("Error: The 'requests' library is not installed.")
        print("Please install it using: pip install requests")
        exit(1)
    run_astronomy_assistant()
    # Removed previous complex key checking logic, relying on checks at the start of run_astronomy_assistant 