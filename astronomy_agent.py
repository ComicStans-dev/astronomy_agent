import google.generativeai as genai
import os
import logging
# import argparse # Remove argparse, no longer needed for generic prompts
import datetime
import time
from abc import ABC, abstractmethod
import requests # Add requests for weather API
import json # Add json for loading equipment specs
from dotenv import load_dotenv
from pathlib import Path

# Add rich library for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

# Initialize rich console
console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Load environment variables from .env file
# Try to find .env in the current directory or in the parent directory
env_path = Path('.env')
if not env_path.exists():
    env_path = Path('Astronomy Agent/.env')
    
load_dotenv(dotenv_path=env_path)

# Get API keys from environment variables (no default values to avoid hardcoding)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

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
    """Main execution function for the astronomy assistant."""
    
    console.print(Panel.fit("[bold blue]🔭 Astronomy Assistant[/bold blue]", title="Welcome", subtitle="Powered by Gemini AI"))
    
    # Check for API keys (critical)
    if not GEMINI_API_KEY:
        console.print("[bold red]❌ Error: Gemini API key not found![/bold red]")
        console.print("[yellow]Please set the GEMINI_API_KEY environment variable or configure it in the script.[/yellow]")
        return
    
    if not WEATHER_API_KEY:
        console.print("[bold yellow]⚠️ Warning: Weather API key not found![/bold yellow]")
        console.print("[yellow]Weather data will not be available. Please set the WEATHER_API_KEY environment variable.[/yellow]")
    
    # Display configuration
    console.print("\n[bold cyan]📍 Location:[/bold cyan] " + LOCATION_NAME)
    console.print(f"[cyan]   Coordinates: {LATITUDE}, {LONGITUDE}[/cyan]")
    
    # Initialize LLM provider
    try:
        with console.status("[bold green]🧠 Initializing Gemini AI...[/bold green]"):
            llm = GeminiProvider(GEMINI_API_KEY)
        console.print("[bold green]✅ Gemini AI initialized successfully[/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ Error initializing Gemini AI: {e}[/bold red]")
        return
    
    # Get current date/time
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"\n[bold cyan]📅 Current Date/Time:[/bold cyan] {current_date}")
    
    # Load equipment specifications
    with console.status("[bold green]📷 Loading equipment specifications...[/bold green]"):
        equipment_specs = load_equipment_specs()
    
    if equipment_specs:
        console.print("[bold green]✅ Equipment specifications loaded successfully[/bold green]")
        # Display summary of equipment
        if "imaging_telescope" in equipment_specs:
            console.print(f"[cyan]   🔭 Telescope: {equipment_specs['imaging_telescope']['model']}[/cyan]")
        if "imaging_camera" in equipment_specs:
            console.print(f"[cyan]   📷 Camera: {equipment_specs['imaging_camera']['model']}[/cyan]")
        if "mount" in equipment_specs:
            console.print(f"[cyan]   🛠️ Mount: {equipment_specs['mount']['model']}[/cyan]")
    else:
        console.print("[bold yellow]⚠️ Warning: No equipment specifications found![/bold yellow]")
        console.print("[yellow]   Using default recommendations without equipment-specific advice.[/yellow]")
    
    # Get weather data
    if WEATHER_API_KEY:
        with console.status("[bold green]☁️ Fetching weather data...[/bold green]"):
            weather = get_weather_data(WEATHER_API_KEY, LATITUDE, LONGITUDE)
        
        if "error" not in weather:
            console.print("[bold green]✅ Weather data retrieved successfully[/bold green]")
            console.print(f"[cyan]   ☁️ Cloud Cover: {weather['cloud_cover_percent']}%[/cyan]")
            console.print(f"[cyan]   🌡️ Temperature: {weather['temperature_c']}°C[/cyan]")
            console.print(f"[cyan]   💧 Humidity: {weather['humidity_percent']}%[/cyan]")
            console.print(f"[cyan]   👁️ Seeing Conditions: {weather['seeing_conditions']}[/cyan]")
            console.print(f"[cyan]   📝 Description: {weather['description']}[/cyan]")
        else:
            console.print(f"[bold yellow]⚠️ Warning: {weather['description']}[/bold yellow]")
            weather = {
                "description": "Weather data unavailable",
                "cloud_cover_percent": -1,
                "seeing_conditions": "Unknown",
                "temperature_c": -999,
                "humidity_percent": -1
            }
    else:
        console.print("[bold yellow]⚠️ Weather data not available (missing API key)[/bold yellow]")
        weather = {
            "description": "Weather data unavailable (no API key)",
            "cloud_cover_percent": -1,
            "seeing_conditions": "Unknown",
            "temperature_c": -999,
            "humidity_percent": -1
        }
    
    # Create prompt
    with console.status("[bold green]🧩 Creating astronomy context...[/bold green]"):
        prompt = create_astronomy_prompt(LOCATION_NAME, LATITUDE, LONGITUDE, current_date, weather, equipment_specs)
    
    # Get recommendations from LLM
    with console.status("[bold green]🔮 Generating astronomy recommendations...[/bold green]") as status:
        try:
            recommendations = llm.generate_response(prompt)
            console.print("[bold green]✅ Recommendations generated[/bold green]\n")
            
            # Display formatted recommendations
            console.print(Panel.fit(
                Markdown(recommendations),
                title="[bold cyan]🌌 Astrophotography Recommendations[/bold cyan]",
                border_style="cyan"
            ))
            
            usage = llm.get_usage_info()
            if usage and "latency_seconds" in usage:
                console.print(f"[dim]Generated in {usage['latency_seconds']} seconds[/dim]")
        except Exception as e:
            console.print(f"[bold red]❌ Error generating recommendations: {e}[/bold red]")
            return
    
    console.print("\n[bold blue]Thanks for using Astronomy Assistant! Clear skies! ✨[/bold blue]")

# --- Execution ---
if __name__ == "__main__":
    run_astronomy_assistant() 