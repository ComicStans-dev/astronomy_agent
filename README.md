# 🔭 Astro Agent

An intelligent agent that provides personalized astrophotography advice based on your location, current weather conditions, and equipment specifications.

## 🌌 Overview

The Astro Agent leverages Google's Gemini generative AI to recommend optimal celestial targets for imaging based on:

- 📍 Your geographical location
- 📅 Current date and time
- ☁️ Real-time weather conditions (cloud cover, seeing conditions, etc.)
- 📷 Your specific astronomy equipment setup

The assistant analyzes these factors to suggest 3-5 deep-sky objects (DSOs) and prominent planets that are well-positioned for imaging, with detailed information about visibility, framing, filter recommendations, and equipment-specific tips.

## ✨ Features

- **🌤️ Weather-Aware Recommendations**: Fetches real-time weather data from OpenWeatherMap and adapts recommendations based on viewing conditions
- **🔧 Equipment-Specific Advice**: Provides tailored advice based on your exact telescope, camera, mount, and accessories
- **👥 Role-Based Equipment Management**: Store all your gear with proper categorization (imaging telescope, guide scope, cameras, etc.)
- **🧠 Advanced Astrophotography Insights**: Includes physics-based explanations for filter choices, exposure recommendations, and technical considerations specific to your setup
- **🔰 Beginner-Friendly Tips**: Each recommendation includes practical advice for successful imaging

## 🛠️ Setup Requirements

### Dependencies

```
pip install google-generativeai requests rich
```

### 🔑 API Keys

The assistant requires two API keys:

1. **🧠 Google Gemini API Key**: 
   - Create or use an existing key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Store in environment variable: `GEMINI_API_KEY`

2. **☁️ OpenWeatherMap API Key**:
   - Create a free account at [OpenWeatherMap](https://openweathermap.org/api)
   - Store in environment variable: `WEATHER_API_KEY`

You can also hardcode these in the script (see Configuration section).

## 🚀 Usage

1. Set up your equipment specifications in `equipment_specs.json` (see format below)
2. Configure your location in the script (see Configuration section)
3. Run the assistant:

```
python astro_agent.py
```

The assistant will:
- ✅ Verify API keys
- 📷 Load your equipment specifications
- ☁️ Fetch current weather data for your location
- 🌌 Generate tailored astrophotography recommendations

## 📝 Equipment Specifications

The `equipment_specs.json` file uses a role-based structure to organize your astronomy gear:

```json
{
  "imaging_telescope": {
    "model": "Your Telescope Model",
    "specs": {
      "optical_design": "Refractor/SCT/Newtonian/etc",
      "aperture_mm": 80,
      "focal_length_mm": 600,
      "focal_ratio": 7.5,
      "image_circle_mm": 44
      // Additional specs...
    }
  },
  "imaging_camera": {
    "model": "Your Camera Model",
    "specs": {
      "sensor_type": "CMOS/CCD",
      "pixel_size_microns": 3.76,
      "resolution_width_px": 4144,
      "resolution_height_px": 2822,
      // Additional specs...
    }
  },
  // Additional components: guide_scope, guide_camera, 
  // mount, filter, focuser, control_computer, etc.
}
```

## ⚙️ Configuration

Edit these parameters at the top of `astro_agent.py`:

```python
# API Keys (can also be set as environment variables)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-default-key-here")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "your-default-key-here")

# Location Configuration
LOCATION_NAME = "Area 51 (Maybe)"  # Human-readable location name
LATITUDE = 37.2350
LONGITUDE = -115.8111
```

## 🔍 How It Works

1. **🔄 Setup & Validation**: The assistant verifies API keys and loads equipment specs
2. **☁️ Weather Analysis**: Fetches current conditions from OpenWeatherMap API
3. **📊 Context Assembly**: Creates a detailed prompt including location, weather, and equipment specifications
4. **🧠 LLM Processing**: Sends the prompt to Google's Gemini model
5. **💡 Response Presentation**: Displays the astronomy advice with target recommendations

## 🔧 Technical Details

The assistant implements:
- Abstract provider interface for LLM services
- Error handling for API failures
- Detailed logging
- Weather condition inference (seeing conditions based on cloud cover)
- Equipment-aware celestial target recommendations

## 📜 License

MIT 