# Planning Agent Guide

The **Planning Agent** is an autonomous travel assistant capable of creating detailed itineraries by interacting with real-world APIs for flights, weather, and attractions.

## Features
- **Flight Search**: Real-time flight data via Amadeus API.
- **Attractions**: Location-based recommendations via Google Places API.
- **Weather**: Forecasts for your trip dates (Mock/Real).
- **Itinerary Generation**: Produces a structured JSON itinerary and a natural language summary.

## Setup

### 1. Environment Variables
Ensure your `.env` file has the following keys:

```ini
# Azure OpenAI (Core LLM)
AZURE_OPENAI_API_VERSION="2024-02-15-preview"
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT="gpt-4o"
OPENAI_API_KEY="your-azure-key"

# Amadeus (Flights)
# Get keys from: https://developers.amadeus.com/
AMADEUS_CLIENT_ID="your-amadeus-api-key"
AMADEUS_CLIENT_SECRET="your-amadeus-client-secret"

# Google Maps (Attractions)
# Enable "Places API" (Legacy) in Google Cloud Console
GPLACES_API_KEY="your-google-maps-key"
```

### 2. Dependencies
Install the required packages (if not already installed):
```bash
uv sync
```

## Usage

1. **Start the Chat Interface**:
   ```bash
   uv run python chat.py
   ```

2. **Select Agent**:
   - In the Gradio UI, select **"Planning Agent"** from the dropdown menu (if applicable) or simply start chatting.

3. **Example Queries**:
   - "Plan a 3-day trip to Auckland from Sydney starting December 25th."
   - "Find me a weekend getaway to Queenstown with a budget of $1000."
   - "What are the best attractions in Wellington for families?"

## Troubleshooting
- **Amadeus Error (401)**: Ensure you are NOT using your OpenAI key for Amadeus. Check `AMADEUS_CLIENT_ID`.
- **Google Places Error (Request Denied)**: Ensure the **"Places API"** (Legacy) is enabled in your Google Cloud Console, not just the "New" one.
