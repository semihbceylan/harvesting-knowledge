import requests
import json

OLLAMA_HOST = "http://localhost:11434"

# 1. Define your tool function
def get_weather(city: str, metric: str = "celsius") -> dict:
    """Fetch current weather data for a city with unit metric."""
    res = requests.get(f"https://wttr.in/{city}?format=j1")
    if res.status_code != 200:
        return {"error": f"Failed to get weather for {city}"}
    c = res.json()["current_condition"][0]
    return {
        "city": city,
        "description": c["weatherDesc"][0]["value"],
        "temperature": c["temp_C" if metric == "celsius" else "temp_F"],
        "humidity": c["humidity"],
        "metric": metric,
    }

# 2. Define tool metadata as expected by LLaMA 3.3 format
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city in Celsius or Fahrenheit",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "metric": {
                    "type": "string",
                    "description": "Temperature unit: celsius or fahrenheit",
                    "default": "celsius",
                },
            },
            "required": ["city"],
        },
    },
}

def chat_with_tools(user_input: str):
    system_prompt = (
        "You are an assistant that can call tools. "
        "Use the provided tool definitions to make structured function calls when needed. "
        "You MUST use the exact tool call format so the API returns tool_calls.\n\n"
        "Available functions:\n" +
        json.dumps([weather_tool["function"]], indent=2)
    )

    # Step 1: Ask model what tools to call
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": "llama3.3:70b-instruct-q3_K_M",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "tools": [weather_tool],
            "stream": False
        },
    ).json()
    print("Raw response from Ollama:", json.dumps(resp, indent=2))

    tool_calls = resp["message"].get("tool_calls", [])
    if not tool_calls:
        print("Model didn't request any tool calls.")
        print("Assistant:", resp["message"].get("content"))
        return

    # Step 2: Execute all tool calls and collect tool responses
    tool_messages = []
    for idx, tool_call in enumerate(tool_calls):
        fn = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]
        if fn == "get_weather" and "city" in args:
            result = get_weather(**args)
            tool_messages.append({
                "role": "tool",
                "name": fn,
                "content": json.dumps(result),
            })
        else:
            print("Skipping unknown function or missing args:", fn, args)

    # Step 3: Send back results for model to summarize
    final_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
        resp["message"],
    ] + tool_messages

    final_resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": "llama3.3:70b-instruct-q3_K_M",
            "messages": final_messages,
            "stream": False
        },
    ).json()

    print("Final Assistant:", final_resp["message"]["content"])
    
# Example use
chat_with_tools("What is the weather in Paris and Istanbul in fahrenheit?")
