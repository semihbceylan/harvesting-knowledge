import os
import glob
import json
import requests
from datetime import datetime
from PIL import Image
import streamlit as st

OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.2:3b"

# Tool function to resolve image path from folder based on input
def get_field_image_path(date: str, lens: str, zoom: str) -> dict:
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        year = date_obj.strftime("%Y")
        month = date_obj.strftime("%m")
        day = date_obj.strftime("%d")

        base_dir = "47.04"
        folder_path = os.path.join(base_dir, year, lens, zoom.upper())
        pattern = f"{year}_{month}_{day}-*-{zoom.lower()}.jpeg"
        full_pattern = os.path.join(folder_path, pattern)
        matches = glob.glob(full_pattern)

        if matches:
            return {"path": matches[0]}
        else:
            return {"error": "Image not found for given parameters."}
    except Exception as e:
        return {"error": str(e)}

image_search_tool = {
    "type": "function",
    "function": {
        "name": "get_field_image_path",
        "description": "Retrieve a field image path from folder by date, lens, and zoom level.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD"},
                "lens": {"type": "string", "description": "Lens name, like K2"},
                "zoom": {"type": "string", "description": "Zoom level, e.g., 10x"}
            },
            "required": ["date", "lens", "zoom"]
        }
    }
}

def chat_with_tools(user_input: str):
    system_prompt = (
        "You are an assistant that can locate field images based on date, lens, and zoom level.\n"
        "Only call the tool if the user provides enough info (date, lens, zoom).\n"
        "If you decide to use the function, use this format: [get_field_image_path(date='YYYY-MM-DD', lens='K2', zoom='10x')]\n\n"
        "Here is a list of available functions:\n\n" + json.dumps([image_search_tool["function"]], indent=2)
    )

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "tools": [image_search_tool],
            "stream": False
        },
    )

    if response.status_code != 200:
        return {"error": response.text}

    resp_json = response.json()
    tool_calls = resp_json.get("message", {}).get("tool_calls", [])

    if not tool_calls:
        return {"error": "No tool call returned by model."}

    for call in tool_calls:
        fn = call["function"]["name"]
        args = call["function"]["arguments"]
        if fn == "get_field_image_path":
            return get_field_image_path(**args)
    return {"error": "No valid tool function call found."}

# Streamlit UI
st.set_page_config(page_title="Smart Field Image Search", layout="centered")
st.title("🌱 Smart Field Image Search")

user_input = st.text_input("Describe the image you want:",
                           "Show me the image from May 23rd 2014 with K2 at 10x zoom.")
if st.button("Find Image"):
    with st.spinner("Searching..."):
        result = chat_with_tools(user_input)
        if "path" in result:
            st.success(f"Image found: {result['path']}")
            st.image(result["path"], caption=os.path.basename(result["path"]), use_column_width=True)
        else:
            st.error(result.get("error", "Unknown error occurred."))
