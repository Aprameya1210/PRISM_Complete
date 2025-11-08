import argparse
import json
import re
import subprocess
import sys
import os
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

class DensityEstimate(BaseModel):
    density_g_ml: float = Field(..., description="The estimated density in g/mL.")
    food_identified: str = Field(..., description="The common name of the food identified by the model.")
    rationale: Optional[str] = Field(None, description="Reasoning for the density value.")

def extract_text_from_genai_response(response):
    pieces = []
    try:
        if hasattr(response, "output") and response.output:
            for out in response.output:
                if hasattr(out, "content") and out.content:
                    for c in out.content:
                        if hasattr(c, "text") and c.text:
                            pieces.append(c.text)
                        elif isinstance(c, dict) and c.get("text"):
                            pieces.append(c.get("text"))
        if not pieces and hasattr(response, "text"):
            pieces.append(response.text)
    except Exception:
        try:
            pieces.append(str(response))
        except Exception:
            pieces = []
    return "\n".join(pieces).strip()

def get_food_density(food_name: str, client: genai.Client) -> float:
    if not food_name:
        print("-> [Gemini Query] No food name provided. Defaulting to 1.0 g/mL (water).")
        return 1.0

    prompt = f"Provide the density (in g/mL) for this food item and return only JSON with keys: density_g_ml (number), food_identified (string), rationale (optional string). Food: {food_name}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[prompt]
        )
        text = extract_text_from_genai_response(response)
        json_text_match = re.search(r"(\{[\s\S]*\})", text)
        if json_text_match:
            json_text = json_text_match.group(1)
        else:
            json_text = text
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            print(f"-> [Gemini Query] Could not parse JSON from model response. Raw response:\n{text}")
            print("-> Defaulting to 1.0 g/mL (water).")
            return 1.0
        if parsed:
            try:
                dto = DensityEstimate(**parsed)
                print(f"-> [Gemini Query] Found density for '{dto.food_identified}': {dto.density_g_ml} g/mL")
                return dto.density_g_ml
            except Exception as e:
                if isinstance(parsed.get("density_g_ml"), (int, float)):
                    print(f"-> [Gemini Query] Warning: Schema validation failed ({e}), but found density.")
                    return float(parsed.get("density_g_ml"))
                print(f"-> [Gemini Query] JSON parsed but did not match schema: {e}")
    except Exception as e:
        print(f"-> [Gemini Query] API call failed: {repr(e)}")

    print("-> Defaulting to 1.0 g/mL (water).")
    return 1.0
 
def run_food_detection(image_path: str):
    command = [sys.executable, "main.py", "--image", image_path]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding="utf-8")
        output = result.stdout
        json_match = re.search(r"--- Detection Results ---\s*(\{[\s\S]*\})", output, re.DOTALL)
        if not json_match:
            print("Error: Could not find '--- Detection Results ---' in main.py output.")
            print(output)
            return None, None
        data = json.loads(json_match.group(1))
        food_percentage = data.get("food_percentage")
        if food_percentage is None:
            print("Error: 'food_percentage' not in JSON output.")
            return None, None
        ignore_classes = {58.0, 31.0, 42.0, 70.0, 83.0, 25.0, 27.0, 22.0, 11.0, 8.0}
        garbage_classes = {35.0}
        main_food_name = None
        max_food_area = 0
        for obj in data.get("objects", []):
            label = obj.get("label")
            if label not in ignore_classes and label not in garbage_classes:
                if obj.get("area", 0) > max_food_area:
                    max_food_area = obj["area"]
                    main_food_name = obj.get("label_name")
        if main_food_name is None:
            print("Warning: Could not identify the main food item.")
        return food_percentage / 100.0, main_food_name
    except subprocess.CalledProcessError as e:
        print(f"Error running main.py: {e}")
        print(getattr(e, "stderr", ""))
        return None, None
    except json.JSONDecodeError:
        print("Error: Could not parse JSON from main.py output.")
        try:
            print(output)
        except Exception:
            pass
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

def run_volume_estimation(image_path: str):
    command = [sys.executable, "-m", "capacity_estimator.cli", "--image", image_path, "--use-gemini"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding="utf-8")
        output = result.stdout
        volume_match = re.search(r"Estimated volume: ([\d\.]+)\s*mL", output)
        if not volume_match:
            print("Error: Could not parse 'Estimated volume' from capacity_estimator output.")
            print(output)
            return None
        volume_ml = float(volume_match.group(1))
        return volume_ml
    except subprocess.CalledProcessError as e:
        print(f"Error running capacity_estimator: {e}")
        print(getattr(e, "stderr", ""))
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Calculate the total mass of food in an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        sys.exit(1)
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error configuring Gemini API client: {e}")
        sys.exit(1)
    food_percentage, food_name = run_food_detection(args.image)
    if food_percentage is None:
        print("Failed to get food detection data. Exiting.")
        sys.exit(1)
    print(f"-> Detected Food: '{food_name}'")
    print(f"-> Detected Food Percentage: {food_percentage * 100:.2f}%")
    total_volume_ml = run_volume_estimation(args.image)
    if total_volume_ml is None:
        print("Failed to get volume estimation. Exiting.")
        sys.exit(1)
    print(f"-> Detected Total Volume: {total_volume_ml:.2f} mL")
    food_volume_ml = total_volume_ml * food_percentage
    print(f"-> Calculated Food Volume: {food_volume_ml:.2f} mL")
    density_g_ml = get_food_density(food_name, client)
    food_mass_g = food_volume_ml * density_g_ml
    print("\n--- FINAL RESULT ---")
    print(f"Estimated food mass: {food_mass_g:.2f} g")

if __name__ == "__main__":
    main()