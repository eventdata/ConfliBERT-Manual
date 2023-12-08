import os
import tempfile
import json
from gradio_client import Client

# Set the temporary directory to the current working directory
tempfile.tempdir = os.getcwd()

# Set your HF_TOKEN
hf_token = "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"  # Replace with your actual HF token

# Clone your running client (must be running on spaces)
client = Client("shreyasmeher/ConfliBERT_Unmask", hf_token="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")

# List of 15 masked sentences
sentences = [
    "Russian forces attacked [MASK] last night.",
    "The [MASK] of the country voiced concerns over the recent events.",
    "International organizations called for a [MASK] to the ongoing conflicts.",
    "The peace treaty was signed in [MASK].",
    "Many civilians fled to [MASK] to escape the violence.",
    "The president announced new measures to ensure [MASK].",
    "Military bases in [MASK] were put on high alert.",
    "The international community was surprised by the sudden [MASK].",
    "Negotiations took place in [MASK] last week.",
    "The prime minister emphasized the importance of [MASK].",
    "The city of [MASK] witnessed heavy artillery fire yesterday.",
    "Humanitarian aid was sent to [MASK] regions affected by the war.",
    "Journalists from around the world gathered in [MASK] for a press conference.",
    "The [MASK] resolution was vetoed by major powers.",
    "Diplomats are working tirelessly in [MASK] to broker a deal."
]

# Process predictions for each sentence and store in desired format
processed_results = {}

for sentence in sentences:
    temp_json_path = client.predict(sentence)
    
    with open(temp_json_path, "r") as f:
        temp_results = json.load(f)
    
    # Assume the structure of temp_results is like the one you provided
    label = temp_results["label"]
    confidences = temp_results["confidences"]
    
    processed_results[sentence] = {
        "label": label,
        "confidences": confidences
    }

# Save processed results to a JSON file in the current working directory
with open("processed_results.json", "w") as f:
    json.dump(processed_results, f, indent=4)

print("Processed results saved to processed_results.json")
