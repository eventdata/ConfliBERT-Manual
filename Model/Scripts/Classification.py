from transformers import BertTokenizer, BertForSequenceClassification
import torch
import csv
import os

def load_model_and_tokenizer(model_path):
    try:
        # Load a standard BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Update its vocabulary with the problematic tokenizer's vocabulary
        tokenizer_vocab_path = model_path + "/vocab.txt"
        tokenizer = BertTokenizer.from_pretrained(tokenizer_vocab_path, do_lower_case=False)
        
        # Set special tokens
        tokenizer.pad_token = "[PAD]"
        tokenizer.cls_token = "[CLS]"
        tokenizer.mask_token = "[MASK]"
        tokenizer.sep_token = "[SEP]"
        tokenizer.unk_token = "[UNK]"

        model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        return tokenizer, model

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

# Paths to the model and tokenizer
model_path = "C:\\Users\\shrey\\Box\\RA\\ConfliBERT\\ConfliBERT-main\\outputs\\IndiaPoliceEvents_sents\\best_model"

# Load tokenizer and model
tokenizer, model = load_model_and_tokenizer(model_path)

# Sample sentences for classification
# Sample sentences for classification
sentences = [
    "The police shot the man, leading to his immediate death.",  # KILL
    "Officers arrested five individuals who were suspected of theft.",  # ARREST
    "Despite the ongoing theft, the police just stood by and watched without intervening.",  # FAIL
    "The police used pepper spray and batons to control the unruly crowd.",  # FORCE
    "The officers patrolled the area and interacted with the local residents.",  # ANY_ACTION
    "A woman was killed in the crossfire between the police and the criminals.",  # KILL
    "The police detained several individuals for questioning.",  # ARREST
    "Even as the violence escalated, the police failed to take any action.",  # FAIL
    "Riot police used shields and rubber bullets to push back the protestors.",  # FORCE
    "The police carried out a search operation in the neighborhood.",  # ANY_ACTION
    "During the raid, two suspects were fatally shot by the police.",  # KILL
    "After a high-speed chase, the police arrested the driver.",  # ARREST
    "The police were criticized for not doing anything during the major heist.",  # FAIL
    "Officers used tear gas to disperse the mob that was gathering.",  # FORCE
    "The police set up checkpoints throughout the city after the terror alert."  # ANY_ACTION
]

results = []

if tokenizer and model:
    class_names = ["KILL", "ARREST", "FAIL", "FORCE", "ANY_ACTION"]
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits

        probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        predicted_class = class_names[torch.argmax(logits, dim=1).item()]

        result = {"sentence": sentence, "predicted_class": predicted_class}
        for class_name, probability in zip(class_names, probabilities):
            result[class_name] = probability
        
        results.append(result)

    # Export to CSV
    with open('classification_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sentence', 'predicted_class'] + class_names
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print("Results exported to classification_results.csv")
else:
    print("Failed to load model or tokenizer.")

# After saving the CSV
absolute_path = os.path.abspath('classification_results.csv')
print(f"Results exported to {absolute_path}")