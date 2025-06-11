import os
import re
import json
from typing import Dict, List, Set
from comet import download_model, load_from_checkpoint

# Define constants for training and validation data directories
TRAIN_DATA_DIR = './data/semeval.train.v2-e0d1c28b78c8dd4969d25eea5d3bc9cc/semeval/train'
VALIDATION_DATA_DIR = './data/semeval.validation.v2-889a1492ba6c3791baa8f4224bc8e685/validation'

# Constants for COMET model configuration
COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"
COMET_NUM_GPUS = 1
COMET_BATCH_SIZE = 32

# List of entity types to be evaluated.
# Used to filter the evaluation to a specific entity type.
ENTITY_TYPES = [
    "Musical work",
    "Artwork",
    "Food",
    "Animal",
    "Plant",
    "Book",
    "Book series",
    "Fictional entity",
    "Landmark",
    "Movie",
    "Place of worship",
    "Natural place",
    "TV series",
    "Person",
]

def get_language_name(short_code):
    lang_map = {
        'ar': 'Arabic',
        'zh': 'Chinese (Traditional)',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'es': 'Spanish',
        'th': 'Thai',
        'tr': 'Turkish',
        'en': 'English',
        # Add more as needed
    }
    return lang_map.get(short_code, short_code)

def download_comet_model():
    comet_model_path = download_model(COMET_MODEL_NAME)
    comet_model = load_from_checkpoint(comet_model_path)
    return comet_model

def calculate_comet_scores(model, references_path, predictions_path):
    references = _load_jsonl_data(references_path)
    predictions = _load_jsonl_data(predictions_path)
    ids = set(references.keys()) & set(predictions.keys())
    num_missing_predictions = len(predictions) - len(ids)

    if num_missing_predictions > 0:
        print(f"Missing predictions for {num_missing_predictions} references")
    else:
        print("All references have a corresponding prediction")

    instance_ids = {}
    instances = []
    current_index = 0

    for id in sorted(list(ids)):
        reference = references[id]
        prediction = predictions[id]

        for target in reference["targets"]:
            instances.append(
                {
                    "src": reference["source"],
                    "ref": target["translation"],
                    "mt": prediction["prediction"],
                }
            )

        instance_ids[id] = [current_index, current_index + len(reference["targets"])]
        current_index += len(reference["targets"])
    print(f"Created {len(instances)} instances")

    outputs = model.predict(instances, batch_size=COMET_BATCH_SIZE, gpus=COMET_NUM_GPUS)

    scores = outputs.scores
    max_scores = []

    for id, indices in instance_ids.items():
        max_score = max(scores[indices[0] : indices[1]])
        max_scores.append(max_score)

    system_score = sum(max_scores) / (len(max_scores) + num_missing_predictions)

    print(f"Average COMET score: {100.*system_score:.2f}")

    return system_score

def calculate_meta_score(references_path, predictions_path, verbose=False):
    reference_data = _load_references(references_path, ENTITY_TYPES)
    mentions = _get_mentions_from_references(reference_data)
    assert len(mentions) == len(reference_data)
    print(f"Loaded {len(reference_data)} instances.")
    prediction_data = _load_predictions(predictions_path)
    print(f"Loaded {len(prediction_data)} predictions.")
    entity_name_translation_accuracy = _compute_entity_name_translation_accuracy(
        prediction_data,
        mentions,
        verbose=verbose,
    )
    correct_instances = entity_name_translation_accuracy["correct"]
    total_instances = entity_name_translation_accuracy["total"]
    accuracy = entity_name_translation_accuracy["accuracy"] * 100.0
    return correct_instances, total_instances, accuracy

def _load_jsonl_data(jsonl_path):
    json_data = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            json_data[data['id']] = data
    return json_data

def _load_references(input_path: str, entity_types: List[str]) -> List[dict]:
    """
    Load data from the input file (JSONL) and return a list of dictionaries, one for each instance in the dataset.

    Args:
        input_path (str): Path to the input file.
        entity_types (List[str]): List of entity types to filter the evaluation.

    Returns:
        List[dict]: List of dictionaries, one for each instance in the dataset.
    """
    data = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line_data = json.loads(line)

            # Skip instances with empty target list and log a warning.
            if not line_data["targets"]:
                print(f"Empty target list for instance {line_data['id']}")
                continue

            # Filter the evaluation to the specified entity types if provided.
            if entity_types and not any(
                e in line_data["entity_types"] for e in entity_types
            ):
                continue

            data.append(line_data)

    return data

def _load_predictions(input_path: str) -> Dict[str, str]:
    """
    Load data from the input file (JSONL) and return a dictionary with the instance ID as key and the prediction as value.

    Args:
        input_path (str): Path to the input file.

    Returns:
        Dict[str, str]: Dictionary with the instance ID as key and the prediction as value.
    """
    data = {}

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line_data = json.loads(line)
            prediction = line_data["prediction"]

            # Get the instance ID from a substring of the ID.
            pattern = re.compile(r"Q[0-9]+_[0-9]")
            match = pattern.match(line_data["id"])
            if not match:
                raise ValueError(f"Invalid instance ID: {line_data['id']}")

            instance_id = match.group(0)
            data[instance_id] = prediction

    return data

def _compute_entity_name_translation_accuracy(
    predictions: Dict[str, str],
    mentions: Dict[str, Set[str]],
    verbose: bool = False,
) -> dict:
    """
    Compute the entity name translation accuracy.

    Args:
        predictions (Dict[str, str]): Predictions of the model.
        mentions (Dict[str, Set[str]]): Ground truth entity mentions.
        verbose (bool): Set to True to print every wrong match.

    Returns:
        dict: Dictionary with the following
            - correct: Number of correct matches.
            - total: Total number of instances.
            - accuracy: Accuracy of the model.
    """
    correct, total = 0, 0

    for instance_id, instance_mentions in mentions.items():
        assert instance_mentions, f"No mentions for instance {instance_id}"
        total += 1

        if instance_id not in predictions:
            if verbose:
                print(
                    f"No prediction for instance {instance_id}. Check that this is expected behavior, as it may affect the evaluation."
                )
            continue

        prediction = predictions[instance_id]
        normalized_translation = prediction.casefold()
        entity_match = False

        for mention in instance_mentions:
            normalized_mention = mention.casefold()

            if normalized_mention in normalized_translation:
                correct += 1
                entity_match = True
                break

        if not entity_match and verbose:
            print(f"Prediction: {prediction}")
            print(f"Ground truth mentions: {instance_mentions}")
            print("")

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
    }

def _get_mentions_from_references(data: List[dict]) -> Dict[str, Set[str]]:
    """
    Load the ground truth entity mentions from the data.

    Args:
        data (List[dict]): List of dictionaries, one for each instance in the dataset.

    Returns:
        Dict[str, Set[str]]: Dictionary with the instance ID as key and the set of entity mentions as value.
    """
    mentions = {}

    for instance in data:
        instance_id = instance["id"]
        instance_mentions = set()

        for target in instance["targets"]:
            mention = target["mention"]
            instance_mentions.add(mention)

        mentions[instance_id] = instance_mentions

    return mentions