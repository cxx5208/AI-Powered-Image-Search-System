import json
import os
import logging  # Add this line to import logging


VECTOR_STORE_PATH = "data/vectors/image_vectors.json"

def load_vectors():
    """
    Load vectors from JSON storage.
    """
    if os.path.exists(VECTOR_STORE_PATH):
        with open(VECTOR_STORE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_vectors(vectors):
    """
    Save vectors to JSON storage.
    """
    with open(VECTOR_STORE_PATH, "w") as f:
        json.dump(vectors, f)

def store_image_vector(image_name, objects):
    """
    Store an image's detected objects and their labels in vector storage.
    """
    vectors = load_vectors()

    # Collect all labels detected in the image
    detected_labels = set(obj['label'] for obj in objects if isinstance(obj, dict))

    vectors[image_name] = {
        'objects': objects,  # List of detected objects
        'labels': list(detected_labels)  # Store unique labels as a list
    }

    save_vectors(vectors)


def get_all_labels():
    """
    Retrieve all unique labels from the vector storage.
    """
    vectors = load_vectors()
    all_labels = set()

    for image_data in vectors.values():
        if isinstance(image_data, dict):  # Ensure image_data is a dictionary
            all_labels.update(image_data.get('labels', []))
        else:
            logging.warning(f"Unexpected format for image data: {image_data}")

    return sorted(list(all_labels))



def query_faiss_index(object_names, confidence_threshold=0.5):
    """
    Query the vector database to find images containing all specified objects with a minimum confidence score.
    """
    vectors = load_vectors()
    matching_images = []

    for image_name, image_data in vectors.items():
        if isinstance(image_data, dict):
            labels = image_data.get('labels', [])
            # Check if all selected object names are present in this image's labels
            if all(object_name.lower() in [label.lower() for label in labels] for object_name in object_names):
                matching_images.append(image_name)

    return matching_images



