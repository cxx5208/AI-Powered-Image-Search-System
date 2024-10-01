import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np

# Add this LABEL_MAP (you may need to extend it to match all COCO classes)
LABEL_MAP = {
    0: "N/A",
    1: "Person",
    2: "Bicycle",
    3: "Car",
    4: "Motorcycle",
    5: "Airplane",
    6: "Bus",
    7: "Train",
    8: "Truck",
    9: "Boat",
    10: "Traffic Light",
    11: "Fire Hydrant",
    12: "Stop Sign",
    13: "Parking Meter",
    14: "Bench",
    15: "Bird",
    16: "Cat",
    17: "Dog",
    18: "Horse",
    19: "Sheep",
    20: "Cow",
    21: "Elephant",
    22: "Bear",
    23: "Zebra",
    24: "Giraffe",
    25: "Backpack",
    26: "Umbrella",
    27: "Handbag",
    28: "Tie",
    29: "Suitcase",
    30: "Frisbee",
    31: "Skis",
    32: "Snowboard",
    33: "Sports Ball",
    34: "Kite",
    35: "Baseball Bat",
    36: "Baseball Glove",
    37: "Skateboard",
    38: "Surfboard",
    39: "Tennis Racket",
    40: "Bottle",
    41: "Wine Glass",
    42: "Cup",
    43: "Fork",
    44: "Knife",
    45: "Spoon",
    46: "Bowl",
    47: "Banana",
    48: "Apple",
    49: "Sandwich",
    50: "Orange",
    51: "Broccoli",
    52: "Carrot",
    53: "Hot Dog",
    54: "Pizza",
    55: "Donut",
    56: "Cake",
    57: "Chair",
    58: "Couch",
    59: "Potted Plant",
    60: "Bed",
    61: "Dining Table",
    62: "Toilet",
    63: "TV",
    64: "Laptop",
    65: "Mouse",
    66: "Remote",
    67: "Keyboard",
    68: "Cell Phone",
    69: "Microwave",
    70: "Oven",
    71: "Toaster",
    72: "Sink",
    73: "Refrigerator",
    74: "Book",
    75: "Clock",
    76: "Vase",
    77: "Scissors",
    78: "Teddy Bear",
    79: "Hair Drier",
    80: "Toothbrush"
}

def detect_objects(image_path):
    """
    Perform object detection using Hugging Face's DETR model.
    """
    try:
        # Load the image
        image = Image.open(image_path)
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Load the model and processor
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        # Prepare image for the model
        inputs = processor(images=image, return_tensors="pt")

        # Perform inference
        outputs = model(**inputs)

        # Process the output, getting bounding boxes and labels
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

        # Extract bounding boxes, labels, and scores
        bounding_boxes = results["boxes"].detach().cpu().numpy()  # (x_min, y_min, x_max, y_max)
        labels = results["labels"].detach().cpu().numpy()  # Label indices
        scores = results["scores"].detach().cpu().numpy()  # Confidence scores

        # Convert numeric labels to human-readable names
        human_readable_labels = [LABEL_MAP[int(label)] for label in labels]

        return bounding_boxes, human_readable_labels, scores

    except Exception as e:
        raise RuntimeError(f"Failed to process object detection: {str(e)}")
