# file: utils.py

import json

def load_vehicle_parts(filepath: str) -> set:
    """
    Loads the vehicle parts JSON file and returns a flattened set of all parts.

    Args:
        filepath (str): The path to the vehicle_parts_2.json file.

    Returns:
        set: A set containing all unique vehicle parts for easy lookup.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Flatten all lists of parts from the dictionary into a single set
        all_parts = set()
        for category in data.values():
            for part in category:
                all_parts.add(part.lower())
        
        print(f"✅ Loaded {len(all_parts)} unique vehicle parts from dictionary.")
        return all_parts
    except Exception as e:
        print(f"❌ Error loading vehicle parts dictionary: {e}")
        return set()