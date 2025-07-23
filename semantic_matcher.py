# file: semantic_matcher.py
import torch 
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

class SemanticMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initializes the matcher by loading the sentence transformer model."""
        print(f"Loading semantic matching model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✅ Semantic matcher loaded successfully.")

    def extract_keywords(self, text_prompt: str, vocabulary: set) -> List[str]:
        """Extracts known keywords from a text prompt."""
        prompt_words = set(text_prompt.lower().split())
        keywords = [word for word in vocabulary if word in prompt_words]
        print(f"   - Extracted keywords from prompt: {keywords}")
        return keywords

    def find_best_match(self, keywords: List[str], detected_objects: List[Dict]) -> Dict | None:
        """
        Finds the best detected object match for the extracted keywords using cosine similarity.

        Args:
            keywords (List[str]): Keywords extracted from the user prompt.
            detected_objects (List[Dict]): Objects detected by YOLO.

        Returns:
            Dict or None: The dictionary of the best matching object, or None if no match.
        """
        if not detected_objects:
            return None

        detected_labels = [obj['label'] for obj in detected_objects]
        
        # If no specific keywords are found in the prompt, use the whole prompt for matching
        if not keywords:
            print("   - No specific keywords found, using the entire prompt for matching.")
            keywords = [ "car", "vehicle" ] # Default to common terms if prompt is generic

        # Encode all keywords and detected labels into vectors
        keyword_embeddings = self.model.encode(keywords, convert_to_tensor=True)
        label_embeddings = self.model.encode(detected_labels, convert_to_tensor=True)

        # Calculate cosine similarity between each keyword and each detected label
        cosine_scores = util.cos_sim(keyword_embeddings, label_embeddings)

        # Find the detected object that has the highest average score across all keywords
        # This helps prioritize objects that match more of the user's intent
        average_scores = torch.mean(cosine_scores, dim=0)
        best_match_index = torch.argmax(average_scores).item()
        best_match_score = average_scores[best_match_index].item()
        
        print(f"   - Best match is '{detected_labels[best_match_index]}' with a similarity score of {best_match_score:.4f}")

        # You can set a threshold to avoid non-sensical matches
        if best_match_score < 0.3: # Confidence threshold
            print("   - Match score is below threshold. No confident match found.")
            return None

        return detected_objects[best_match_index]