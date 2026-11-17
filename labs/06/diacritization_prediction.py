'''Generates diacritized text, reconstructs predictions, and evaluates their accuracy.'''

import math
import re

from diacritization_features import prepare_character_data
from diacritization_dictionary import Dictionary

def _build_character_variants(letters_nodia, letters_dia):
    # Every eligible character can remain unchanged or receive a diacritic.
    variants = {
        char: {char} for char in set(letters_nodia)
    }

    # Add all possible diacritic variants for each original character.
    for base, diacritic in zip(letters_nodia, letters_dia):
        variants[base].add(diacritic)

    return variants


def _predict_valid_characters(classifier, X, data, positions, variants):
    # Obtain prediction scores
    scores = classifier.predict_proba(X)

    # Map each output character to its corresponding score column
    class_indices = {
        char: i for i, char in enumerate(classifier.classes_)
    }

    predictions = []

    for row, position in enumerate(positions):
        # Find the original character at this position.
        original_char = data[position].lower()

        # Only allow the original character and its valid diacritic variants
        allowed_variants = [
            char for char in variants[original_char]
            if char in class_indices
        ]

        # Select the allowed character with the highest classifier score
        prediction = max(
            allowed_variants,
            key=lambda char: scores[row, class_indices[char]]
        )

        predictions.append(prediction)

    return predictions


def _reconstruct_text(data, predictions, positions):
    # Start with the original text, preserving punctuation and whitespace.
    reconstructed = list(data)

    for position, prediction in zip(positions, predictions):
        # Restore uppercase if the original character was uppercase.
        if data[position].isupper():
            prediction = prediction.upper()

        reconstructed[position] = prediction

    return "".join(reconstructed)

class Diacritizer:
    def __init__(
        self,
        classifier,
        vectorizer,
        letters_nodia,
        letters_dia,
        window,
        dictionary: Dictionary
    ):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.letters_nodia = letters_nodia
        self.letters_dia = letters_dia
        self.window = window
        self.dictionary = dictionary

        self.character_variants = _build_character_variants(
            letters_nodia,
            letters_dia
        )

    def _score_dictionary_candidate(
        self,
        candidate,
        word_start,
        word_positions,
        position_to_row,
        prediction_scores,
        class_indices
    ):
        candidate_score = 0.0
        # Minimum probability used to prevent log(0) during candidate scoring.
        min_char_probability = 1e-12

        # Calculate the candidate's total log-probability across its characters.
        for character_position in word_positions:
            character = candidate[character_position - word_start]
            class_index = class_indices.get(character)

            # Reject candidates containing characters unknown to the classifier.
            if class_index is None:
                return None

            prediction_row = position_to_row[character_position]
            character_probability = prediction_scores[prediction_row, class_index]

            # Use a probability floor to avoid taking the logarithm of zero.
            candidate_score += math.log(
                max(
                    float(character_probability),
                    min_char_probability
                )
            )

        return candidate_score


    def _get_best_dictionary_candidate(
        self,
        candidates,
        word_start,
        word_end,
        word_positions,
        position_to_row,
        prediction_scores,
        class_indices
    ):
        best_candidate = None
        best_score = -math.inf

        # Evaluate every dictionary variant and retain the highest-scoring one.
        for candidate in candidates:
            candidate = candidate.lower()

            # Only consider candidates with the same length as the original word.
            if len(candidate) != (word_end - word_start):
                continue

            candidate_score = self._score_dictionary_candidate(
                candidate,
                word_start,
                word_positions,
                position_to_row,
                prediction_scores,
                class_indices
            )

            # Ignore invalid candidates and update the best match when appropriate.
            if (candidate_score is not None) and (candidate_score > best_score):
                best_candidate = candidate
                best_score = candidate_score

        return best_candidate


    def _apply_dictionary(
        self,
        input_text,
        predictions,
        character_positions,
        feature_matrix
    ):
        # Obtain the classifier's probability distribution for each character.
        prediction_scores = self.classifier.predict_proba(feature_matrix)

        # Map each character to its corresponding classifier probability column.
        class_indices = {
            character: index
            for index, character in enumerate(self.classifier.classes_)
        }

        # Map original text positions to their corresponding prediction rows.
        position_to_row = {
            character_position: prediction_row
            for prediction_row, character_position in enumerate(character_positions)
        }

        # Work on a copy so the original predictions remain unchanged.
        corrected_predictions = list(predictions)

        # Find individual words (exclude punctuation or whitespace).
        word_pattern = r"[^\W\d_]+"
        for word_match in re.finditer(word_pattern, input_text, flags=re.UNICODE):
            word_start, word_end = word_match.span()

            # Retrieve possible diacritized variants from the dictionary.
            dictionary_candidates = self.dictionary.variants.get(
                word_match.group().lower()
            )
            if not dictionary_candidates: continue

            # Keep only character positions for which the classifier has predictions.
            word_positions = [
                character_position
                for character_position in range(word_start, word_end)
                if character_position in position_to_row
            ]
            if not word_positions: continue

            # Select the dictionary variant that best matches the classifier's probabilities.
            best_candidate = self._get_best_dictionary_candidate(
                dictionary_candidates,
                word_start,
                word_end,
                word_positions,
                position_to_row,
                prediction_scores,
                class_indices
            )
            if best_candidate is None: continue

            # Apply the selected variant while preserving the original capitalization.
            for character_position in word_positions:
                character = best_candidate[character_position - word_start]

                if input_text[character_position].isupper():
                    character = character.upper()

                prediction_row = position_to_row[character_position]
                corrected_predictions[prediction_row] = character

        return corrected_predictions

    def predict(self, data):
        # Extract character features and remember their original positions.
        X, _, positions = prepare_character_data(
            data,
            self.letters_nodia,
            window=self.window
        )

        # Apply the same fitted vectorizer used during training.
        X = self.vectorizer.transform(X)

        # Ensure sparse matrix indices use 32-bit integers.
        X.indices = X.indices.astype("int32")
        X.indptr = X.indptr.astype("int32")

        # Predict valid character variants.
        predictions = _predict_valid_characters(
            self.classifier,
            X,
            data,
            positions,
            self.character_variants
        )

        predictions = self._apply_dictionary(
            input_text=data,
            predictions=predictions,
            character_positions=positions,
            feature_matrix=X
        )

        # Reconstruct and return the complete text.
        return _reconstruct_text(data, predictions, positions)
