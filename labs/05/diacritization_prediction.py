'''Generates diacritized text, reconstructs predictions, and evaluates their accuracy.'''

from diacritization_features import prepare_character_data

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
        window
    ):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.letters_nodia = letters_nodia
        self.letters_dia = letters_dia
        self.window = window

        self.character_variants = _build_character_variants(
            letters_nodia,
            letters_dia
        )

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

        # Reconstruct and return the complete text.
        return _reconstruct_text(data, predictions, positions)