'''Extracts character-level context features and prepares numerical input matrices.'''

import numpy as np
import sklearn.feature_extraction

def _remove_line_ending(line):
    # Remove newline characters while preserving all other whitespace.
    return line.rstrip("\r\n")


def _extract_character_features(line, eligible_chars, window):
    # Pad the line so characters near its edges still have a full context window.
    padded_line = (
        ["<PAD>"] * window
        + list(line)
        + ["<PAD>"] * window
    )

    features = []

    for i, char in enumerate(line):
        # Only create training examples for characters that can receive diacritics.
        if char not in eligible_chars:
            continue

        center = i + window

        # Store the characters surrounding the current character as separate features.
        character_features = {
            f"char_{relative_offset}" : padded_line[center + relative_offset]
            for relative_offset in range(-window, window + 1)
        }

        # Keep the original position for later text reconstruction.
        features.append((i, character_features))

    return features


def prepare_character_data(data, letters_nodia, target=None, window=4):
    X = []
    y = []
    positions = []

    # Create a set for efficient character membership checks.
    eligible_chars = set(letters_nodia)

    # Keep line endings because they count towards positions in the original text.
    data_lines = data.splitlines(keepends=True)

    if target is not None:
        target_lines = target.splitlines(keepends=True)
    else:
        # During prediction, there are no target characters to collect.
        target_lines = [None] * len(data_lines)

    global_position = 0

    for raw_line, raw_target_line in zip(data_lines, target_lines):
        # Save the original length before removing the newline.
        original_length = len(raw_line)

        # Lowercase the input so uppercase and lowercase share the same features.
        line = _remove_line_ending(raw_line).lower()

        if raw_target_line is not None:
            target_line = _remove_line_ending(raw_target_line).lower()
        else:
            target_line = None

        # Extract context features for every eligible character in this line.
        character_features = _extract_character_features(
            line, eligible_chars, window
        )

        for i, features in character_features:
            X.append(features)

            # Convert the position within this line into a position in the full text.
            positions.append(global_position + i)

            if target_line is not None:
                # Store the correct diacritized character.
                y.append(target_line[i])

        # Advance to the next line, including newline characters in the offset.
        global_position += original_length

    return X, y, positions

def prepare_full_training_data(data, target, letters_nodia, window):
    # Extract character-level training features and labels.
    X, y, _ = prepare_character_data(
        data,
        letters_nodia,
        target,
        window
    )

    # Fit a new vectorizer using the complete training dataset.
    vectorizer = sklearn.feature_extraction.DictVectorizer(
        dtype=np.float32
    )

    X = vectorizer.fit_transform(X)

    # Convert sparse matrix indices to 32-bit integers (or Python will complain).
    X.indices = X.indices.astype(np.int32)
    X.indptr = X.indptr.astype(np.int32)

    return X, y, vectorizer