
import json
import logging
from collections import Counter
from typing import Any

import tiktoken


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# TOKEN COUNTER
# --------------------------------------------------

class TokenCounter:

    def __init__(self, encoding_name="cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text):
        return len(self.encoding.encode(text))


# --------------------------------------------------
# METADATA VALIDATION HELPERS
# --------------------------------------------------

def is_valid_metadata_value(value):
    """
    Check whether a value is suitable for common
    vector database metadata fields.

    Supports:
        str, int, float, bool
        Lists of strings, integers, floats or booleans

    Nested dictionaries and lists of dictionaries
    are not considered vector-database compatible.
    """

    if isinstance(value, (str, int, float, bool)):
        return True

    if isinstance(value, list):
        return all(
            isinstance(item, (str, int, float, bool))
            for item in value
        )

    return False


def is_json_serializable(value):

    try:
        json.dumps(value, default=None)
        return True

    except (TypeError, ValueError):
        return False


# --------------------------------------------------
# CHECK INPUT STRUCTURE
# --------------------------------------------------

def validate_input_structure(chunks):

    errors = []

    if not isinstance(chunks, list):
        errors.append("Input must be a list of chunks.")
        return errors

    if not chunks:
        errors.append("Input chunk list is empty.")

    return errors


# --------------------------------------------------
# VALIDATE INDIVIDUAL CHUNK
# --------------------------------------------------

def validate_single_chunk(
    chunk,
    index,
    token_counter=None,
    max_chunk_size=None,
    required_metadata_fields=None
):

    errors = []
    warnings = []

    # Check dictionary structure
    if not isinstance(chunk, dict):

        errors.append(
            f"Chunk {index}: Expected a dictionary."
        )

        return errors, warnings

    # Check text
    text = chunk.get("text")

    if not isinstance(text, str):

        errors.append(
            f"Chunk {index}: Text must be a string."
        )

        return errors, warnings

    if not text.strip():

        errors.append(
            f"Chunk {index}: Text is empty."
        )

    # Check metadata
    metadata = chunk.get("metadata")

    if not isinstance(metadata, dict):

        errors.append(
            f"Chunk {index}: Metadata must be a dictionary."
        )

        return errors, warnings

    # Required metadata fields
    for field in required_metadata_fields or []:

        if field not in metadata or metadata[field] is None:
            errors.append(
                f"Chunk {index}: Missing metadata field '{field}'."
            )

    # Check token count
    actual_token_count = None

    if token_counter is not None and isinstance(text, str):

        actual_token_count = token_counter.count(text)

        if max_chunk_size is not None:

            if actual_token_count > max_chunk_size:

                errors.append(
                    f"Chunk {index}: Token limit exceeded. "
                    f"Actual={actual_token_count}, "
                    f"Maximum={max_chunk_size}."
                )

        # Check stored token count
        stored_count = metadata.get("chunk_token_count")

        if stored_count is not None:

            if not isinstance(stored_count, int) or isinstance(stored_count, bool):
                errors.append(
                    f"Chunk {index}: chunk_token_count must be an integer."
                )

            elif stored_count != actual_token_count:

                warnings.append(
                    f"Chunk {index}: Stored token count "
                    f"({stored_count}) differs from actual "
                    f"count ({actual_token_count})."
                )

    # Check recorded chunk size
    recorded_limit = metadata.get(
        "max_chunk_size",
        metadata.get("chunk_size")
    )

    if recorded_limit is not None:

        if (
            not isinstance(recorded_limit, (int, float))
            or isinstance(recorded_limit, bool)
            or recorded_limit <= 0
        ):

            errors.append(
                f"Chunk {index}: Invalid recorded chunk size."
            )

    # Check chunk ID
    chunk_id = metadata.get("chunk_id")

    if chunk_id is not None and not isinstance(chunk_id, str):

        errors.append(
            f"Chunk {index}: chunk_id must be a string."
        )

    # Check source references
    has_source_reference = any(
        field in metadata
        for field in [
            "source_block_index",
            "source_blocks",
            "block_number",
            "page_number",
            "row_number",
            "json_path"
        ]
    )

    if not has_source_reference:

        warnings.append(
            f"Chunk {index}: No source location reference found."
        )

    # Check source_blocks structure
    source_blocks = metadata.get("source_blocks")

    if source_blocks is not None:

        if not isinstance(source_blocks, list):

            errors.append(
                f"Chunk {index}: source_blocks must be a list."
            )

        elif not all(
            isinstance(item, dict)
            for item in source_blocks
        ):

            errors.append(
                f"Chunk {index}: source_blocks must contain dictionaries."
            )

    # Check source offsets
    start = metadata.get("start_index")
    end = metadata.get("end_index")

    if start is not None and (
        not isinstance(start, int)
        or isinstance(start, bool)
        or start < 0
    ):

        errors.append(
            f"Chunk {index}: Invalid start_index."
        )

    if end is not None and (
        not isinstance(end, int)
        or isinstance(end, bool)
        or end < 0
    ):

        errors.append(
            f"Chunk {index}: Invalid end_index."
        )

    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and end < start
    ):

        errors.append(
            f"Chunk {index}: end_index is before start_index."
        )

    # Check JSON serializability
    if not is_json_serializable(metadata):

        warnings.append(
            f"Chunk {index}: Metadata is not JSON serializable."
        )

    # Check vector database compatibility
    for key, value in metadata.items():

        if not is_valid_metadata_value(value):

            warnings.append(
                f"Chunk {index}: Metadata field '{key}' "
                "may not be directly supported by common "
                "vector databases. Consider flattening it "
                "or storing it separately."
            )

    return errors, warnings


# --------------------------------------------------
# CHECK DUPLICATE CHUNK IDS
# --------------------------------------------------

def check_duplicate_ids(chunks):

    errors = []
    seen_ids = set()

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            continue

        metadata = chunk.get("metadata")

        if not isinstance(metadata, dict):
            continue

        chunk_id = metadata.get("chunk_id")

        if chunk_id is None:
            continue

        if chunk_id in seen_ids:

            errors.append(
                f"Chunk {index}: Duplicate chunk_id '{chunk_id}'."
            )

        else:
            seen_ids.add(chunk_id)

    return errors


# --------------------------------------------------
# CHECK DUPLICATE TEXT
# --------------------------------------------------

def check_duplicate_text(chunks):

    warnings = []
    seen_text = {}

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            continue

        text = chunk.get("text")

        if not isinstance(text, str) or not text.strip():
            continue

        normalized_text = " ".join(text.split())

        if normalized_text in seen_text:

            warnings.append(
                f"Chunk {index}: Duplicate text found "
                f"with chunk {seen_text[normalized_text]}."
            )

        else:
            seen_text[normalized_text] = index

    return warnings


# --------------------------------------------------
# CHECK CHUNK NUMBER SEQUENCE
# --------------------------------------------------

def check_chunk_numbers(chunks):

    warnings = []
    numbers = []

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            continue

        metadata = chunk.get("metadata")

        if not isinstance(metadata, dict):
            continue

        number = metadata.get("chunk_number")

        if number is None:
            continue

        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number <= 0
        ):

            warnings.append(
                f"Chunk {index}: Invalid chunk_number."
            )

            continue

        numbers.append(number)

    duplicates = [
        number
        for number, count in Counter(numbers).items()
        if count > 1
    ]

    if duplicates:

        warnings.append(
            f"Duplicate chunk numbers found: {duplicates}"
        )

    return warnings


# --------------------------------------------------
# CALCULATE CHUNK STATISTICS
# --------------------------------------------------

def calculate_statistics(
    chunks,
    token_counter=None
):

    token_counts = []
    text_lengths = []

    file_types = Counter()
    chunking_methods = Counter()

    valid_text_chunks = 0

    for chunk in chunks:

        if not isinstance(chunk, dict):
            continue

        text = chunk.get("text")

        if not isinstance(text, str):
            continue

        metadata = chunk.get("metadata", {})

        if not isinstance(metadata, dict):
            continue

        valid_text_chunks += 1

        text_lengths.append(len(text))

        if token_counter:
            token_counts.append(
                token_counter.count(text)
            )

        file_type = metadata.get("file_type")

        if file_type:
            file_types[file_type] += 1

        method = metadata.get("chunking_method")

        if method:
            chunking_methods[method] += 1

    statistics = {

        "total_chunks": len(chunks),

        "valid_text_chunks": valid_text_chunks,

        "file_type_distribution": dict(file_types),

        "chunking_method_distribution": dict(chunking_methods),

        "average_characters": (
            round(sum(text_lengths) / len(text_lengths), 2)
            if text_lengths else 0
        )

    }

    if token_counts:

        statistics.update({

            "total_tokens": sum(token_counts),

            "average_tokens": round(
                sum(token_counts) / len(token_counts),
                2
            ),

            "minimum_tokens": min(token_counts),

            "maximum_tokens": max(token_counts)

        })

    return statistics


# --------------------------------------------------
# FINAL VALIDATION FUNCTION
# --------------------------------------------------

def final_function(
    chunks,
    chunk_size=None,
    encoding_name="cl100k_base",
    required_metadata_fields=None,
    check_duplicates=True,
    check_token_count=True
):
    """
    Validate chunks generated by any chunking method.

    Returns:
        {
            "is_valid": bool,
            "valid_chunks": [...],
            "errors": [...],
            "warnings": [...],
            "statistics": {...}
        }
    """

    errors = []
    warnings = []

    # Validate input
    structure_errors = validate_input_structure(chunks)

    if structure_errors:

        return {
            "is_valid": False,
            "valid_chunks": [],
            "errors": structure_errors,
            "warnings": [],
            "statistics": {
                "total_chunks": 0
            }
        }

    # Initialize tokenizer
    token_counter = None

    if check_token_count:

        try:
            token_counter = TokenCounter(encoding_name)

        except Exception as exc:

            errors.append(
                f"Unable to initialize tokenizer: {exc}"
            )

            return {
                "is_valid": False,
                "valid_chunks": [],
                "errors": errors,
                "warnings": warnings,
                "statistics": {
                    "total_chunks": len(chunks)
                }
            }

    # Validate individual chunks
    valid_chunks = []

    for index, chunk in enumerate(chunks):

        chunk_errors, chunk_warnings = validate_single_chunk(
            chunk=chunk,
            index=index,
            token_counter=token_counter,
            max_chunk_size=chunk_size,
            required_metadata_fields=required_metadata_fields
        )

        errors.extend(chunk_errors)
        warnings.extend(chunk_warnings)

        if not chunk_errors:
            valid_chunks.append(chunk)

    # Cross-chunk validation
    if check_duplicates:

        errors.extend(
            check_duplicate_ids(chunks)
        )

        warnings.extend(
            check_duplicate_text(chunks)
        )

        warnings.extend(
            check_chunk_numbers(chunks)
        )

    # Statistics
    statistics = calculate_statistics(
        chunks=chunks,
        token_counter=token_counter
    )

    statistics["invalid_chunks"] = (
        len(chunks) - len(valid_chunks)
    )

    statistics["error_count"] = len(errors)
    statistics["warning_count"] = len(warnings)

    is_valid = len(errors) == 0

    logger.info(
        "Validation completed: %d chunks, %d errors, %d warnings.",
        len(chunks),
        len(errors),
        len(warnings)
    )

    return {

        "is_valid": is_valid,

        "valid_chunks": valid_chunks,

        "errors": errors,

        "warnings": warnings,

        "statistics": statistics

    }


# --------------------------------------------------
# EXECUTION EXAMPLE
# --------------------------------------------------

if __name__ == "__main__":

    sample_chunks = [

        {
            "text": "Machine learning is a subset of AI.",

            "metadata": {
                "chunk_id": "chunk_001",
                "chunk_number": 1,
                "document_id": "doc_001",
                "document_name": "AI.txt",
                "file_type": "txt",
                "chunking_method": "recursive_character",
                "chunk_token_count": 9,
                "source_block_index": 0
            }
        },

        {
            "text": "Deep learning uses neural networks.",

            "metadata": {
                "chunk_id": "chunk_002",
                "chunk_number": 2,
                "document_id": "doc_001",
                "document_name": "AI.txt",
                "file_type": "txt",
                "chunking_method": "recursive_character",
                "chunk_token_count": 8,
                "source_block_index": 1
            }
        }

    ]

    result = final_function(
        chunks=sample_chunks,
        chunk_size=500,
        required_metadata_fields=[
            "chunk_id",
            "document_id",
            "file_type",
            "chunking_method"
        ]
    )
    with open("chunking/validation_output.json", "w") as f:
        json.dump(result, f, indent=4)
    print("\nVALID:", result["is_valid"])

    print("\nERRORS:")
    for error in result["errors"]:
        print("-", error)

    print("\nWARNINGS:")
    for warning in result["warnings"]:
        print("-", warning)

    print("\nSTATISTICS:")
    print(result["statistics"])
