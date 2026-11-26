
import json
import uuid
import logging

from pathlib import Path
from charset_normalizer import from_bytes


# ============================================================
# 1. LOGGING CONFIGURATION
# ============================================================

logger = logging.getLogger("json_processor")

if not logger.handlers:

    handler = logging.FileHandler(
        "json_processing.log",
        encoding="utf-8"
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
    )

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ============================================================
# 2. FILE VALIDATION
# ============================================================

def validate_file(file_path):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            "Provided path is not a file."
        )

    if path.suffix.lower() != ".json":
        raise ValueError(
            "Expected a .json file."
        )

    return path


# ============================================================
# 3. ENCODING DETECTION
# ============================================================

def detect_encoding(raw_bytes):

    bom_encodings = [
        (b"\x00\x00\xfe\xff", "utf-32"),
        (b"\xff\xfe\x00\x00", "utf-32"),
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xfe\xff", "utf-16"),
        (b"\xff\xfe", "utf-16")
    ]

    for bom, encoding in bom_encodings:

        if raw_bytes.startswith(bom):
            return encoding

    try:
        raw_bytes.decode("utf-8")
        return "utf-8"

    except UnicodeDecodeError:
        pass

    detected = from_bytes(raw_bytes).best()

    if detected is not None and detected.encoding:
        return detected.encoding

    raise UnicodeError(
        "Unable to determine a usable JSON encoding."
    )


# ============================================================
# 4. READ AND PARSE JSON FILE
# ============================================================

def json_open(file_path):

    path = Path(file_path)

    try:

        raw_bytes = path.read_bytes()

        if not raw_bytes:
            logger.warning(
                "Empty JSON file: %s",
                path.name
            )
            return None, "unknown"

        encoding = detect_encoding(raw_bytes)

        text = raw_bytes.decode(encoding)

        # Reject non-standard JSON constants such as NaN
        # and Infinity.

        def reject_constant(value):
            raise ValueError(
                f"Invalid JSON constant: {value}"
            )

        data = json.loads(
            text,
            parse_constant=reject_constant
        )

        logger.info(
            "JSON parsed successfully using %s",
            encoding
        )

        return data, encoding

    except json.JSONDecodeError as e:

        logger.exception(
            "Invalid JSON syntax."
        )

        raise ValueError(
            f"Invalid JSON at line {e.lineno}, "
            f"column {e.colno}: {e.msg}"
        ) from e

    except Exception:

        logger.exception(
            "JSON reading failed: %s",
            path
        )

        raise


# ============================================================
# 5. HELPER: JSON POINTER ESCAPING
# ============================================================

def escape_pointer(value):

    return str(value).replace(
        "~", "~0"
    ).replace(
        "/", "~1"
    )


def build_pointer(parent, key):

    return (
        parent + "/" + escape_pointer(key)
    )


# ============================================================
# 6. HELPER: FORMAT JSON VALUES
# ============================================================

def format_value(value):

    if value is None:
        return "null"

    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False
        )

    return str(value)


# ============================================================
# 7. FLATTEN NESTED JSON OBJECTS
# ============================================================

def flatten_fields(data, prefix=""):

    fields = []

    if isinstance(data, dict):

        for key, value in data.items():

            current_key = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            if isinstance(value, dict):

                nested_fields = flatten_fields(
                    value,
                    current_key
                )

                fields.extend(nested_fields)

            elif isinstance(value, list):

                # Preserve primitive arrays together.
                # Arrays containing objects are extracted
                # separately by the recursive record parser.

                if not any(
                    isinstance(item, dict)
                    for item in value
                ):

                    fields.append(
                        f"{current_key}: "
                        f"{format_value(value)}"
                    )

                else:

                    # Preserve scalar values in mixed arrays.
                    scalar_items = [
                        item for item in value
                        if not isinstance(item, dict)
                    ]

                    if scalar_items:

                        fields.append(
                            f"{current_key} (scalar items): "
                            f"{format_value(scalar_items)}"
                        )

            else:

                fields.append(
                    f"{current_key}: {format_value(value)}"
                )

    else:

        fields.append(
            f"{prefix}: {format_value(data)}"
            if prefix else format_value(data)
        )

    return fields


# ============================================================
# 8. CREATE OUTPUT BLOCK
# ============================================================

def create_block(
    text,
    block_number,
    document_id,
    document_name,
    source_path,
    encoding,
    json_path,
    parent_path,
    structure_type,
    top_level_type,
    top_level_keys,
    context_fields=None,
    record_index=None
):

    metadata = {

        "document_id": document_id,
        "document_name": document_name,
        "source_path": source_path,

        "file_type": "json",
        "encoding": encoding,

        "block_type": "json_record",
        "block_number": block_number,

        "structure_type": structure_type,

        "json_path": json_path,
        "parent_path": parent_path,

        "record_index": record_index,

        "top_level_type": top_level_type,
        "top_level_keys": top_level_keys,

        "context_fields": context_fields or []
    }

    return {
        "text": text,
        "metadata": metadata
    }


# ============================================================
# 9. JSON TO STRUCTURED BLOCKS
# ============================================================

def json_to_blocks(
    data,
    document_id,
    document_name,
    source_path,
    encoding
):

    blocks = []

    block_number = 0

    top_level_type = (
        "object" if isinstance(data, dict)
        else "array" if isinstance(data, list)
        else "scalar"
    )

    top_level_keys = (
        list(data.keys())
        if isinstance(data, dict)
        else []
    )

    # --------------------------------------------------------
    # Helper: Add a block
    # --------------------------------------------------------

    def add_block(
        fields,
        json_path,
        parent_path,
        structure_type,
        context_fields=None,
        record_index=None
    ):

        nonlocal block_number

        all_fields = []

        if context_fields:
            all_fields.extend(context_fields)

        all_fields.extend(fields)

        if not all_fields:

            if structure_type == "empty_object":
                all_fields.append("[Empty JSON object]")

            elif structure_type == "empty_array":
                all_fields.append("[Empty JSON array]")

            else:
                return

        text = "\n".join(all_fields)

        block_number += 1

        blocks.append(
            create_block(
                text=text,
                block_number=block_number,
                document_id=document_id,
                document_name=document_name,
                source_path=source_path,
                encoding=encoding,
                json_path=json_path,
                parent_path=parent_path,
                structure_type=structure_type,
                top_level_type=top_level_type,
                top_level_keys=top_level_keys,
                context_fields=context_fields,
                record_index=record_index
            )
        )

    # --------------------------------------------------------
    # Recursive extraction
    # --------------------------------------------------------

    def walk(
        value,
        json_path="",
        context_fields=None,
        parent_path="",
        record_index=None
    ):

        if context_fields is None:
            context_fields = []

        # ====================================================
        # OBJECT HANDLING
        # ====================================================

        if isinstance(value, dict):

            record_arrays = {}

            normal_fields = {}

            # Separate arrays containing objects.

            for key, item in value.items():

                if (
                    isinstance(item, list)
                    and any(
                        isinstance(element, dict)
                        for element in item
                    )
                ):

                    record_arrays[key] = item

                else:

                    normal_fields[key] = item

            # Flatten ordinary fields.

            current_fields = flatten_fields(
                normal_fields
            )

            combined_context = (
                context_fields + current_fields
            )

            # If object contains arrays of records,
            # extract each record independently.

            if record_arrays:

                for array_name, array_values in record_arrays.items():

                    array_path = build_pointer(
                        json_path,
                        array_name
                    )

                    for index, item in enumerate(array_values):

                        item_path = build_pointer(
                            array_path,
                            index
                        )

                        if isinstance(item, dict):

                            walk(
                                value=item,
                                json_path=item_path,
                                context_fields=combined_context + [
                                    f"Collection: {array_name}, "
                                    f"record: {index + 1}"
                                ],
                                parent_path=array_path,
                                record_index=index + 1
                            )

                        else:

                            # Handle scalar items in mixed arrays.

                            add_block(
                                fields=[
                                    f"{array_name}: "
                                    f"{format_value(item)}"
                                ],
                                json_path=item_path,
                                parent_path=array_path,
                                structure_type="array_item",
                                context_fields=combined_context,
                                record_index=index + 1
                            )

                return

            # Ordinary object without record arrays.

            add_block(
                fields=current_fields,
                json_path=json_path,
                parent_path=parent_path,
                structure_type=(
                    "object"
                    if current_fields
                    else "empty_object"
                ),
                context_fields=context_fields,
                record_index=record_index
            )

        # ====================================================
        # ARRAY HANDLING
        # ====================================================

        elif isinstance(value, list):

            if not value:

                add_block(
                    fields=["[Empty JSON array]"],
                    json_path=json_path,
                    parent_path=parent_path,
                    structure_type="empty_array",
                    context_fields=context_fields,
                    record_index=record_index
                )

                return

            # Each array item becomes an independent block
            # when it is an object.

            for index, item in enumerate(value):

                item_path = build_pointer(
                    json_path,
                    index
                )

                if isinstance(item, (dict, list)):

                    walk(
                        value=item,
                        json_path=item_path,
                        context_fields=context_fields,
                        parent_path=json_path,
                        record_index=index + 1
                    )

                else:

                    add_block(
                        fields=[
                            f"Item {index + 1}: "
                            f"{format_value(item)}"
                        ],
                        json_path=item_path,
                        parent_path=json_path,
                        structure_type="array_item",
                        context_fields=context_fields,
                        record_index=index + 1
                    )

        # ====================================================
        # SCALAR HANDLING
        # ====================================================

        else:

            add_block(
                fields=[
                    format_value(value)
                ],
                json_path=json_path,
                parent_path=parent_path,
                structure_type="scalar",
                context_fields=context_fields,
                record_index=record_index
            )

    # Start recursive extraction.

    walk(data)

    return blocks


# ============================================================
# 10. VALIDATE OUTPUT BLOCKS
# ============================================================

def validate_blocks(blocks):

    valid_blocks = []

    required_fields = [
        "document_id",
        "document_name",
        "file_type",
        "block_type",
        "block_number",
        "json_path"
    ]

    for block in blocks:

        text = block.get("text")
        metadata = block.get("metadata")

        if not isinstance(text, str):
            logger.warning(
                "Invalid text type. Block skipped."
            )
            continue

        if not text.strip():
            logger.warning(
                "Empty block skipped."
            )
            continue

        if not isinstance(metadata, dict):
            logger.warning(
                "Invalid metadata. Block skipped."
            )
            continue

        if not all(
            field in metadata
            for field in required_fields
        ):

            logger.warning(
                "Required metadata missing."
            )
            continue

        valid_blocks.append(block)

    logger.info(
        "Validated %s out of %s blocks",
        len(valid_blocks),
        len(blocks)
    )

    return valid_blocks


# ============================================================
# 11. FINAL JSON PROCESSING FUNCTION
# ============================================================

def final_function(file_path):

    path = Path(file_path)

    document_id = str(uuid.uuid4())

    document_name = path.name

    source_path = str(path.resolve())

    logger.info(
        "Starting JSON processing: %s",
        document_name
    )

    # Step 1: Read and parse JSON.

    data, encoding = json_open(path)

    if data is None:
        return []

    # Step 2: Extract structured records.

    blocks = json_to_blocks(
        data=data,
        document_id=document_id,
        document_name=document_name,
        source_path=source_path,
        encoding=encoding
    )

    # Step 3: Validate blocks.

    output = validate_blocks(blocks)

    logger.info(
        "JSON processing completed: %s, %s blocks",
        document_name,
        len(output)
    )

    return output


# ============================================================
# 12. EXECUTION
# ============================================================

if __name__ == "__main__":

    file_path = r"C:\Users\yashw\Desktop\RAG\simple-object.json"

    output = final_function(file_path)

    print("Total blocks:", len(output))

    for block in output[:10]:

        print("\nTEXT:")
        print(block["text"])

        print("\nMETADATA:")
        print(block["metadata"])
