
import csv
import io
import uuid
import logging

from pathlib import Path
from charset_normalizer import from_bytes


# ============================================================
# 1. LOGGING CONFIGURATION
# ============================================================

logger = logging.getLogger("csv_processor")

if not logger.handlers:
    handler = logging.FileHandler("csv_processing.log")

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

    if path.suffix.lower() != ".csv":
        raise ValueError(
            "Expected a .csv file."
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
        "Unable to determine a usable CSV encoding."
    )


# ============================================================
# 4. READ CSV FILE
# ============================================================

def csv_open(file_path):

    path = validate_file(file_path)

    try:
        raw_bytes = path.read_bytes()

        if not raw_bytes:
            logger.warning("Empty CSV file: %s", path.name)
            return "", "unknown"

        encoding = detect_encoding(raw_bytes)

        text = raw_bytes.decode(encoding)

        if "\x00" in text:
            raise ValueError(
                "CSV file contains null characters."
            )

        logger.info(
            "CSV file read using encoding: %s",
            encoding
        )

        return text, encoding

    except Exception:

        logger.exception(
            "CSV reading failed: %s",
            path
        )

        raise


# ============================================================
# 5. DETECT CSV DELIMITER
# ============================================================

def detect_delimiter(text):

    sample = text[:65536]

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t|"
        )

        return dialect

    except csv.Error:

        logger.warning(
            "Could not detect delimiter. Using comma."
        )

        return csv.excel


# ============================================================
# 6. CLEAN AND NORMALIZE HEADERS
# ============================================================

def normalize_headers(headers):

    normalized = []
    used_names = set()

    for index, header in enumerate(headers, start=1):

        name = str(header).strip()

        if not name:
            name = f"column_{index}"

        original_name = name
        counter = 2

        while name in used_names:

            name = f"{original_name}_{counter}"
            counter += 1

        used_names.add(name)

        normalized.append(name)

    return normalized


# ============================================================
# 7. HELPER: CREATE TEXT FROM A CSV ROW
# ============================================================

def row_to_text(headers, row):

    parts = []

    for column, value in zip(headers, row):

        value = str(value).strip()

        parts.append(
            f"{column}: {value if value else '[Empty]'}"
        )

    return "\n".join(parts)


# ============================================================
# 8. CSV TO STRUCTURED BLOCKS
# ============================================================

def csv_to_blocks(
    text,
    document_id,
    document_name,
    source_path,
    encoding,
    has_header=True
):

    if not text.strip():
        return []

    dialect = detect_delimiter(text)

    stream = io.StringIO(text, newline="")

    reader = csv.reader(
        stream,
        dialect=dialect,
        strict=True
    )

    blocks = []

    try:

        # ----------------------------------------------------
        # Read first row
        # ----------------------------------------------------

        try:
            first_row = next(reader)

        except StopIteration:
            return []

        if not first_row or not any(
            str(value).strip() for value in first_row
        ):
            raise ValueError(
                "CSV file does not contain a valid first row."
            )

        if has_header:

            headers = normalize_headers(first_row)

        else:

            headers = [
                f"column_{i}"
                for i in range(1, len(first_row) + 1)
            ]

        # ----------------------------------------------------
        # Extract data rows
        # ----------------------------------------------------

        block_number = 0
        data_row_number = 0

        def process_row(row, physical_line):

            nonlocal block_number
            nonlocal data_row_number
            nonlocal headers

            if not row or not any(
                str(value).strip() for value in row
            ):
                logger.info(
                    "Empty row skipped near line %s",
                    physical_line
                )
                return

            # Handle rows with more values than headers.

            if len(row) > len(headers):

                for i in range(len(headers), len(row)):

                    headers.append(
                        f"extra_column_{i + 1}"
                    )

                logger.warning(
                    "Extra columns found near line %s",
                    physical_line
                )

            # Handle rows with fewer values than headers.

            if len(row) < len(headers):

                row = row + [
                    ""
                ] * (len(headers) - len(row))

                logger.warning(
                    "Missing values found near line %s",
                    physical_line
                )

            data_row_number += 1

            text_content = row_to_text(
                headers,
                row
            )

            block_number += 1

            metadata = {
                "document_id": document_id,
                "document_name": document_name,
                "source_path": source_path,
                "file_type": "csv",
                "encoding": encoding,
                "block_type": "table_row",
                "block_number": block_number,
                "row_number": data_row_number,
                "source_line_end": physical_line,
                "columns": list(headers),
                "has_header": has_header,
                "delimiter": dialect.delimiter
            }

            blocks.append({
                "text": text_content,
                "metadata": metadata
            })

        # Process first row if there is no header.

        if not has_header:

            process_row(
                first_row,
                reader.line_num
            )

        # Process remaining rows.

        for row in reader:

            process_row(
                row,
                reader.line_num
            )

    except csv.Error as e:

        logger.exception(
            "Malformed CSV data near physical line %s",
            reader.line_num
        )

        raise ValueError(
            f"CSV parsing error near line "
            f"{reader.line_num}: {e}"
        ) from e

    logger.info(
        "Extracted %s data rows",
        len(blocks)
    )

    return blocks


# ============================================================
# 9. VALIDATE BLOCKS
# ============================================================

def validate_blocks(blocks):

    valid_blocks = []

    required_fields = [
        "document_id",
        "document_name",
        "file_type",
        "block_type",
        "block_number",
        "row_number"
    ]

    for block in blocks:

        text = block.get("text")
        metadata = block.get("metadata")

        if not isinstance(text, str):
            logger.warning(
                "Invalid text block skipped."
            )
            continue

        if not text.strip():
            logger.warning(
                "Empty text block skipped."
            )
            continue

        if not isinstance(metadata, dict):
            logger.warning(
                "Invalid metadata skipped."
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
# 10. FINAL CSV PROCESSING FUNCTION
# ============================================================

def final_function(file_path, has_header=True):

    path = validate_file(file_path)

    document_id = str(uuid.uuid4())

    document_name = path.name

    source_path = str(path.resolve())

    logger.info(
        "Starting CSV processing: %s",
        document_name
    )

    # Step 1: Read CSV file.

    text, encoding = csv_open(path)

    if not text.strip():
        return []

    # Step 2: Convert CSV rows into structured blocks.

    blocks = csv_to_blocks(
        text=text,
        document_id=document_id,
        document_name=document_name,
        source_path=source_path,
        encoding=encoding,
        has_header=has_header
    )

    # Step 3: Validate blocks.

    output = validate_blocks(blocks)

    logger.info(
        "CSV processing completed: %s, %s blocks",
        document_name,
        len(output)
    )

    return output


# ============================================================
# 11. EXECUTION
# ============================================================

if __name__ == "__main__":

    file_path = r"C:\Users\yashw\Desktop\RAG\basic-data.csv"

    output = final_function(
        file_path,
        has_header=True
    )

    print("Total blocks:", len(output))

    for block in output[:10]:

        print("\nTEXT:")
        print(block["text"])

        print("\nMETADATA:")
        print(block["metadata"])
