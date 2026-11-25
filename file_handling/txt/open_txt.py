
import re
import uuid
import logging
from pathlib import Path
from charset_normalizer import from_bytes


# ============================================================
# 1. LOGGING
# ============================================================

logger = logging.getLogger("txt_processor")

if not logger.handlers:
    handler = logging.FileHandler("txt_processing.log")
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
            "The supplied path is not a file."
        )

    if path.suffix.lower() != ".txt":
        raise ValueError(
            "Only .txt files are supported."
        )

    return path


# ============================================================
# 3. ENCODING DETECTION AND FILE READING
# ============================================================

def detect_encoding(raw_bytes):

    # Check BOM signatures first.
    # UTF-32 must be checked before UTF-16.

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

    # Try UTF-8 first because it is the most common encoding.

    try:
        raw_bytes.decode("utf-8")
        return "utf-8"

    except UnicodeDecodeError:
        pass

    # Use charset-normalizer for other encodings.

    result = from_bytes(raw_bytes).best()

    if result is not None and result.encoding:
        return result.encoding

    # Last-resort fallback for common Western text.
    # This is not guaranteed to recover the original encoding.

    try:
        raw_bytes.decode("cp1252")
        return "cp1252"

    except UnicodeDecodeError as e:
        raise UnicodeError(
            "Unable to determine a usable text encoding."
        ) from e


def txt_open(file_path):

    path = validate_file(file_path)

    try:
        raw_bytes = path.read_bytes()

        if not raw_bytes:
            logger.warning("Empty TXT file: %s", path.name)
            return "", "unknown"

        encoding = detect_encoding(raw_bytes)

        # Decode using the detected encoding.

        text = raw_bytes.decode(encoding)

        # Detect probable binary content.

        if "\x00" in text:
            raise ValueError(
                "File appears to contain binary data, "
                "not ordinary text."
            )

        control_chars = sum(
            1 for char in text
            if ord(char) < 32 and char not in "\n\r\t\f"
        )

        if text and control_chars / len(text) > 0.01:
            raise ValueError(
                "File contains too many control characters "
                "to be treated as a normal text file."
            )

        logger.info(
            "Read %s using encoding %s",
            path.name,
            encoding
        )

        return text, encoding

    except Exception:
        logger.exception(
            "Failed to read TXT file: %s",
            path
        )
        raise


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):

    # Normalize line endings.

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters if any remain.

    text = text.replace("\x00", "")

    # Normalize horizontal whitespace.
    # Preserve indentation because it can indicate nested lists.

    lines = []

    for line in text.split("\n"):

        line = line.rstrip(" \t")

        lines.append(line)

    text = "\n".join(lines)

    # Limit excessive blank lines.

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# 5. HELPER: CREATE BLOCK
# ============================================================

def create_block(
    text,
    block_type,
    block_number,
    line_start,
    line_end,
    document_id,
    document_name,
    source_path,
    encoding,
    heading_path,
    **extra_metadata
):

    metadata = {
        "document_id": document_id,
        "document_name": document_name,
        "source_path": source_path,
        "file_type": "txt",
        "encoding": encoding,
        "block_type": block_type,
        "block_number": block_number,
        "line_start": line_start,
        "line_end": line_end,
        "heading_path": list(heading_path),
        "section": heading_path[0]
            if len(heading_path) >= 1 else None,
        "subsection": heading_path[1]
            if len(heading_path) >= 2 else None,
        "subsubsection": heading_path[2]
            if len(heading_path) >= 3 else None
    }

    metadata.update(extra_metadata)

    return {
        "text": text,
        "metadata": metadata
    }


# ============================================================
# 6. STRUCTURE DETECTION HELPERS
# ============================================================

HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*#*\s*$"
)

LIST_PATTERN = re.compile(
    r"^(\s*)([-+*]|\d+[.)])\s+(.+)$"
)

TABLE_SEPARATOR_PATTERN = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


def get_heading(line):

    match = HEADING_PATTERN.match(line.strip())

    if not match:
        return None

    return len(match.group(1)), match.group(2).strip()


def is_list_item(line):

    return LIST_PATTERN.match(line) is not None


def is_markdown_table(lines, index):

    if index + 1 >= len(lines):
        return False

    current = lines[index].strip()
    following = lines[index + 1].strip()

    return (
        "|" in current
        and bool(TABLE_SEPARATOR_PATTERN.match(following))
    )


# ============================================================
# 7. TEXT TO STRUCTURED BLOCKS
# ============================================================

def txt_to_blocks(
    text,
    document_id,
    document_name,
    source_path,
    encoding
):

    lines = text.splitlines()

    blocks = []

    heading_stack = {}

    block_number = 0

    paragraph_buffer = []
    paragraph_start = None

    def add_block(
        content,
        block_type,
        line_start,
        line_end,
        **extra
    ):

        nonlocal block_number

        content = content.strip()

        if not content:
            return

        block_number += 1

        heading_path = [
            heading_stack[level]
            for level in sorted(heading_stack)
        ]

        blocks.append(
            create_block(
                text=content,
                block_type=block_type,
                block_number=block_number,
                line_start=line_start,
                line_end=line_end,
                document_id=document_id,
                document_name=document_name,
                source_path=source_path,
                encoding=encoding,
                heading_path=heading_path,
                **extra
            )
        )

    # --------------------------------------------------------
    # Paragraph buffer handling
    # --------------------------------------------------------

    def flush_paragraph():

        nonlocal paragraph_start

        if not paragraph_buffer:
            return

        if paragraph_start is None:
            return

        content = " ".join(
            line.strip()
            for line in paragraph_buffer
        )

        line_end = (
            paragraph_start + len(paragraph_buffer) - 1
        )

        add_block(
            content=content,
            block_type="paragraph",
            line_start=paragraph_start,
            line_end=line_end
        )

        paragraph_buffer.clear()
        paragraph_start = None

    # --------------------------------------------------------
    # Main parsing loop
    # --------------------------------------------------------

    i = 0

    while i < len(lines):

        line = lines[i]
        line_number = i + 1

        # Skip blank lines and finish current paragraph.

        if not line.strip():

            flush_paragraph()
            i += 1
            continue

        # ----------------------------------------------------
        # CODE BLOCKS
        # ----------------------------------------------------

        if line.strip().startswith("```"):

            flush_paragraph()

            opening_line = line.strip()
            language = opening_line[3:].strip() or None

            code_lines = []
            start_line = line_number

            i += 1

            while i < len(lines):

                if lines[i].strip().startswith("```"):
                    break

                code_lines.append(lines[i])
                i += 1

            if i == len(lines):
                logger.warning(
                    "Unclosed code block starting at line %s",
                    start_line
                )

            add_block(
                content="\n".join(code_lines),
                block_type="code_block",
                line_start=start_line,
                line_end=i + 1 if i < len(lines) else len(lines),
                language=language
            )

            i += 1
            continue

        # ----------------------------------------------------
        # MARKDOWN TABLES
        # ----------------------------------------------------

        if is_markdown_table(lines, i):

            flush_paragraph()

            start_line = line_number
            table_lines = []

            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i].strip())
                i += 1

            add_block(
                content="\n".join(table_lines),
                block_type="table",
                line_start=start_line,
                line_end=start_line + len(table_lines) - 1
            )

            continue

        # ----------------------------------------------------
        # MARKDOWN HEADINGS
        # ----------------------------------------------------

        heading = get_heading(line)

        if heading:

            flush_paragraph()

            level, heading_text = heading

            heading_stack = {
                key: value
                for key, value in heading_stack.items()
                if key < level
            }

            heading_stack[level] = heading_text

            add_block(
                content=heading_text,
                block_type="heading",
                line_start=line_number,
                line_end=line_number,
                heading_level=level
            )

            i += 1
            continue

        # ----------------------------------------------------
        # LIST ITEMS
        # ----------------------------------------------------

        list_match = LIST_PATTERN.match(line)

        if list_match:

            flush_paragraph()

            indentation, marker, content = list_match.groups()

            add_block(
                content=content.strip(),
                block_type="list_item",
                line_start=line_number,
                line_end=line_number,
                list_marker=marker,
                list_depth=len(indentation.expandtabs(4)),
                list_type=(
                    "ordered"
                    if marker[0].isdigit()
                    else "unordered"
                )
            )

            i += 1
            continue

        # ----------------------------------------------------
        # ORDINARY PARAGRAPHS
        # ----------------------------------------------------

        if paragraph_start is None:
            paragraph_start = line_number

        paragraph_buffer.append(line)

        i += 1

    # Flush final paragraph.

    flush_paragraph()

    return blocks


# ============================================================
# 8. BLOCK VALIDATION
# ============================================================

def validate_blocks(blocks):

    valid_blocks = []

    for block in blocks:

        text = block.get("text")
        metadata = block.get("metadata")

        if not isinstance(text, str):
            logger.warning("Invalid text type. Block skipped.")
            continue

        if not text.strip():
            logger.warning("Empty block skipped.")
            continue

        if not isinstance(metadata, dict):
            logger.warning("Invalid metadata. Block skipped.")
            continue

        required_fields = [
            "document_id",
            "document_name",
            "file_type",
            "block_type",
            "block_number"
        ]

        if not all(field in metadata for field in required_fields):
            logger.warning(
                "Required metadata missing. Block skipped."
            )
            continue

        valid_blocks.append(block)

    logger.info(
        "Validated %s blocks out of %s",
        len(valid_blocks),
        len(blocks)
    )

    return valid_blocks


# ============================================================
# 9. FINAL TXT PROCESSING FUNCTION
# ============================================================

def final_function(file_path):

    path = validate_file(file_path)

    document_id = str(uuid.uuid4())

    document_name = path.name

    source_path = str(path.resolve())

    logger.info(
        "Starting TXT processing: %s",
        document_name
    )

    # Step 1: Read file and detect encoding.

    text, encoding = txt_open(path)

    if not text.strip():
        logger.warning(
            "No content found in TXT file: %s",
            document_name
        )
        return []

    # Step 2: Clean text.

    text = clean_text(text)

    # Step 3: Convert text into structured blocks.

    blocks = txt_to_blocks(
        text=text,
        document_id=document_id,
        document_name=document_name,
        source_path=source_path,
        encoding=encoding
    )

    # Step 4: Validate blocks.

    output = validate_blocks(blocks)

    logger.info(
        "Completed TXT processing: %s, %s blocks",
        document_name,
        len(output)
    )

    return output


# ============================================================
# 10. EXECUTION
# ============================================================

if __name__ == "__main__":

    file_path = r"C:\Users\yashw\Desktop\RAG\simple.txt"

    output = final_function(file_path)

    print("Total blocks:", len(output))

    for block in output[:10]:

        print("\nTEXT:")
        print(block["text"])

        print("\nMETADATA:")
        print(block["metadata"])
