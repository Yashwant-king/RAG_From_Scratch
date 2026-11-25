
import re
import uuid
import logging

from pathlib import Path
from charset_normalizer import from_bytes
from markdown_it import MarkdownIt


# ============================================================
# 1. LOGGING CONFIGURATION
# ============================================================

logger = logging.getLogger("md_processor")

if not logger.handlers:
    handler = logging.FileHandler("md_processing.log")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ============================================================
# 2. MARKDOWN PARSER CONFIGURATION
# ============================================================

md_parser = (
    MarkdownIt("commonmark")
    .enable(["table", "strikethrough"])
)


# ============================================================
# 3. FILE VALIDATION
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

    if path.suffix.lower() not in [".md", ".markdown"]:
        raise ValueError(
            "Expected a .md or .markdown file."
        )

    return path


# ============================================================
# 4. ENCODING DETECTION
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
        "Unable to determine a usable encoding."
    )


# ============================================================
# 5. READ MARKDOWN FILE
# ============================================================

def md_open(file_path):

    path = validate_file(file_path)

    try:
        raw_bytes = path.read_bytes()

        if not raw_bytes:
            logger.warning("Empty Markdown file: %s", path.name)
            return "", "unknown"

        encoding = detect_encoding(raw_bytes)

        text = raw_bytes.decode(encoding)

        if "\x00" in text:
            raise ValueError(
                "File contains null characters and may be binary."
            )

        logger.info(
            "Read %s using encoding %s",
            path.name,
            encoding
        )

        return text, encoding

    except Exception:
        logger.exception(
            "Markdown reading failed: %s",
            path
        )
        raise


# ============================================================
# 6. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")

    # Remove excessive blank lines outside code blocks
    # by leaving structural whitespace to the Markdown parser.

    return text.strip()


# ============================================================
# 7. FRONTMATTER EXTRACTION
# ============================================================

def extract_frontmatter(text):

    lines = text.splitlines()

    if not lines or lines[0].strip() != "---":
        return None, text, 0

    for index in range(1, len(lines)):

        if lines[index].strip() in ["---", "..."]:

            frontmatter = "\n".join(lines[1:index])

            remaining_text = "\n".join(
                lines[index + 1:]
            )

            return frontmatter, remaining_text, index + 1

    # If no closing delimiter exists, preserve it as Markdown.
    return None, text, 0


# ============================================================
# 8. HELPER: LINE NUMBER EXTRACTION
# ============================================================

def get_line_range(token, line_offset=0):

    if token.map is None:
        return None, None

    start = token.map[0] + 1 + line_offset
    end = token.map[1] + line_offset

    return start, end


# ============================================================
# 9. HELPER: EXTRACT INLINE LINKS AND IMAGES
# ============================================================

def extract_inline_metadata(token):

    links = []
    images = []

    if not token.children:
        return links, images

    for child in token.children:

        if child.type == "link_open":

            links.append({
                "url": child.attrGet("href"),
                "title": child.attrGet("title")
            })

        elif child.type == "image":

            images.append({
                "alt": child.content,
                "url": child.attrGet("src"),
                "title": child.attrGet("title")
            })

    return links, images


# ============================================================
# 10. HELPER: CREATE OUTPUT BLOCK
# ============================================================

def create_block(
    text,
    block_type,
    block_number,
    document_id,
    document_name,
    source_path,
    encoding,
    line_start,
    line_end,
    heading_stack,
    **extra_metadata
):

    heading_path = [
        heading_stack[level]
        for level in sorted(heading_stack)
    ]

    metadata = {
        "document_id": document_id,
        "document_name": document_name,
        "source_path": source_path,
        "file_type": "md",
        "encoding": encoding,
        "block_type": block_type,
        "block_number": block_number,
        "line_start": line_start,
        "line_end": line_end,
        "heading_path": heading_path,
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
# 11. TABLE EXTRACTION
# ============================================================

def extract_table_tokens(tokens):

    rows = []
    current_row = []

    for token in tokens:

        if token.type == "tr_open":
            current_row = []

        elif token.type in ["th_open", "td_open"]:
            pass

        elif token.type == "inline":
            current_row.append(token.content.strip())

        elif token.type == "tr_close":

            if current_row:
                rows.append(current_row)

    if not rows:
        return ""

    formatted_rows = []

    for row_index, row in enumerate(rows, start=1):

        formatted_rows.append(
            f"Row {row_index}: " +
            " | ".join(
                cell if cell else "[Empty cell]"
                for cell in row
            )
        )

    return "\n".join(formatted_rows)


# ============================================================
# 12. MARKDOWN TO STRUCTURED BLOCKS
# ============================================================

def md_to_blocks(
    text,
    document_id,
    document_name,
    source_path,
    encoding
):

    blocks = []

    heading_stack = {}

    block_number = 0

    # Extract optional YAML-style frontmatter.
    frontmatter, text, line_offset = extract_frontmatter(text)

    def add_block(
        content,
        block_type,
        token=None,
        **extra
    ):

        nonlocal block_number

        content = content.strip()

        if not content and block_type != "horizontal_rule":
            return

        line_start, line_end = (
            get_line_range(token, line_offset)
            if token is not None
            else (None, None)
        )

        block_number += 1

        blocks.append(
            create_block(
                text=content,
                block_type=block_type,
                block_number=block_number,
                document_id=document_id,
                document_name=document_name,
                source_path=source_path,
                encoding=encoding,
                line_start=line_start,
                line_end=line_end,
                heading_stack=heading_stack,
                **extra
            )
        )

    # Preserve frontmatter as a separate block.

    if frontmatter is not None:

        add_block(
            content=frontmatter,
            block_type="frontmatter",
            frontmatter_format="yaml"
        )

    tokens = md_parser.parse(text)

    list_stack = []
    blockquote_depth = 0

    i = 0

    while i < len(tokens):

        token = tokens[i]
        token_type = token.type

        # ----------------------------------------------------
        # HEADINGS
        # ----------------------------------------------------

        if token_type == "heading_open":

            inline_token = tokens[i + 1]

            level = int(token.tag[1])
            heading_text = inline_token.content.strip()

            heading_stack = {
                key: value
                for key, value in heading_stack.items()
                if key < level
            }

            heading_stack[level] = heading_text

            add_block(
                content=heading_text,
                block_type="heading",
                token=token,
                heading_level=level
            )

            links, images = extract_inline_metadata(
                inline_token
            )

            if links or images:
                blocks[-1]["metadata"]["links"] = links
                blocks[-1]["metadata"]["images"] = images

            i += 3
            continue

        # ----------------------------------------------------
        # PARAGRAPHS
        # ----------------------------------------------------

        if token_type == "paragraph_open":

            inline_token = tokens[i + 1]

            paragraph_text = inline_token.content.strip()

            if list_stack:
                block_type = "list_item"

            elif blockquote_depth:
                block_type = "blockquote"

            else:
                block_type = "paragraph"

            extra = {}

            if list_stack:
                extra.update({
                    "list_depth": len(list_stack),
                    "list_type": list_stack[-1]
                })

            links, images = extract_inline_metadata(
                inline_token
            )

            if links:
                extra["links"] = links

            if images:
                extra["images"] = images

            # An image-only paragraph gets its own image type.
            children = inline_token.children or []

            nonempty_children = [
                child for child in children
                if child.type != "text"
                or child.content.strip()
            ]

            if (
                images
                and len(nonempty_children) == len(images)
                and all(
                    child.type == "image"
                    for child in nonempty_children
                )
            ):
                block_type = "image"

            add_block(
                content=paragraph_text,
                block_type=block_type,
                token=token,
                **extra
            )

            i += 3
            continue

        # ----------------------------------------------------
        # LIST TRACKING
        # ----------------------------------------------------

        if token_type in [
            "bullet_list_open",
            "ordered_list_open"
        ]:

            list_type = (
                "ordered"
                if token_type == "ordered_list_open"
                else "unordered"
            )

            list_stack.append(list_type)

            i += 1
            continue

        if token_type in [
            "bullet_list_close",
            "ordered_list_close"
        ]:

            if list_stack:
                list_stack.pop()

            i += 1
            continue

        # ----------------------------------------------------
        # BLOCKQUOTE TRACKING
        # ----------------------------------------------------

        if token_type == "blockquote_open":

            blockquote_depth += 1

            i += 1
            continue

        if token_type == "blockquote_close":

            blockquote_depth = max(
                0,
                blockquote_depth - 1
            )

            i += 1
            continue

        # ----------------------------------------------------
        # TABLES
        # ----------------------------------------------------

        if token_type == "table_open":

            start_token = token
            table_tokens = []

            j = i + 1

            while (
                j < len(tokens)
                and tokens[j].type != "table_close"
            ):

                table_tokens.append(tokens[j])
                j += 1

            table_text = extract_table_tokens(
                table_tokens
            )

            add_block(
                content=table_text,
                block_type="table",
                token=start_token
            )

            i = j + 1
            continue

        # ----------------------------------------------------
        # FENCED CODE AND INDENTED CODE
        # ----------------------------------------------------

        if token_type in ["fence", "code_block"]:

            add_block(
                content=token.content,
                block_type="code_block",
                token=token,
                language=(
                    token.info.strip().split()[0]
                    if token.info.strip()
                    else None
                )
            )

            i += 1
            continue

        # ----------------------------------------------------
        # RAW HTML
        # ----------------------------------------------------

        if token_type == "html_block":

            add_block(
                content=token.content,
                block_type="html",
                token=token
            )

            i += 1
            continue

        # ----------------------------------------------------
        # HORIZONTAL RULE
        # ----------------------------------------------------

        if token_type == "hr":

            add_block(
                content="---",
                block_type="horizontal_rule",
                token=token
            )

            i += 1
            continue

        i += 1

    return blocks


# ============================================================
# 13. BLOCK VALIDATION
# ============================================================

def validate_blocks(blocks):

    valid_blocks = []

    required_fields = [
        "document_id",
        "document_name",
        "file_type",
        "block_type",
        "block_number"
    ]

    for block in blocks:

        text = block.get("text")
        metadata = block.get("metadata")

        if not isinstance(text, str):
            logger.warning("Invalid text type. Block skipped.")
            continue

        if not text.strip() and metadata.get("block_type") != "horizontal_rule":
            logger.warning("Empty block skipped.")
            continue

        if not isinstance(metadata, dict):
            logger.warning("Invalid metadata. Block skipped.")
            continue

        if not all(
            field in metadata
            for field in required_fields
        ):
            logger.warning("Required metadata missing.")
            continue

        valid_blocks.append(block)

    logger.info(
        "Validated %s out of %s blocks",
        len(valid_blocks),
        len(blocks)
    )

    return valid_blocks


# ============================================================
# 14. FINAL MARKDOWN PROCESSING FUNCTION
# ============================================================

def final_function(file_path):

    path = validate_file(file_path)

    document_id = str(uuid.uuid4())

    document_name = path.name

    source_path = str(path.resolve())

    logger.info(
        "Starting Markdown processing: %s",
        document_name
    )

    # Step 1: Read Markdown file.

    text, encoding = md_open(path)

    if not text.strip():
        logger.warning(
            "Empty Markdown file: %s",
            document_name
        )
        return []

    # Step 2: Clean text.

    text = clean_text(text)

    # Step 3: Parse Markdown into structured blocks.

    blocks = md_to_blocks(
        text=text,
        document_id=document_id,
        document_name=document_name,
        source_path=source_path,
        encoding=encoding
    )

    # Step 4: Validate extracted blocks.

    output = validate_blocks(blocks)

    logger.info(
        "Markdown processing completed: %s, %s blocks",
        document_name,
        len(output)
    )

    return output


# ============================================================
# 15. EXECUTION
# ============================================================

if __name__ == "__main__":

    file_path = r"C:\Users\yashw\Desktop\RAG\sample-1.md"

    output = final_function(file_path)

    print("Total blocks:", len(output))

    for block in output[:10]:

        print("\nTEXT:")
        print(block["text"])

        print("\nMETADATA:")
        print(block["metadata"])
