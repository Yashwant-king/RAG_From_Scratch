
import mammoth
import logging
import uuid

from pathlib import Path
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString


# ============================================================
# 1. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    filename="docx_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. DOCX TO HTML EXTRACTION
# ============================================================

def docx_open(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError("Provided path is not a file.")

    if file_path.suffix.lower() != ".docx":
        raise ValueError("Expected a .docx file.")

    try:
        with file_path.open("rb") as docx_file:

            result = mammoth.convert_to_html(docx_file)

            html_content = result.value
            messages = result.messages

        return html_content, messages

    except Exception as e:
        logger.exception("DOCX extraction failed.")
        raise RuntimeError(
            f"Could not extract DOCX: {e}"
        ) from e


# ============================================================
# 3. HELPER: EXTRACT DIRECT LIST ITEM TEXT
# ============================================================

def get_list_item_text(element):

    parts = []

    for child in element.children:

        if isinstance(child, NavigableString):
            parts.append(str(child))

        elif isinstance(child, Tag):

            # Avoid including nested list content twice
            if child.name not in ["ul", "ol"]:
                parts.append(
                    child.get_text(" ", strip=True)
                )

    return " ".join(parts).strip()


# ============================================================
# 4. HELPER: EXTRACT TABLE CONTENT
# ============================================================

def extract_table(table):

    rows = []

    for row in table.find_all("tr"):

        # Ignore rows belonging to nested tables
        if row.find_parent("table") != table:
            continue

        cells = row.find_all(
            ["th", "td"],
            recursive=False
        )

        cell_values = [
            cell.get_text(" ", strip=True)
            for cell in cells
        ]

        if any(cell_values):
            rows.append(cell_values)

    if not rows:
        return ""

    # Do not assume the first row is a header.
    # Preserve every row in its original order.

    formatted_rows = []

    for index, row in enumerate(rows, start=1):

        row_text = " | ".join(
            value if value else "[Empty cell]"
            for value in row
        )

        formatted_rows.append(
            f"Row {index}: {row_text}"
        )

    return "\n".join(formatted_rows)


# ============================================================
# 5. HTML TO STRUCTURED BLOCKS
# ============================================================

def html_to_blocks(
    html_content,
    document_name,
    document_id,
    source_path
):

    soup = BeautifulSoup(
        html_content,
        "html.parser"
    )

    root = soup.body if soup.body else soup

    blocks = []

    # Store heading hierarchy
    heading_stack = {}

    block_number = 0
    table_number = 0
    image_number = 0

    elements = root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6",
         "p", "li", "table", "img"]
    )

    for element in elements:

        tag = element.name

        # ----------------------------------------------------
        # Skip paragraphs inside tables or list items.
        # Their text is handled by their parent elements.
        # ----------------------------------------------------

        if tag == "p":

            if element.find_parent("table"):
                continue

            if element.find_parent("li"):
                continue

        # ----------------------------------------------------
        # Skip nested tables because their parent table
        # will be processed separately.
        # ----------------------------------------------------

        if tag == "table":

            parent_table = element.find_parent("table")

            if parent_table is not None:
                continue

        # ----------------------------------------------------
        # HEADING EXTRACTION
        # ----------------------------------------------------

        if tag in [
            "h1", "h2", "h3",
            "h4", "h5", "h6"
        ]:

            level = int(tag[1])

            text = element.get_text(
                " ",
                strip=True
            )

            if not text:
                continue

            # Remove headings at the same or deeper level
            heading_stack = {
                key: value
                for key, value in heading_stack.items()
                if key < level
            }

            heading_stack[level] = text

            block_type = "heading"

            extra_metadata = {
                "heading_level": level
            }

        # ----------------------------------------------------
        # PARAGRAPH EXTRACTION
        # ----------------------------------------------------

        elif tag == "p":

            text = element.get_text(
                " ",
                strip=True
            )

            if not text:
                continue

            block_type = "paragraph"

            extra_metadata = {}

        # ----------------------------------------------------
        # LIST EXTRACTION
        # ----------------------------------------------------

        elif tag == "li":

            text = get_list_item_text(element)

            if not text:
                continue

            parent_list = element.find_parent(
                ["ul", "ol"]
            )

            list_type = (
                parent_list.name
                if parent_list
                else None
            )

            list_depth = len(
                element.find_parents(["ul", "ol"])
            )

            block_type = "list_item"

            extra_metadata = {
                "list_type": list_type,
                "list_depth": list_depth
            }

        # ----------------------------------------------------
        # TABLE EXTRACTION
        # ----------------------------------------------------

        elif tag == "table":

            text = extract_table(element)

            if not text:
                continue

            table_number += 1

            block_type = "table"

            extra_metadata = {
                "table_number": table_number,
                "row_count": len(
                    element.find_all("tr")
                )
            }

        # ----------------------------------------------------
        # IMAGE EXTRACTION
        # ----------------------------------------------------

        elif tag == "img":

            alt_text = element.get("alt", "").strip()

            image_number += 1

            if alt_text:
                text = f"[Image: {alt_text}]"
            else:
                text = "[Image without alternative text]"

                logger.warning(
                    f"Image {image_number} has no alt text."
                )

            block_type = "image"

            extra_metadata = {
                "image_number": image_number,
                "has_alt_text": bool(alt_text)
            }

        else:
            continue

        # ----------------------------------------------------
        # COMMON METADATA
        # ----------------------------------------------------

        block_number += 1

        heading_path = [
            heading_stack[level]
            for level in sorted(heading_stack)
        ]

        metadata = {
            "document_id": document_id,
            "document_name": document_name,
            "source_path": source_path,
            "file_type": "docx",
            "block_type": block_type,
            "block_number": block_number,

            "heading_path": heading_path,

            "section": heading_path[0]
                if len(heading_path) >= 1 else None,

            "subsection": heading_path[1]
                if len(heading_path) >= 2 else None,

            "subsubsection": heading_path[2]
                if len(heading_path) >= 3 else None
        }

        metadata.update(extra_metadata)

        blocks.append({
            "text": text,
            "metadata": metadata
        })

    return blocks


# ============================================================
# 6. BLOCK VALIDATION
# ============================================================

def validate_blocks(blocks):

    valid_blocks = []

    for block in blocks:

        text = block.get("text", "")
        metadata = block.get("metadata", {})

        if not isinstance(text, str):
            logger.warning("Invalid text type. Block skipped.")
            continue

        if not text.strip():
            logger.warning("Empty block skipped.")
            continue

        if not metadata.get("document_id"):
            logger.warning("Missing document ID.")
            continue

        if not metadata.get("block_type"):
            logger.warning("Missing block type.")
            continue

        valid_blocks.append(block)

    logger.info(
        f"Validation complete: {len(valid_blocks)} valid blocks."
    )

    return valid_blocks


# ============================================================
# 7. FINAL DOCX PROCESSING FUNCTION
# ============================================================

def final_function(file_path):

    file_path = Path(file_path)

    # Generate unique document ID
    document_id = str(uuid.uuid4())

    document_name = file_path.name

    logger.info(
        f"Starting DOCX processing: {document_name}"
    )

    # Step 1: DOCX to HTML
    html_content, messages = docx_open(file_path)

    # Step 2: Record Mammoth conversion warnings
    for message in messages:

        logger.warning(
            f"Mammoth warning: {message}"
        )

    # Step 3: HTML to structured blocks
    blocks = html_to_blocks(
        html_content=html_content,
        document_name=document_name,
        document_id=document_id,
        source_path=str(file_path.resolve())
    )

    # Step 4: Validate extracted blocks
    output = validate_blocks(blocks)

    logger.info(
        f"DOCX processing completed: "
        f"{document_name}, "
        f"{len(output)} blocks"
    )

    return output


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":

    file_path = r"C:\Users\yashw\Desktop\RAG\ClimateChangemapping15-August.docx.docx"

    output = final_function(file_path)

    print("Total extracted blocks:", len(output))

    for block in output[:5]:
        print(block)
