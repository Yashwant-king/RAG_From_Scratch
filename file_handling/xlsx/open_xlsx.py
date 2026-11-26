
import logging
import uuid
from pathlib import Path
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# FILE VALIDATION
# --------------------------------------------------

def validate_file(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if path.suffix.lower() != ".xlsx":
        raise ValueError("Only .xlsx files are supported.")

    if path.stat().st_size == 0:
        raise ValueError("The Excel file is empty.")

    return path


# --------------------------------------------------
# OPEN EXCEL WORKBOOK
# --------------------------------------------------

def xlsx_open(file_path, formula_mode="formula"):
    """
    formula_mode:
        formula -> preserve formulas such as =SUM(A1:A5)
        cached  -> read the last calculated values saved in Excel
    """

    if formula_mode not in ("formula", "cached"):
        raise ValueError(
            "formula_mode must be 'formula' or 'cached'."
        )

    path = validate_file(file_path)

    try:
        workbook = load_workbook(
            path,
            data_only=(formula_mode == "cached"),
            read_only=False
        )

        logger.info("Workbook opened: %s", path.name)
        return workbook

    except (InvalidFileException, BadZipFile) as exc:
        raise ValueError(
            "Invalid or corrupted XLSX workbook."
        ) from exc


# --------------------------------------------------
# VALUE NORMALIZATION
# --------------------------------------------------

def normalize_value(value):
    """
    Convert Excel cell values into readable strings.
    """

    if value is None:
        return ""

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if isinstance(value, bool):
        return str(value)

    return str(value).strip()


# --------------------------------------------------
# HEADER NORMALIZATION
# --------------------------------------------------

def normalize_headers(raw_headers):
    """
    Handle missing and duplicate column names.
    """

    headers = []
    counts = {}

    for index, value in enumerate(raw_headers, start=1):

        name = normalize_value(value)

        if not name:
            name = f"column_{index}"

        name = name.strip()
        counts[name] = counts.get(name, 0) + 1

        if counts[name] > 1:
            name = f"{name}_{counts[name]}"

        headers.append(name)

    return headers


# --------------------------------------------------
# FIND HEADER ROW
# --------------------------------------------------

def find_header_row(worksheet):
    """
    Find the first row containing at least one value.
    Returns None if the sheet is empty.
    """

    for row in worksheet.iter_rows():
        if any(cell.value is not None for cell in row):
            return row[0].row

    return None


# --------------------------------------------------
# CREATE TEXT FROM ROW
# --------------------------------------------------

def row_to_text(headers, row_values):
    """
    Convert a spreadsheet row into readable text.
    """

    lines = []

    for header, value in zip(headers, row_values):

        value = normalize_value(value)

        if value:
            lines.append(f"{header}: {value}")

    return "\n".join(lines)


# --------------------------------------------------
# CREATE METADATA
# --------------------------------------------------

def create_metadata(
    document_id,
    path,
    worksheet,
    block_number,
    row_number,
    headers,
    cells,
    header_row,
    formulas,
    hyperlinks
):

    return {
        "document_id": document_id,
        "document_name": path.name,
        "source_path": str(path),

        "file_type": "xlsx",
        "block_type": "spreadsheet_row",
        "block_number": block_number,

        "sheet_name": worksheet.title,
        "sheet_state": worksheet.sheet_state,

        "row_number": row_number,
        "header_row": header_row,

        "columns": headers,
        "cell_coordinates": [
            cell.coordinate for cell in cells
            if cell.value is not None
        ],

        "formulas": formulas,
        "hyperlinks": hyperlinks,

        "merged_ranges": [
            str(merged_range)
            for merged_range in worksheet.merged_cells.ranges
        ]
    }


# --------------------------------------------------
# CONVERT WORKSHEET INTO BLOCKS
# --------------------------------------------------

def worksheet_to_blocks(
    worksheet,
    path,
    document_id,
    block_counter,
    has_header=True,
    header_row=None,
    include_empty_rows=False
):

    blocks = []

    # Find the header row
    if has_header:

        if header_row is None:
            header_row = find_header_row(worksheet)

        if header_row is None:
            logger.warning(
                "Skipping empty worksheet: %s",
                worksheet.title
            )
            return blocks, block_counter

        raw_headers = [
            worksheet.cell(
                row=header_row,
                column=column
            ).value
            for column in range(1, worksheet.max_column + 1)
        ]

        headers = normalize_headers(raw_headers)

        data_start_row = header_row + 1

    else:

        headers = [
            f"column_{column}"
            for column in range(1, worksheet.max_column + 1)
        ]

        data_start_row = 1
        header_row = None

    # Read data rows
    for row in worksheet.iter_rows(
        min_row=data_start_row,
        max_col=max(worksheet.max_column, len(headers))
    ):

        row_number = row[0].row

        # Handle rows wider than the header
        row_values = [
            cell.value for cell in row
        ]

        if len(row_values) > len(headers):

            for column in range(
                len(headers) + 1,
                len(row_values) + 1
            ):
                headers.append(f"column_{column}")

        # Pad shorter rows
        row_values.extend(
            [""] * (len(headers) - len(row_values))
        )

        is_empty = all(
            value is None or str(value).strip() == ""
            for value in row_values
        )

        if is_empty and not include_empty_rows:
            continue

        text = row_to_text(headers, row_values)

        if not text and not include_empty_rows:
            continue

        formulas = {}
        hyperlinks = {}

        for cell in row:

            if cell.value is None:
                continue

            if (
                isinstance(cell.value, str)
                and cell.value.startswith("=")
            ):
                formulas[cell.coordinate] = cell.value

            if cell.hyperlink:
                hyperlinks[cell.coordinate] = (
                    cell.hyperlink.target
                )

        block_counter += 1

        metadata = create_metadata(
            document_id=document_id,
            path=path,
            worksheet=worksheet,
            block_number=block_counter,
            row_number=row_number,
            headers=headers.copy(),
            cells=row,
            header_row=header_row,
            formulas=formulas,
            hyperlinks=hyperlinks
        )

        blocks.append({
            "text": text,
            "metadata": metadata
        })

    return blocks, block_counter


# --------------------------------------------------
# CONVERT ENTIRE WORKBOOK INTO BLOCKS
# --------------------------------------------------

def xlsx_to_blocks(
    workbook,
    path,
    has_header=True,
    header_row=None,
    include_empty_rows=False,
    include_hidden_sheets=False
):

    all_blocks = []

    document_id = str(uuid.uuid4())
    block_counter = 0

    for worksheet in workbook.worksheets:

        if (
            worksheet.sheet_state != "visible"
            and not include_hidden_sheets
        ):
            logger.info(
                "Skipping hidden worksheet: %s",
                worksheet.title
            )
            continue

        logger.info(
            "Processing worksheet: %s",
            worksheet.title
        )

        blocks, block_counter = worksheet_to_blocks(
            worksheet=worksheet,
            path=path,
            document_id=document_id,
            block_counter=block_counter,
            has_header=has_header,
            header_row=header_row,
            include_empty_rows=include_empty_rows
        )

        all_blocks.extend(blocks)

    return all_blocks


# --------------------------------------------------
# VALIDATE OUTPUT BLOCKS
# --------------------------------------------------

def validate_blocks(blocks):

    if not isinstance(blocks, list):
        raise TypeError("Output must be a list.")

    for index, block in enumerate(blocks):

        if not isinstance(block, dict):
            raise TypeError(
                f"Block {index} must be a dictionary."
            )

        if "text" not in block:
            raise ValueError(
                f"Block {index} is missing text."
            )

        if "metadata" not in block:
            raise ValueError(
                f"Block {index} is missing metadata."
            )

        if not isinstance(block["metadata"], dict):
            raise TypeError(
                f"Block {index} metadata must be a dictionary."
            )

        if not isinstance(block["text"], str):
            raise TypeError(
                f"Block {index} text must be a string."
            )

        if not block["text"].strip():
            if block["metadata"].get("block_type") != "spreadsheet_row":
                raise ValueError(
                    f"Block {index} has empty text."
                )

        required_fields = [
            "document_id",
            "file_type",
            "block_type",
            "block_number",
            "sheet_name",
            "row_number"
        ]

        for field in required_fields:
            if field not in block["metadata"]:
                raise ValueError(
                    f"Block {index} missing metadata field: {field}"
                )

    logger.info("Validated %d blocks.", len(blocks))
    return blocks


# --------------------------------------------------
# FINAL FUNCTION
# --------------------------------------------------

def final_function(
    file_path,
    has_header=True,
    header_row=None,
    formula_mode="formula",
    include_empty_rows=False,
    include_hidden_sheets=False
):
    """
    Main function.

    Input:
        file_path

    Output:
        List of dictionaries containing text and metadata.
    """

    path = validate_file(file_path)
    workbook = None

    try:

        workbook = xlsx_open(
            path,
            formula_mode=formula_mode
        )

        blocks = xlsx_to_blocks(
            workbook=workbook,
            path=path,
            has_header=has_header,
            header_row=header_row,
            include_empty_rows=include_empty_rows,
            include_hidden_sheets=include_hidden_sheets
        )

        blocks = validate_blocks(blocks)

        logger.info(
            "Extraction completed. Total blocks: %d",
            len(blocks)
        )

        return blocks

    finally:

        if workbook is not None:
            workbook.close()


# --------------------------------------------------
# EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    file_path = r"C:\Users\yashw\Desktop\RAG\sample3.xlsx"

    output = final_function(file_path)

    for block in output:
        print("\nTEXT:")
        print(block["text"])

        print("\nMETADATA:")
        print(block["metadata"])
