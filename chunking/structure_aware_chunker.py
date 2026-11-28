
import json
import logging
import uuid
from copy import deepcopy

import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

class TokenCounter:

    def __init__(self, encoding_name="cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text):
        return len(self.encoding.encode(text))


# --------------------------------------------------
# STRUCTURAL METADATA
# --------------------------------------------------

STRUCTURE_FIELDS = [
    "heading_path",
    "section",
    "subsection",
    "subsubsection",
    "sheet_name",
    "table_number",
    "table_id",
    "parent_path"
]


# --------------------------------------------------
# NORMALIZE STRUCTURAL VALUES
# --------------------------------------------------

def freeze_value(value):
    """
    Convert lists and dictionaries into comparable values.
    """

    if isinstance(value, dict):
        return tuple(
            sorted(
                (key, freeze_value(val))
                for key, val in value.items()
            )
        )

    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)

    return value


# --------------------------------------------------
# GET CONTENT FAMILY
# --------------------------------------------------

def get_content_family(metadata):

    block_type = str(
        metadata.get("block_type", "paragraph")
    ).lower()

    if block_type in {
        "json_record",
        "json_object",
        "json_array_item"
    }:
        return "atomic_json"

    if block_type in {
        "spreadsheet_row",
        "csv_row",
        "table_row",
        "table"
    }:
        return "tabular"

    if block_type in {
        "code",
        "code_block",
        "fenced_code"
    }:
        return "code"

    return "text"


# --------------------------------------------------
# BUILD GROUPING KEY
# --------------------------------------------------

def get_group_key(block):

    metadata = block["metadata"]

    family = get_content_family(metadata)

    structure = tuple(
        (
            field,
            freeze_value(metadata.get(field))
        )
        for field in STRUCTURE_FIELDS
        if metadata.get(field) is not None
    )

    # JSON records should not be merged with other records.
    record_identity = None

    if family == "atomic_json":
        record_identity = (
            metadata.get("json_path"),
            metadata.get("record_index"),
            metadata.get("block_number")
        )

    return (
        metadata.get("document_id"),
        metadata.get("file_type"),
        family,
        structure,
        record_identity
    )


# --------------------------------------------------
# EXTRACT HEADING CONTEXT
# --------------------------------------------------

def get_heading_context(metadata):

    heading_path = metadata.get("heading_path")

    if isinstance(heading_path, (list, tuple)):
        return " > ".join(str(item) for item in heading_path)

    if isinstance(heading_path, str) and heading_path.strip():
        return heading_path

    sections = [
        metadata.get("section"),
        metadata.get("subsection"),
        metadata.get("subsubsection")
    ]

    sections = [
        str(item) for item in sections
        if item is not None and str(item).strip()
    ]

    return " > ".join(sections)


# --------------------------------------------------
# BUILD SOURCE REFERENCES
# --------------------------------------------------

def get_source_reference(block, block_index):

    metadata = block["metadata"]

    location_fields = [
        "block_number",
        "page_number",
        "row_number",
        "line_start",
        "line_end",
        "sheet_name",
        "json_path",
        "record_index",
        "cell_coordinates"
    ]

    reference = {
        "source_block_index": block_index
    }

    for field in location_fields:

        if field in metadata:
            reference[field] = deepcopy(metadata[field])

    return reference


# --------------------------------------------------
# CREATE RECURSIVE FALLBACK SPLITTER
# --------------------------------------------------

def create_recursive_splitter(
    chunk_size,
    chunk_overlap,
    encoding_name
):

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            ""
        ],
        keep_separator="start"
    )


# --------------------------------------------------
# BUILD TEXT FROM BLOCKS
# --------------------------------------------------

def combine_blocks(blocks, include_heading=True):

    texts = []

    for block in blocks:

        text = block["text"].strip()

        if text:
            texts.append(text)

    combined_text = "\n\n".join(texts)

    if not include_heading or not blocks:
        return combined_text

    heading = get_heading_context(blocks[0]["metadata"])

    if heading and not combined_text.startswith(heading):
        return f"{heading}\n\n{combined_text}"

    return combined_text


# --------------------------------------------------
# CREATE OUTPUT CHUNK
# --------------------------------------------------


def create_chunk(
    text,
    source_blocks,
    chunk_number,
    chunk_size,
    token_counter,
    include_heading=True
):

    # source_blocks contains (block, block_index) tuples
    first_block = source_blocks[0][0]

    first_metadata = first_block["metadata"]

    chunk_metadata = {}

    # Preserve document-level metadata
    document_fields = [
        "document_id",
        "document_name",
        "source_path",
        "file_type"
    ]

    for field in document_fields:
        if field in first_metadata:
            chunk_metadata[field] = deepcopy(
                first_metadata[field]
            )

    # Preserve structural context
    for field in STRUCTURE_FIELDS:
        if field in first_metadata:
            chunk_metadata[field] = deepcopy(
                first_metadata[field]
            )

    chunk_metadata.update({

        "chunk_id": str(uuid.uuid4()),

        "chunk_number": chunk_number,

        "chunking_method": "structure_aware",

        "chunk_token_count": token_counter.count(text),

        "max_chunk_size": chunk_size,

        "source_block_count": len(source_blocks),

        "source_blocks": [
            get_source_reference(block, index)
            for block, index in source_blocks
        ],

        "include_heading_context": include_heading

    })

    return {
        "text": text,
        "metadata": chunk_metadata
    }



# --------------------------------------------------
# PROCESS ONE STRUCTURAL GROUP
# --------------------------------------------------

def process_group(
    group,
    output,
    chunk_size,
    chunk_overlap,
    token_counter,
    recursive_splitter,
    include_heading=True
):

    if not group:
        return

    current_blocks = []

    def flush_current():

        if not current_blocks:
            return

        text = combine_blocks(
            [item[0] for item in current_blocks],
            include_heading=include_heading
        )

        output.append(
            create_chunk(
                text=text,
                source_blocks=current_blocks,
                chunk_number=len(output) + 1,
                chunk_size=chunk_size,
                token_counter=token_counter,
                include_heading=include_heading
            )
        )

        current_blocks.clear()

    for block, block_index in group:

        candidate_blocks = current_blocks + [
            (block, block_index)
        ]

        candidate_text = combine_blocks(
            [item[0] for item in candidate_blocks],
            include_heading=include_heading
        )

        if token_counter.count(candidate_text) <= chunk_size:

            current_blocks.append((block, block_index))
            continue

        # Save the current chunk before processing
        # the next block.
        flush_current()

        single_text = combine_blocks(
            [block],
            include_heading=include_heading
        )

        # Oversized block: use recursive fallback.
        if token_counter.count(single_text) > chunk_size:

            pieces = recursive_splitter.split_text(single_text)

            for piece in pieces:

                output.append(
                    create_chunk(
                        text=piece,
                        source_blocks=[(block, block_index)],
                        chunk_number=len(output) + 1,
                        chunk_size=chunk_size,
                        token_counter=token_counter,
                        include_heading=include_heading
                    )
                )

        else:
            current_blocks.append((block, block_index))

    flush_current()


# --------------------------------------------------
# GROUP CONSECUTIVE COMPATIBLE BLOCKS
# --------------------------------------------------

def group_blocks(blocks):

    groups = []
    current_group = []
    previous_key = None

    for index, block in enumerate(blocks):

        if not isinstance(block, dict):
            logger.warning("Skipping invalid block: %s", index)
            continue

        if not isinstance(block.get("text"), str):
            logger.warning("Skipping invalid text: %s", index)
            continue

        metadata = block.get("metadata", {})

        if not isinstance(metadata, dict):
            logger.warning("Skipping invalid metadata: %s", index)
            continue

        if not block["text"].strip():
            continue

        key = get_group_key(block)

        if current_group and key != previous_key:
            groups.append(current_group)
            current_group = []

        current_group.append((block, index))
        previous_key = key

        # JSON records must remain independent.
        if get_content_family(metadata) == "atomic_json":
            groups.append(current_group)
            current_group = []
            previous_key = None

    if current_group:
        groups.append(current_group)

    return groups


# --------------------------------------------------
# VALIDATE OUTPUT
# --------------------------------------------------

def validate_chunks(chunks, chunk_size, token_counter):

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk.get("text"), str):
            raise TypeError(f"Invalid text in chunk {index}")

        if not isinstance(chunk.get("metadata"), dict):
            raise TypeError(f"Invalid metadata in chunk {index}")

        if not chunk["text"].strip():
            raise ValueError(f"Empty chunk at index {index}")

        actual_tokens = token_counter.count(chunk["text"])

        if actual_tokens > chunk_size:
            raise ValueError(
                f"Chunk {index} exceeds maximum size: "
                f"{actual_tokens} > {chunk_size}"
            )

    logger.info("Validated %d chunks.", len(chunks))

    return chunks


# --------------------------------------------------
# FINAL FUNCTION
# --------------------------------------------------

def final_function(
    blocks,
    chunk_size=500,
    chunk_overlap=75,
    encoding_name="cl100k_base",
    include_heading=True
):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and < chunk_size."
        )

    token_counter = TokenCounter(encoding_name)

    recursive_splitter = create_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name
    )

    groups = group_blocks(blocks)

    output = []

    for group in groups:

        process_group(
            group=group,
            output=output,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            token_counter=token_counter,
            recursive_splitter=recursive_splitter,
            include_heading=include_heading
        )

    output = validate_chunks(
        output,
        chunk_size,
        token_counter
    )

    logger.info(
        "Structure-aware chunking completed: %d chunks",
        len(output)
    )

    return output


# --------------------------------------------------
# EXECUTION EXAMPLE
# --------------------------------------------------

if __name__ == "__main__":

    with open("run_files/run_files/result.json", "r") as f:
        sample_blocks = json.load(f)
        sample_blocks= sample_blocks[0]

    
         
# sample_blocks = [{"text": "This is a sample block.", "metadata": {"source": "example"}}]

    result = final_function(            # give list take list
        blocks=sample_blocks,
        chunk_size=100,
        chunk_overlap=15
    )

    with open("chunking/structure_aware_output.json", "w") as f:
        json.dump(result, f, indent=4)
    
