vscode_lsp_terminal_prompt_tracker = {}
from copy import deepcopy
from uuid import uuid4
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

def final_function(
    blocks,
    chunk_size=500,
    chunk_overlap=75
):
    """
    Input:
        List of extracted blocks:
        [{"text": "...", "metadata": {...}}]

    Output:
        List of chunks with preserved metadata.
    """

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
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
        keep_separator="start",
        add_start_index=True
    )

    output = []

    for block_index, block in enumerate(blocks):

        text = block.get("text", "")
        metadata = block.get("metadata", {})

        if not isinstance(text, str) or not text.strip():
            continue

        if not isinstance(metadata, dict):
            continue

        documents = splitter.create_documents(
            texts=[text],
            metadatas=[metadata]
        )
        print(type(documents))

        for chunk_index, document in enumerate(documents):

            chunk_metadata = deepcopy(document.metadata)

            chunk_metadata.update({
                "chunk_id": str(uuid4()),
                "chunk_number": len(output) + 1,
                "source_block_index": block_index,
                "source_block_chunk_index": chunk_index,
                "chunking_method": "recursive_character"
            })

            output.append({
                "text": document.page_content,
                "metadata": chunk_metadata
            })

   

    return output   # list of chunks with preserved metadata




if __name__ == "__main__":
    blocks = [
        {
            "text": "This is a sample text block. It contains multiple sentences. The purpose of this block is to demonstrate the chunking process.",
            "metadata": {"source": "sample_source"}
        },
        {
            "text": "Another text block is here. It also has multiple sentences. This block will be chunked as well.",
            "metadata": {"source": "another_source"}
        }
    ]

    chunks = final_function(blocks, chunk_size=4, chunk_overlap=2)

    with open("chunking/rcs_output.json", "w") as f:
          chunks= [{"input": blocks, "output": chunks}]
          json.dump(chunks, f, indent=4)
       
    
    # for  i, chunk in enumerate(chunks):
    #     print(f"Chunk {i + 1}:")
    #     print(f"Text: {chunk['text']}")
    #     print(f"Metadata: {chunk['metadata']}")
    #     print()