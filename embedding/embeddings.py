import json



import logging
from copy import deepcopy

import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# EMBEDDING GENERATOR
# --------------------------------------------------

class EmbeddingGenerator:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
        normalize_embeddings=True
    ):

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        logger.info("Loading embedding model: %s", model_name)

        self.model = SentenceTransformer(
            model_name,
            device=device
        )

        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        self.max_sequence_length = self.model.max_seq_length

        logger.info("Model loaded successfully.")
        logger.info(
            "Embedding dimension: %s",
            self.embedding_dimension
        )
        logger.info(
            "Maximum sequence length: %s",
            self.max_sequence_length
        )


    # --------------------------------------------------
    # GENERATE EMBEDDINGS
    # --------------------------------------------------

    def generate_embeddings(
        self,
        texts,
        batch_size=32,
        show_progress_bar=True
    ):

        if not texts:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32
            )

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        return embeddings


    # --------------------------------------------------
    # VALIDATE EMBEDDINGS
    # --------------------------------------------------

    def validate_embeddings(self, embeddings):

        if not isinstance(embeddings, np.ndarray):
            raise TypeError(
                "Embeddings must be a NumPy array."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                "Embedding dimension mismatch."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain NaN or infinite values."
            )

        if self.normalize_embeddings and len(embeddings):

            norms = np.linalg.norm(embeddings, axis=1)

            if not np.allclose(
                norms,
                1.0,
                atol=1e-3
            ):
                raise ValueError(
                    "Expected normalized embeddings."
                )

        return True


    # --------------------------------------------------
    # PROCESS CHUNKS
    # --------------------------------------------------

    def embed_chunks(
        self,
        chunks,
        batch_size=32,
        show_progress_bar=True
    ):

        if not isinstance(chunks, list):
            raise TypeError(
                "Chunks must be provided as a list."
            )

        if not chunks:
            logger.warning("No chunks received.")
            return []

        # Validate input and prepare texts.
        texts = []

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, dict):
                raise TypeError(
                    f"Chunk {index} must be a dictionary."
                )

            text = chunk.get("text")
            metadata = chunk.get("metadata")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Chunk {index} contains invalid text."
                )

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Chunk {index} has invalid metadata."
                )

            texts.append(text)

        # Check model sequence length.
        for index, text in enumerate(texts):

            token_count = len(
                self.model.tokenizer.encode(
                    text,
                    add_special_tokens=True
                )
            )

            if token_count > self.max_sequence_length:

                logger.warning(
                    "Chunk %d has %d tokens, exceeding "
                    "model sequence limit %d. The model "
                    "may truncate its input.",
                    index,
                    token_count,
                    self.max_sequence_length
                )

        # Generate vectors in batches.
        embeddings = self.generate_embeddings(
            texts=texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )

        self.validate_embeddings(embeddings)

        # Combine embeddings with original data.
        output = []

        for index, chunk in enumerate(chunks):

            metadata = deepcopy(chunk["metadata"])

            metadata.update({

                "embedding_model": self.model_name,

                "embedding_dimension": self.embedding_dimension,

                "embedding_normalized": self.normalize_embeddings

            })

            output.append({

                "chunk_id": metadata.get("chunk_id"),

                "text": chunk["text"],

                "metadata": metadata,

                "embedding": embeddings[index].tolist()

            })

        logger.info(
            "Generated %d embeddings successfully.",
            len(output)
        )

        return output


# --------------------------------------------------
# FINAL FUNCTION
# --------------------------------------------------

def final_function(
    chunks,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=32,
    device=None,
    normalize_embeddings=True,
    show_progress_bar=True
):
    """
    Generate embeddings for validated RAG chunks.

    Input:
        List of dictionaries containing text and metadata.

    Output:
        List of dictionaries containing:
            chunk_id
            text
            metadata
            embedding
    """

    generator = EmbeddingGenerator(
        model_name=model_name,
        device=device,
        normalize_embeddings=normalize_embeddings
    )

    output = generator.embed_chunks(
        chunks=chunks,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar
    )

    return output


# --------------------------------------------------
# EXECUTION EXAMPLE
# --------------------------------------------------

if __name__ == "__main__":

    sample_chunks = [

        {
            "text": "Machine learning is a subset of artificial intelligence.",

            "metadata": {
                "chunk_id": "chunk_001",
                "chunk_number": 1,
                "document_id": "doc_001",
                "document_name": "AI.txt",
                "file_type": "txt",
                "chunking_method": "recursive_character"
            }
        },

        {
            "text": "Deep learning uses neural networks to learn patterns.",

            "metadata": {
                "chunk_id": "chunk_002",
                "chunk_number": 2,
                "document_id": "doc_001",
                "document_name": "AI.txt",
                "file_type": "txt",
                "chunking_method": "recursive_character"
            }
        }

    ]
    # with open("run_files/run_files/results.json", "r") as f:
        # json.dump(sample_chunks, f)

    output = final_function(
        chunks=sample_chunks,
        batch_size=16
    )
    with open("embedding/embedding_output.json", "w") as f:
        json.dump(output, f, indent=4)
    for item in output:

        print("\nCHUNK ID:")
        print(item["chunk_id"])

        print("\nTEXT:")
        print(item["text"])

        print("\nEMBEDDING:")
        print(item["embedding"][:10])

        print("\nEMBEDDING DIMENSION:")
        print(len(item["embedding"]))

        print("\nMETADATA:")
        print(item["metadata"])
