"""
VectorStore: a thin, validated wrapper around a persistent ChromaDB collection.

Expected input chunk shape (one dict per chunk):

    {
        "chunk_id":  "doc_1_chunk_0",          # required, unique string
        "text":      "chunk content",          # required, non-empty string
        "embedding": [0.12, -0.03, 0.88],      # required, list / tuple / numpy array
        "metadata":  {                         # required, dict
            "document_id": "doc_1",            # required (used for delete / replace)
            "source": "report.pdf",            # optional extras
            "page": 3
        }
    }
"""

import json
import logging
import math
from pathlib import Path
from typing import Literal

import chromadb

logger = logging.getLogger(__name__)


# ==================================================
# HELPERS
# ==================================================

def _default_db_path():
    """Project-relative path; falls back to the working directory
    in notebooks (Kaggle / Colab), where __file__ is undefined."""
    try:
        base = Path(__file__).resolve().parent.parent
    except NameError:
        base = Path.cwd()
    return base / "storage" / "chroma_db"


def _to_float_list(values, name):
    """Validate a vector and return it as a plain list of Python floats.
    Accepts list, tuple, or numpy arrays."""

    if hasattr(values, "tolist"):          # numpy array -> python list
        values = values.tolist()

    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty list of numbers.")

    if not all(
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
        for v in values
    ):
        raise ValueError(f"{name} contains non-numeric or non-finite values.")

    result = [float(v) for v in values]

    # A zero vector has no direction, so cosine distance is undefined.
    if not any(result):
        raise ValueError(f"{name} is a zero vector.")

    return result


# ==================================================
# VECTOR STORE
# ==================================================

class VectorStore:

    def __init__(self, collection_name="rag_documents", db_path=None):

        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.db_path))

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # The distance metric is fixed when a collection is created.
        existing_space = (self.collection.metadata or {}).get("hnsw:space")
        if existing_space != "cosine":
            logger.warning(
                "Collection '%s' uses distance metric '%s', not cosine. "
                "Similarity scores will not mean what you expect.",
                collection_name, existing_space
            )

        logger.info("Connected to ChromaDB collection: %s", collection_name)

    # --------------------------------------------------
    # Metadata cleaning
    # --------------------------------------------------

    def clean_metadata(self, metadata):
        """Chroma metadata values must be str / int / float / bool."""

        clean = {}

        for key, value in metadata.items():

            if key == "embedding" or value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"Non-finite metadata value for '{key}'.")
                clean[key] = value

            elif isinstance(value, (dict, list, tuple)):
                # JSON so it can be parsed back with json.loads
                clean[key] = json.dumps(value, ensure_ascii=False, default=str)

            else:
                clean[key] = str(value)

        return clean

    # --------------------------------------------------
    # Input validation
    # --------------------------------------------------

    def validate_chunks(self, chunks):

        if not isinstance(chunks, list):
            raise TypeError("Input must be a list of chunks.")
        if not chunks:
            raise ValueError("No chunks provided.")

        validated = []
        expected_dimension = None

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk {index} must be a dictionary.")

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            metadata = chunk.get("metadata")
            embedding = chunk.get("embedding")

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"Invalid chunk_id at index {index}.")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid text at index {index}.")

            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid metadata at index {index}.")

            # metadata["chunk_id"] is optional, but must agree if present
            if "chunk_id" in metadata and metadata["chunk_id"] != chunk_id:
                raise ValueError(f"chunk_id mismatch at index {index}.")

            # delete_document / replace_document depend on this key
            document_id = metadata.get("document_id")
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError(
                    f"metadata['document_id'] must be a non-empty string "
                    f"at index {index}."
                )

            vector = _to_float_list(embedding, f"Embedding at index {index}")

            if expected_dimension is None:
                expected_dimension = len(vector)
            elif len(vector) != expected_dimension:
                raise ValueError("All embeddings must have the same dimension.")

            declared = metadata.get("embedding_dimension")
            if declared is not None and declared != len(vector):
                raise ValueError(f"embedding_dimension mismatch at index {index}.")

            validated.append({
                "id": chunk_id,
                "text": text,
                "embedding": vector,
                "metadata": self.clean_metadata({**metadata, "chunk_id": chunk_id}),
            })

        ids = [item["id"] for item in validated]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs found in input.")

        self._check_collection_dimension(expected_dimension)

        return validated

    def _check_collection_dimension(self, dimension):
        """Compare against vectors already stored in the collection."""

        existing = self.collection.get(limit=1, include=["embeddings"])
        embeddings = existing.get("embeddings")

        if embeddings is not None and len(embeddings) > 0:
            stored_dimension = len(embeddings[0])
            if stored_dimension != dimension:
                raise ValueError(
                    f"Embedding dimension {dimension} does not match the "
                    f"collection's existing dimension {stored_dimension}. "
                    f"Use the same embedding model or a new collection."
                )

    # --------------------------------------------------
    # Storing
    # --------------------------------------------------

    def _write(self, validated, batch_size):

        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")

        # Chroma enforces a maximum batch size per call
        try:
            batch_size = min(batch_size, self.client.get_max_batch_size())
        except Exception:
            pass

        total = len(validated)

        for start in range(0, total, batch_size):
            batch = validated[start:start + batch_size]

            self.collection.upsert(
                ids=[item["id"] for item in batch],
                documents=[item["text"] for item in batch],
                embeddings=[item["embedding"] for item in batch],
                metadatas=[item["metadata"] for item in batch],
            )

            logger.info("Stored %s / %s chunks", start + len(batch), total)

        return {
            "status": "success",
            "stored_chunks": total,
            "collection_name": self.collection.name,
            "embedding_dimension": len(validated[0]["embedding"]),
            "total_collection_chunks": self.collection.count(),
        }

    def store_chunks(self, chunks, batch_size=100):
        """Insert or overwrite chunks by ID."""
        validated = self.validate_chunks(chunks)
        return self._write(validated, batch_size)

    def replace_document(self, document_id, chunks, batch_size=100):
        """Delete every old chunk of a document, then store the new ones.
        Prevents stale chunks when a re-ingested document gets shorter.
        Validation runs first, so bad input never deletes existing data."""

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("A valid document_id is required.")

        validated = self.validate_chunks(chunks)

        wrong = {
            item["metadata"]["document_id"]
            for item in validated
            if item["metadata"]["document_id"] != document_id
        }
        if wrong:
            raise ValueError(
                f"All chunks must have document_id '{document_id}', "
                f"found: {sorted(wrong)}"
            )

        self.delete_document(document_id)
        return self._write(validated, batch_size)

    # --------------------------------------------------
    # Reading
    # --------------------------------------------------

    def get_by_ids(self, ids):

        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}

        return self.collection.get(ids=ids, include=["documents", "metadatas"])

    def view_stored_data(self, limit=5, show_embeddings=False):

        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer.")

        include: list[Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]] = [
            "documents",
            "metadatas",
        ]
        if show_embeddings:
            include.append("embeddings")

        results = self.collection.get(limit=limit, include=include)

        ids = results["ids"]
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        embeddings = results.get("embeddings") if show_embeddings else None

        records = []

        for index, chunk_id in enumerate(ids):

            record = {
                "chunk_id": chunk_id,
                "text": documents[index] if documents is not None else None,
                "metadata": metadatas[index] if metadatas is not None else None,
            }

            if embeddings is not None:
                vector = embeddings[index]
                vector = list(vector)
                record["embedding_dimension"] = len(vector)
                record["embedding_preview"] = vector[:10]

            records.append(record)

        return records

    def get_collection_info(self):

        return {
            "collection_name": self.collection.name,
            "total_chunks": self.collection.count(),
            "database_path": str(self.db_path),
        }

    # --------------------------------------------------
    # Similarity search
    # --------------------------------------------------

    def similarity_search(self, query_embedding, top_k=5, where=None):

        query = _to_float_list(query_embedding, "Query embedding")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        total = self.collection.count()
        if total == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query],
            n_results=min(top_k, total),
            where=where or None,          # Chroma rejects an empty {} filter
            include=["documents", "metadatas", "distances"],
        )

        ids = results["ids"][0]
        document_rows = results.get("documents")
        documents = (
            document_rows[0]
            if document_rows and document_rows[0] is not None
            else [None] * len(ids)
        )
        metadata_rows = results.get("metadatas")
        metadatas = (
            metadata_rows[0]
            if metadata_rows and metadata_rows[0] is not None
            else [None] * len(ids)
        )
        distance_rows = results.get("distances")
        distances = (
            distance_rows[0]
            if distance_rows and distance_rows[0] is not None
            else [None] * len(ids)
        )

        return [
            {
                "chunk_id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],           # cosine distance (lower = closer)
                "similarity": 1.0 - distance if (distance := distances[i]) is not None else None,
            }
            for i in range(len(ids))
        ]

    # --------------------------------------------------
    # Deleting
    # --------------------------------------------------

    def delete_document(self, document_id):

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("A valid document_id is required.")

        self.collection.delete(where={"document_id": document_id})
        logger.info("Deleted chunks for document: %s", document_id)

    def delete_chunks(self, chunk_ids):

        if not chunk_ids:
            return

        self.collection.delete(ids=chunk_ids)
        logger.info("Deleted %s chunks", len(chunk_ids))


# ==================================================
# FINAL FUNCTION
# ==================================================

def final_function(
    embedded_chunks,
    collection_name="rag_documents",
    batch_size=100,
    db_path=None,
):

    store = VectorStore(collection_name=collection_name, db_path=db_path)
    return store.store_chunks(chunks=embedded_chunks, batch_size=batch_size)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    sample = [
        {
            "chunk_id": "doc_1_chunk_0",
            "text": "Sample text 1",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"document_id": "doc_1", "page": 1},
        },
        {
            "chunk_id": "doc_1_chunk_1",
            "text": "Sample text 2",
            "embedding": [0.4, 0.5, 0.6],
            "metadata": {"document_id": "doc_1", "page": 2},
        },
    ]

    print(final_function(sample, collection_name="test_collection", batch_size=2))