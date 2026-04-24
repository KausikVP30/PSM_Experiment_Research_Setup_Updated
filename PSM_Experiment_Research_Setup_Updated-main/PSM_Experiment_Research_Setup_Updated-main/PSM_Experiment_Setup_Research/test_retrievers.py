import numpy as np
from retrieval.hybrid_retriever import HybridRetriever

# Test Data 

DOCUMENTS = [
    "A linked list is a linear data structure where elements are stored in nodes.",
    "Each node in a linked list contains data and a pointer to the next node.",
    "Binary search trees allow fast lookup, insertion, and deletion of elements.",
    "A doubly linked list has pointers to both the next and previous nodes.",
    "Stacks follow Last In First Out (LIFO) principle for data storage.",
    "Queues follow First In First Out (FIFO) principle for data storage.",
    "Hash tables use a hash function to map keys to values for fast retrieval.",
    "Graph traversal algorithms include BFS and DFS for visiting all nodes.",
    "Dynamic programming breaks problems into overlapping subproblems.",
    "Bubble sort repeatedly swaps adjacent elements if they are in wrong order.",
]

# Helpers

def print_results(test_name, results):
    print(f"\n{'='*55}")
    print(f"  {test_name}")
    print(f"{'='*55}")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc[:80]}...")

def check_relevance(results, keywords):
    """Returns True if at least one result contains any of the keywords."""
    for doc in results:
        if any(kw.lower() in doc.lower() for kw in keywords):
            return True
    return False

# Tests

def test_index_builds_correctly(retriever):
    print("\n>>> TEST 1: Index builds correctly")
    try:
        assert retriever.index is not None,        "FAIL: HNSW index is None"
        assert retriever.bm25 is not None,         "FAIL: BM25 is None"
        assert len(retriever.documents) == len(DOCUMENTS), "FAIL: Document count mismatch"
        print("    PASS: Index built — HNSW + BM25 ready")
        print(f"    INFO: {len(retriever.documents)} documents indexed")
    except AssertionError as e:
        print(f"    {e}")


def test_returns_correct_k(retriever):
    print("\n>>> TEST 2: Returns exactly k results")
    for k in [1, 2, 4]:
        results = retriever.retrieve("linked list", k=k)
        status = "PASS" if len(results) == k else "FAIL"
        print(f"    {status}: k={k} → got {len(results)} result(s)")


def test_semantic_query(retriever):
    print("\n>>> TEST 3: Semantic query (dense retrieval working)")
    query = "How do nodes connect to each other in a list?"
    results = retriever.retrieve(query, k=3)
    passed = check_relevance(results, ["linked list", "node", "pointer"])
    print(f"    {'PASS' if passed else 'FAIL'}: Semantic query returned relevant docs")
    print_results("Semantic Query Results", results)


def test_keyword_query(retriever):
    print("\n>>> TEST 4: Exact keyword query (BM25 working)")
    query = "doubly linked list previous pointer"
    results = retriever.retrieve(query, k=3)
    passed = check_relevance(results, ["doubly", "previous"])
    print(f"    {'PASS' if passed else 'FAIL'}: Keyword query returned relevant docs")
    print_results("Keyword Query Results", results)


def test_bm25_heavy_weights(retriever):
    print("\n>>> TEST 5: BM25-heavy weights (0.8 / 0.2)")
    query = "hash function keys values"
    results = retriever.retrieve(query, k=3, bm25_weight=0.8, dense_weight=0.2)
    passed = check_relevance(results, ["hash", "keys", "values"])
    print(f"    {'PASS' if passed else 'FAIL'}: BM25-heavy retrieval returned relevant docs")
    print_results("BM25-Heavy Results", results)


def test_dense_heavy_weights(retriever):
    print("\n>>> TEST 6: Dense-heavy weights (0.2 / 0.8)")
    query = "sorting elements in the wrong position"
    results = retriever.retrieve(query, k=3, bm25_weight=0.2, dense_weight=0.8)
    passed = check_relevance(results, ["sort", "swap", "bubble"])
    print(f"    {'PASS' if passed else 'FAIL'}: Dense-heavy retrieval returned relevant docs")
    print_results("Dense-Heavy Results", results)


def test_no_duplicate_results(retriever):
    print("\n>>> TEST 7: No duplicate results in top-k")
    results = retriever.retrieve("data structure nodes", k=4)
    passed = len(results) == len(set(results))
    print(f"    {'PASS' if passed else 'FAIL'}: No duplicates in results")


def test_unrelated_query(retriever):
    print("\n>>> TEST 8: Unrelated query (graceful handling)")
    query = "the french revolution history of napoleon"
    results = retriever.retrieve(query, k=2)
    passed = len(results) == 2  # should still return k results, just less relevant
    print(f"    {'PASS' if passed else 'FAIL'}: Returned {len(results)} result(s) for unrelated query")
    print_results("Unrelated Query Results", results)


def test_scores_are_normalized(retriever):
    print("\n>>> TEST 9: BM25 scores normalize without error (all-zero edge case)")
    # Inject a query with no matching tokens
    query = "xyzzy foobar qqqqqq"
    try:
        results = retriever.retrieve(query, k=2)
        print(f"    PASS: No crash on zero BM25 scores, returned {len(results)} result(s)")
    except Exception as e:
        print(f"    FAIL: Crashed with → {e}")


def test_embedding_shape(retriever):
    print("\n>>> TEST 10: Embedding shapes are correct for FAISS")
    query_embedding = retriever.embedding_model.encode_query("test query")
    doc_embeddings  = retriever.embedding_model.encode_documents(["test doc"])
    q_ok = query_embedding.ndim == 2 and query_embedding.shape[0] == 1
    d_ok = doc_embeddings.ndim  == 2 and doc_embeddings.shape[0]  == 1
    print(f"    {'PASS' if q_ok else 'FAIL'}: Query embedding shape  → {query_embedding.shape} (expected (1, dims))")
    print(f"    {'PASS' if d_ok else 'FAIL'}: Doc embedding shape    → {doc_embeddings.shape}  (expected (1, dims))")


# Main

if __name__ == "__main__":
    print("\nInitializing HybridRetriever and building index...")
    retriever = HybridRetriever(ef_construction=200, M=32, ef_search=50)
    retriever.build_index(DOCUMENTS)
    print("Index ready.\n")

    test_index_builds_correctly(retriever)
    test_returns_correct_k(retriever)
    test_semantic_query(retriever)
    test_keyword_query(retriever)
    test_bm25_heavy_weights(retriever)
    test_dense_heavy_weights(retriever)
    test_no_duplicate_results(retriever)
    test_unrelated_query(retriever)
    test_scores_are_normalized(retriever)
    test_embedding_shape(retriever)

    print(f"\n{'='*55}")
    print("  All tests completed!")
    print(f"{'='*55}\n")