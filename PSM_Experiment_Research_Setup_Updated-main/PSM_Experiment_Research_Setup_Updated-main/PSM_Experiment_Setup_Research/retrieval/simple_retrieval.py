from retrieval.hybrid_retriever import HybridRetriever

DOCUMENTS = [
    "A linked list is a linear data structure where elements are stored in nodes.",
    "Graph traversal algorithms include BFS and DFS for visiting all nodes.",
    "Each node in a linked list contains data and a pointer to the next node.",
    "A doubly linked list has pointers to both the next and previous nodes.",
    "Stacks follow Last In First Out (LIFO) principle for data storage.",
    "Queues follow First In First Out (FIFO) principle for data storage.",
    "Binary search trees allow fast lookup, insertion, and deletion of elements.",
    "Hash tables use a hash function to map keys to values for fast retrieval.",
    "Dynamic programming breaks problems into overlapping subproblems.",
    "Bubble sort repeatedly swaps adjacent elements if they are in wrong order.",
]

# ── Build index ───────────────────────────────────────────────
retriever = HybridRetriever()
retriever.build_index(DOCUMENTS)
print("Index ready.\n")

# ── Your query ────────────────────────────────────────────────
query = input("Enter your query: ")
results = retriever.retrieve(query, k=3)

# ── Results ───────────────────────────────────────────────────
print(f"\nTop {len(results)} chunks for: '{query}'\n")
for i, chunk in enumerate(results, 1):
    print(f"[{i}] {chunk}")