from router.router import Router

DOCUMENTS = [\
    "A linked list is a linear data structure where elements are stored in nodes.",
    "Each node in a linked list contains data and a pointer to the next node.",
    "A doubly linked list has pointers to both the next and previous nodes.",
    "Binary search trees allow fast lookup, insertion, and deletion of elements.",
    "Stacks follow Last In First Out (LIFO) principle for data storage.",
    "Queues follow First In First Out (FIFO) principle for data storage.",
    "Hash tables use a hash function to map keys to values for fast retrieval.",
    "Graph traversal algorithms include BFS and DFS for visiting all nodes.",
    "Dynamic programming breaks problems into overlapping subproblems.",
    "Bubble sort repeatedly swaps adjacent elements if they are in wrong order.",
]


def is_valid_query(text):
    cleaned = (text or "").strip()
    if not cleaned:
        return False

    alnum_count = sum(1 for ch in cleaned if ch.isalnum())
    if alnum_count < 3:
        return False

    return any(ch.isalpha() for ch in cleaned)

if __name__ == "__main__":
    router = Router(documents=DOCUMENTS)

    while True:
        query = input("\nEnter your query, 'metrics', or 'exit': ").strip()
        if query.lower() == "exit":
            break

        if query.lower() == "metrics":
            metric_query = input("Enter query text to search metrics: ").strip()
            metrics = router.logger.get_metrics_for_query(metric_query)

            if not metrics:
                print("\nNo matching metric entries found for this query.")
                continue

            latest = metrics["latest"]

            def _fmt(value):
                return "NA" if value is None else f"{float(value):.3f}"

            print("\n--- query metrics ---")
            print(f"query             : {metrics['query']}")
            print(f"matched_runs      : {metrics['matches']}")
            print(f"avg_confidence    : {_fmt(metrics['avg_confidence'])}")
            print(f"max_confidence    : {_fmt(metrics['max_confidence'])}")
            print(f"avg_latency_s     : {_fmt(metrics['avg_latency'])}")
            print(f"min_latency_s     : {_fmt(metrics['min_latency'])}")
            print(
                "source_split      : "
                f"memory={metrics['source_counts']['memory']}, "
                f"retrieval={metrics['source_counts']['retrieval']}, "
                f"other={metrics['source_counts']['other']}"
            )
            print("latest_run:")
            print(f"  timestamp       : {latest['timestamp']}")
            print(f"  source          : {latest['source']}")
            print(f"  confidence      : {_fmt(latest['confidence'])}")
            print(f"  latency_s       : {_fmt(latest['latency'])}")
            print(f"  memory_size     : {latest['memory_size']}")
            print(f"  retrieval_count : {latest['retrieval_count']}")
            print(f"  memory_count    : {latest['memory_count']}")
            continue

        if not is_valid_query(query):
            print("\nPlease enter a valid question (text with meaningful words).")
            continue

        answer, confidence = router.process_query(query)

        print(f"\nConfidence : {confidence:.2f}")
        print(f"Answer     : {answer}")