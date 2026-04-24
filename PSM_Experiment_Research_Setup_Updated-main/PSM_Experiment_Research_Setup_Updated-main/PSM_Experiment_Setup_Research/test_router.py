from router.router import Router

documents = [
    "Artificial Intelligence is the simulation of human intelligence.",
    "Machine learning is a subset of AI that learns from data.",
    "Deep learning uses neural networks.",
    "Natural language processing deals with text and language.",
    "Computer vision deals with images.",
    "Reinforcement learning learns using rewards and punishments.",
    "Transformers are used in modern NLP models.",
    "Large language models are trained on massive text data."
]

router = Router(documents=documents)

# Build hybrid retriever index
router.hybrid_retriever.build_index(documents)

while True:
    query = input("\nEnter query: ")

    result = router.route(query)

    print("\nSource:", result["source"])
    print("Confidence:", result["confidence"])
    print("Documents:")
    for doc in result["docs"]:
        print("-", doc)

    # Fake answer for now
    answer = input("\nEnter generated answer (simulate LLM): ")

    if result["source"] == "retrieval":
        router.store_memory(query, result["docs"], answer)