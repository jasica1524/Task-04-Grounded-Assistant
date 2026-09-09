from sentence_transformers import SentenceTransformer
import numpy as np
import os

chunks = []
total_questions = 0
total_tokens = 0

for file in os.listdir("docs"):
    if file.endswith(".txt"):
        with open("docs/" + file, "r", encoding="utf-8") as f:
            text = f.read()

        sentences = text.split("\n")

        for sentence in sentences:
            if sentence.strip():
                chunks.append((file, sentence.strip()))

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_vectors = model.encode(
    [chunk for file, chunk in chunks],
    normalize_embeddings=True
)

print("College Grounded Assistant is ready!")

while True:

    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    total_questions += 1
    total_tokens += len(question.split())

    allowed_words = [
        "college", "library", "book", "books", "student",
        "internship", "campus", "course", "class", "exam",
        "attendance", "faculty", "university"
    ]

    if not any(word in question.lower() for word in allowed_words):
        print("\nAnswer:")
        print("I can only answer questions related to college and internship information.")
        continue

    question_vector = model.encode(
        question,
        normalize_embeddings=True
    )

    scores = np.dot(chunk_vectors, question_vector)

    for i, (file, chunk) in enumerate(chunks):

        if "fine" in question.lower() and "fine" in chunk.lower():
            scores[i] += 0.5

        if "borrow" in question.lower() and "borrow" in chunk.lower():
            scores[i] += 0.5

        if "open" in question.lower() and "open" in chunk.lower():
            scores[i] += 0.5

    best = np.argmax(scores)

    if scores[best] < 0.35:
        print("\nAnswer:")
        print("I could not find this information in the provided documents.")
        continue

    file, answer = chunks[best]

    print("\nAnswer:")
    print(answer)

    print("\nSource:")
    print(file)

    print("\nRetrieved Passage:")
    print(answer)

print("\nUsage Report")
print("------------")
print("Questions asked:", total_questions)
print("Approximate tokens used:", total_tokens)
print("Estimated API cost: $0.00")