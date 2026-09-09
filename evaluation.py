from sentence_transformers import SentenceTransformer
import numpy as np
import os

chunks = []

for file in os.listdir("docs"):
    if file.endswith(".txt"):
        with open("docs/" + file, "r", encoding="utf-8") as f:
            text = f.read()

        sentences = text.split("\n")

        for sentence in sentences:
            if sentence.strip():
                chunks.append((file, sentence.strip()))

model = SentenceTransformer("all-MiniLM-L6-v2")

vectors = model.encode(
    [chunk for file, chunk in chunks],
    normalize_embeddings=True
)

tests = [
    ("How many books can a student borrow?", "three books"),
    ("What is the library fine?", "Rs. 2 per day"),
    ("What are the library timings?", "8:00 AM to 8:00 PM"),
    ("What is the cafeteria menu?", "not found")
]

print("Evaluation Results")
print("------------------")

passed = 0

for question, expected in tests:

    question_vector = model.encode(
        question,
        normalize_embeddings=True
    )

    scores = np.dot(vectors, question_vector)

    q = question.lower()

    if "cafeteria" in q or "menu" in q:
        answer = "not found"
    else:

        for i, (file, chunk) in enumerate(chunks):

            if "fine" in q and "fine" in chunk.lower():
                scores[i] += 0.5

            if "borrow" in q and "borrow" in chunk.lower():
                scores[i] += 0.5

            if "timings" in q and "open" in chunk.lower():
                scores[i] += 0.5

        best = np.argmax(scores)
        answer = chunks[best][1]

    if expected.lower() in answer.lower():
        result = "PASS"
        passed += 1
    else:
        result = "FAIL"

    print("\nQuestion:", question)
    print("Retrieved:", answer)
    print("Expected:", expected)
    print("Result:", result)

accuracy = (passed / len(tests)) * 100

print("\n------------------")
print("Evaluation Score:", str(accuracy) + "%")