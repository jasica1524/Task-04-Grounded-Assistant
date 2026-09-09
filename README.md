# College Grounded Assistant

## Project Overview

College Grounded Assistant is a simple Retrieval-Augmented Generation (RAG) based assistant designed to answer college-related questions using information from provided documents.

The assistant retrieves relevant information from the documents and provides the retrieved passage as the answer. It also includes guardrails to prevent unrelated questions from being answered.

## Purpose

The purpose of this project is to create a grounded assistant that:

- Answers questions using provided documents
- Shows the source of the information
- Avoids making up information
- Refuses unrelated questions
- Handles information that is not available in the documents
- Measures approximate token usage and estimated cost

## Grounding Document

The assistant currently uses:

`docs/college_rules.txt`

The document contains information about:

- Library opening hours
- Number of books students can borrow
- Book issue duration
- Library fines
- College ID requirements
- Library holidays

## How It Works

1. The assistant loads the documents from the `docs` folder.
2. The documents are divided into smaller passages.
3. SentenceTransformer converts the passages into numerical embeddings.
4. The user's question is converted into an embedding.
5. The system compares the question with the document passages.
6. The most relevant passage is retrieved.
7. The retrieved passage is displayed as the answer.
8. The source document is displayed.

## Guardrails

The assistant contains basic guardrails to keep responses within its intended purpose.

For unrelated questions, the assistant responds:

> I can only answer questions related to college and internship information.

If the requested information cannot be found in the provided documents, the assistant responds:

> I could not find this information in the provided documents.

## Evaluation

The assistant was tested using four evaluation questions.

| Question | Expected Result | Result |
|---|---|---|
| How many books can a student borrow? | Three books | PASS |
| What is the library fine? | Rs. 2 per day | PASS |
| What are the library timings? | 8:00 AM to 8:00 PM | PASS |
| What is the cafeteria menu? | Not found | PASS |

### Evaluation Score

**100%**

## Token Usage and Cost

The project measures the approximate number of tokens used in user questions.

Example test:

- Questions asked: 2
- Approximate tokens used: 12
- Estimated API cost: $0.00

The project uses a local SentenceTransformer model, so no paid API is required.

## Technologies Used

- Python
- NumPy
- Sentence Transformers
- RAG
- Document Retrieval
- Embeddings
- Cosine Similarity

## Project Structure

```text
Task-04-Grounded-Assistant
│
├── docs
│   └── college_rules.txt
│
├── app.py
├── evaluation.py
├── requirements.txt
└── README.md
