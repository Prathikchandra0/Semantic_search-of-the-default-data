🔎 Semantic Search with Sentence Transformers

<p align="center">
  <img src="semantic-search-flow.gif" alt="Animated semantic search workflow" width="850">
</p>


Use this as your `README.md`:

````markdown
# Semantic Search using Sentence Transformers

## Overview

This project demonstrates a basic **Semantic Search** system using
Sentence Transformers and Cosine Similarity.

The system retrieves documents from a predefined knowledge base based on
their **semantic meaning**, rather than relying only on exact keyword
matching.

The implementation represents the fundamental retrieval component used
in modern **Retrieval-Augmented Generation (RAG)** systems.

---

## Objective

The objective of this project is to understand how a semantic search
system can:

1. Convert text into numerical embeddings.
2. Represent documents in a vector space.
3. Convert a user query into an embedding.
4. Measure similarity between the query and documents.
5. Rank documents according to their similarity.
6. Retrieve the most relevant documents using Top-K search.

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Implementation |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Pre-trained embedding model |
| NumPy | Vector and mathematical operations |
| Cosine Similarity | Measuring semantic similarity |
| Jupyter Notebook | Development and experimentation |

---

## System Workflow

```text
                    User Query
                        |
                        v
              Sentence Transformer
                        |
                        v
                 Query Embedding
                        |
                        v
             +----------------------+
             | Cosine Similarity     |
             |                      |
             | Query vs Documents   |
             +----------------------+
                        |
                        v
                Similarity Scores
                        |
                        v
                  Sort Results
                        |
                        v
                   Top-K Results
````

---

## Knowledge Base

The project uses a small knowledge base containing machine-learning
concepts such as:

* Linear Regression
* Decision Trees
* Random Forests
* Overfitting
* Cross-Validation
* Feature Scaling
* Logistic Regression
* K-Means Clustering
* Gradient Descent
* Regularization
* Confusion Matrix
* Precision
* Recall
* ROC Curve
* SMOTE

Each document is converted into an embedding before performing the
search.

---

## Embedding Generation

The project uses the following Sentence Transformer model:

```python
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
```

The knowledge-base documents are converted into embeddings using:

```python
kb_embeddings = model.encode(knowledge_base)
```

Conceptually:

```text
Document
   |
   v
Sentence Transformer
   |
   v
Numerical Vector
```

For example:

```text
"Overfitting occurs when a model memorizes training data."
                         |
                         v
              [0.12, -0.43, 0.87, ...]
```

The numerical vector represents the semantic characteristics of the
sentence.

---

## Semantic Search Process

When a user enters a query, the query is also converted into an
embedding:

```python
query_emb = model.encode(query)
```

The query embedding is then compared with every document embedding.

```text
Query Embedding
       |
       +------------------+
       |                  |
       v                  v
Document 1           Document 2
       |                  |
       +--------+---------+
                |
                v
       Cosine Similarity
                |
                v
        Similarity Scores
```

---

## Cosine Similarity

Cosine similarity measures the similarity between two vectors based on
the angle between them.

The implementation uses:

```python
def cosine_similarity(vec1, vec2):

    vec1, vec2 = np.array(vec1), np.array(vec2)

    return float(np.dot(vec1, vec2)) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )
```

A higher similarity score indicates that two embeddings are more closely
related in the embedding space.

The score is primarily used to **rank the retrieved documents**.

---

## Ranking and Top-K Retrieval

For every document, the system stores:

```python
(similarity_score, document_index)
```

Example:

```text
Similarity       Document Index
--------------------------------
0.91             3
0.84             9
0.78             4
0.65             1
0.52             7
```

The results are sorted in descending order:

```python
result.sort(reverse=True)
```

If:

```python
top_k = 3
```

only the three highest-ranked documents are returned.

```python
result[:top_k]
```

---

## Example

### Input Query

```text
How do I stop my model from memorizing the data?
```

The knowledge base does not contain this exact sentence.

However, it contains information about **overfitting**:

```text
Overfitting happens when a model memorizes the training
data including noise, performing great on training but
poorly on new data.
```

The semantic search system identifies this document because the
**meaning of the query is related to the meaning of the document**.

### Retrieval Result

| Rank | Topic            | Relevance |
| ---- | ---------------- | --------- |
| 1    | Overfitting      | High      |
| 2    | Regularization   | High      |
| 3    | Cross-Validation | Related   |

The exact similarity values depend on the embedding model and input.

---

## Keyword Search vs Semantic Search

| Feature                    | Keyword Search | Semantic Search |
| -------------------------- | -------------- | --------------- |
| Matching method            | Exact words    | Meaning         |
| Requires same wording      | Usually        | No              |
| Uses embeddings            | No             | Yes             |
| Handles paraphrasing       | Limited        | Better          |
| Vector similarity          | No             | Yes             |
| Suitable for RAG retrieval | Limited        | Common approach |

### Example

Keyword-based search:

```text
Query:
"model memorizing data"

Document:
"Overfitting occurs when a model memorizes training data."

Result:
Possible keyword match
```

Semantic search:

```text
Query:
"Why does my model remember the training examples too well?"

Document:
"Overfitting occurs when a model memorizes training data."

Result:
Semantically related
```

---

## Implementation

The complete search process is implemented using:

```python
def semantic_Search(query, top_k=3):

    query_emb = model.encode(query)

    result = []

    for i, doc_emb in enumerate(kb_embeddings):

        sim = cosine_similarity(query_emb, doc_emb)

        result.append((sim, i))

    result.sort(reverse=True)

    for rank, (sim, idx) in enumerate(result[:top_k], 1):

        print(f"\n#{rank} [similarity: {sim:.4f}]")
        print(knowledge_base[idx])

    return result[:top_k]
```

---

## Architecture

```text
+----------------------+
|    Knowledge Base    |
+----------+-----------+
           |
           v
+----------------------+
| Sentence Transformer |
+----------+-----------+
           |
           v
+----------------------+
| Document Embeddings  |
+----------+-----------+
           |
           |
           |              +----------------+
           |              |   User Query   |
           |              +-------+--------+
           |                      |
           |                      v
           |              +----------------------+
           |              | Sentence Transformer |
           |              +----------+-----------+
           |                         |
           |                         v
           |                Query Embedding
           |                         |
           +------------+------------+
                        |
                        v
              +--------------------+
              | Cosine Similarity  |
              +---------+----------+
                        |
                        v
              +--------------------+
              | Similarity Ranking |
              +---------+----------+
                        |
                        v
              +--------------------+
              |    Top-K Results   |
              +--------------------+
```

---

## Relation to RAG

Semantic search represents the **retrieval component** of a RAG system.

This project:

```text
User Query
     |
     v
Semantic Search
     |
     v
Relevant Documents
```

A complete RAG system extends this process:

```text
User Query
     |
     v
Semantic Search
     |
     v
Relevant Documents
     |
     v
LLM + Retrieved Context
     |
     v
Generated Answer
```

Therefore, this project provides the foundation for understanding how
documents can be retrieved before supplying them as context to a
language model.

---

## Project Structure

```text
Semantic_Search/
|
+-- Semantic_Search.ipynb
+-- README.md
+-- semantic-search-flow.gif
```

---

## Installation

Install the required libraries:

```bash
pip install sentence-transformers numpy
```

Then open:

```text
Semantic_Search.ipynb
```

and execute the notebook cells sequentially.

---

## Key Concepts Demonstrated

* Text Embeddings
* Sentence Transformers
* Vector Representation
* Cosine Similarity
* Semantic Similarity
* Document Ranking
* Top-K Retrieval
* Knowledge Base Retrieval
* Foundations of RAG

---

## Future Improvements

The current implementation can be extended by:

1. Adding a larger document collection.
2. Implementing document chunking.
3. Adding a vector database.
4. Introducing similarity thresholds.
5. Integrating an LLM for answer generation.
6. Building a complete RAG pipeline.
7. Exposing the search system through a REST API.
8. Creating a web-based user interface.

---

## Conclusion

This project demonstrates how semantic search can retrieve relevant
information based on **meaning rather than exact keyword matching**.

It provides a practical introduction to embeddings, vector similarity,
document ranking, and the retrieval stage of RAG systems.

````

### I would also change the GitHub layout

Instead of making the GIF the main visual element, keep the README **professional**:

```text
# Semantic Search using Sentence Transformers

Overview
   ↓
Objective
   ↓
Technologies
   ↓
System Workflow
   ↓
Knowledge Base
   ↓
Embedding Generation
   ↓
Cosine Similarity
   ↓
Ranking & Top-K
   ↓
Practical Example
   ↓
Architecture
   ↓
RAG Connection
   ↓
Installation
   ↓
Future Improvements
````

This presentation makes it much clearer that **you built a semantic retrieval system**, rather than making the repository look like a generic AI demo.
