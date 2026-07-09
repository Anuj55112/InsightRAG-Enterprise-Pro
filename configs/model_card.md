# Model Card: InsightRAG Pro

This model card documents the specifications and details of the analysis, routing, and scoring models deployed within the **InsightRAG Pro** platform.

## Model Details
- **Developer**: Portfolio Owner
- **Model Types**:
  - **QueryRouter**: Categorizes query intent into vector-search, graphrag, comparison, or standard hybrid routing buckets using heuristic mappings.
  - **SentenceEmbedder**: Text embeddings created using the SentenceTransformer `all-MiniLM-L6-v2` architecture.
  - **CrossEncoderReranker**: Reranking relevance scorer utilizing the `cross-encoder/ms-marco-MiniLM-L-6-v2` pipeline.
  - **KnowledgeGraph**: Stateful GraphRAG network mapping entity extractions using NetworkX.
  - **RAGBenchEvaluator**: Measures contextual relevance, faithfulness, correctness, and hallucination scores mathematically.
- **Task**: Enterprise knowledge retrieval, multi-document synthesis, and RAG quality evaluations.

## Intended Use
- **Primary Intended Use**: Research, benchmarking, and demonstration of production-grade Enterprise RAG systems containing structural GraphRAG and evaluation verification checkers.
- **Out of Scope Use**: High-stakes legal or clinical diagnosis applications without human verification checkpoints.

## Datasets & Preprocessing
- **Source Data**: Multi-format local documents (PDF, text, Markdown) ingested dynamically.
- **Preprocessing**: Recursive character chunking preserving word boundaries and table layouts.
