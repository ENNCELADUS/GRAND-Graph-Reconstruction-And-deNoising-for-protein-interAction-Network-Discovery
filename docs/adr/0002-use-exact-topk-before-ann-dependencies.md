# Use exact top-k before ANN dependencies

Accepted. TCCIG retrieval uses chunkable exact torch top-k for the first graph-prior retrieval implementation instead of adding FAISS, HNSW, PyG, or DGL dependencies. The current human/BFS PRING scale can validate the modeling change without destabilizing `uv.lock` or the HPC environment, and the config keeps `retrieval.backend: exact` so an ANN backend can be introduced later behind the same interface.
