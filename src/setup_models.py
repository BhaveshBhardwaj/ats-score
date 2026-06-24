#!/usr/bin/env python3
"""
Pre-downloads the FastEmbed ONNX models so the pipeline can run completely offline.
This script MUST be run before executing rank.py in an environment with no network access.
"""

import os
from fastembed import TextEmbedding

def setup():
    print("=" * 60)
    print(" NEXUS v2 — Local Model Setup")
    print("=" * 60)
    print("\nDownloading BAAI/bge-small-en-v1.5 (if not already cached)...")
    
    # Initialize the model. This triggers the download and caches it in ~/.cache/fastembed
    # or the specified cache_dir.
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Do a quick test to ensure it works
    print("Testing inference...")
    embeddings = list(model.embed(["Test document for setup."]))
    
    if len(embeddings) > 0:
        print(f"Success! Model cached and ready for offline use. (Dim: {len(embeddings[0])})")
    else:
        print("Failed to generate test embedding.")
        
    print("\nYou can now run rank.py with the network disabled.")

if __name__ == "__main__":
    setup()
