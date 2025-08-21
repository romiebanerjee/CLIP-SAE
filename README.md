
## CLIP-SAE: A repo of sparse auto-encoders (SAE) trained on OpenCLIP model embeddings. \href{https://github.com/romiebanerjee/CLIP-SAE}{GitHib}
 - **Models**: clip-vit-base-patch32, CLIP-ViT-bigG-14-laion2B-39B-b160k
 - **Dataset**: LAION400M. 
 - Training scripts with vanilla SAE with L1 reg for smaller model
 - TopK activation, sparse GPU kernels and data parallel for bigger models
 - Sparse kernels Triton code adapted from https://github.com/openai/sparse_autoencoder