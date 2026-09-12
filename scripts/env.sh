# Source before any run: keeps model weights off the small root partition.
export HF_HOME=/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/Probabilistic-Coherence-as-a-Test-of-Clinical-Reasoning-in-Language-Models/build/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
export PATH=/home/george/.local/bin:$PATH
# Blackwell sm_120: FlashInfer JIT-compiles with the system nvcc (CUDA 12.4),
# which cannot target sm_120, so its sampler is disabled and vLLM falls back
# to the PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
export TORCH_CUDA_ARCH_LIST=12.0
