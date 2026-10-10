# Upstream projects

The repository integrates ideas through stable interfaces and does not vendor upstream code.

- Gorse (`gorse-io/gorse`, Apache-2.0): optional production recommendation service.
- Qdrant (`qdrant/qdrant`, Apache-2.0): dense text and image vector storage.
- Chinese-CLIP (`OFA-Sys/Chinese-CLIP`, MIT): Chinese text-to-image and image-to-image retrieval.
- vLLM (`vllm-project/vllm`, Apache-2.0): optional OpenAI-compatible local model server.
- OpenP5 (`agiresearch/OpenP5`, Apache-2.0): experiment reference for LLM recommendation.
- ToolRec (`Go0day/ToolRec-Code`): research reference for recommendation tool use; unmaintained.
- FlagEmbedding (`FlagOpen/FlagEmbedding`, MIT): server embedding and reranking provider.
- Qwen3-VL (`QwenLM/Qwen3-VL`, Apache-2.0 repository): multimodal provider; verify each checkpoint license.
- Amazon Berkeley Objects (Amazon/UC Berkeley, CC BY 4.0): official listing metadata and image URLs
  used to build the balanced local demo catalog. See `data/demo/ATTRIBUTION.md`.

Pin an upstream commit before importing code or results. Keep its license and attribution in the resulting artifact.

