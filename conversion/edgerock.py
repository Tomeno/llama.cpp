from __future__ import annotations

from .base import ModelBase, gguf
from .qwen import Qwen3_5TextModel


@ModelBase.register("EdgeRockForCausalLM")
@ModelBase.example("rockmatrix-ai/EdgeRock")
class EdgeRockModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.EDGEROCK
    no_mtp = True  # the MTP head is not used by llama.cpp

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        n_layer = self.hparams["num_hidden_layers"]
        swa_layers = sorted(self.hparams["fixed_memory_swa_layers"])
        attn_layers = [il for il, t in enumerate(self.hparams["layer_types"][:n_layer]) if t != "linear_attention"]
        if swa_layers != attn_layers:
            raise ValueError(f"fixed_memory_swa_layers {swa_layers} must be the attention layers {attn_layers}")

        mem = self.hparams["independent_memory"]
        arch = self.gguf_writer.arch
        self.gguf_writer.add_sliding_window(self.hparams["fixed_memory_swa_window"])
        self.gguf_writer.add_uint32(gguf.Keys.Memory.CAPACITY.format(arch=arch),            mem["capacity"])
        self.gguf_writer.add_uint32(gguf.Keys.Memory.COMMIT_CHUNK.format(arch=arch),        mem["max_commit_chunk"])
        self.gguf_writer.add_uint32(gguf.Keys.Memory.ADMISSION_RANK.format(arch=arch),      mem["admission_rank"])
        self.gguf_writer.add_uint32(gguf.Keys.Memory.WRITER_INTERMEDIATE.format(arch=arch), mem["writer_intermediate_size"])

    # the memory norms (memory.{input,q,k}_norm) use the same 1 + w RMSNorm as the rest of the model,
    # the parent adds the 1 to every *norm.weight that is not the gated delta-net norm
