from dataclasses import dataclass
from typing import Optional, Tuple, Union

import os, sys
import numpy as np
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint


class MoELayer(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules,
        is_expert: bool = False,
        input_size: int = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size != None else self.config.hidden_size

        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.activation_func = self.config.activation_func

        # python3 -m pip install -v -U --no-build-isolation git+https://github.com/microsoft/tutel@main
        from tutel import moe as tutel_moe
        import logging
        logging.getLogger().setLevel(logging.WARNING)

        # from megatron.core.extensions.transformer_engine import TENorm
        self.layer_norm = lambda x : x # TENorm(config, self.input_size)

        self.local_moe_layer = tutel_moe.moe_layer(
            gate_type={'type': 'top', 'k': int(os.environ.get('TOP', 1)), 'capacity_factor': float(os.environ.get('C', -1.2))},
            model_dim=self.input_size,
            experts={
                'count_per_node': int(os.environ.get('TUT', -int(os.environ['WORLD_SIZE']))),
                'type': 'ffn', 'hidden_size_per_expert': ffn_hidden_size, 'activation_fn': self.activation_func,
                'has_fc1_bias': False, 'has_fc2_bias': False,
            },
            parallel_type = 'adaptive:' + os.environ.get('R', '1'),
            a2a_ffn_overlap_degree = int(os.environ.get('O', 1)),
            scan_expert_func = lambda name, param: setattr(param, 'allreduce', False),
        )
        # self.config.init_method
        # self.config.output_layer_init_method
        # self.config.add_bias_linear

    def forward(self, hidden_states):
        y = self.local_moe_layer(self.layer_norm(hidden_states))
        return y, None

        '''
        if self.config.bias_activation_fusion:
            if self.activation_func == F.gelu:
                if self.config.gated_linear_unit:
                    intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                else:
                    assert self.config.add_bias_linear is True
                    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
            elif self.activation_func == F.silu and self.config.gated_linear_unit:
                intermediate_parallel = bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    self.config.activation_func_fp8_input_store,
                )
            else:
                raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)
        '''


if int(os.environ.get('MLP', 0)) == 1:
  from megatron.core.transformer.mlp import MLP
  MoELayer = MLP

