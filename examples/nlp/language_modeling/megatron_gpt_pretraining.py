# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# To suppress BF16 compile related issue in the CI runs with turing/V100
import torch._dynamo
import torch.multiprocessing as mp

from omegaconf.omegaconf import OmegaConf, open_dict
from typing import Dict, Any, Optional, Union
from argparse import Namespace

from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from apex import amp
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

torch._dynamo.config.suppress_errors = True

mp.set_start_method("spawn", force=True)

def calculate_model_size(
    vocab_size: int = None,
    seq_length: int = None,
    hidden_size: int = None,
    num_layers: int = None,
    ffn_size: int = None,
    att_heads: int = None,
    model_name: str = "gpt3",
):
    model_size = (
        12
        * num_layers
        * hidden_size ** 2
        * (1 + (13 / (12 * hidden_size)) + ((vocab_size + seq_length) / (12 * num_layers * hidden_size)))
        / 1e9
    )
    return model_size

def throughput_calculator(model, cfg, iteration_time, total_iterations):
    if total_iterations == 0.0:
        return 0, 0, 0
    world_size = torch.distributed.get_world_size()
    micro_batch_size = cfg.model.micro_batch_size
    data_parallel_size = parallel_state.get_data_parallel_world_size()
    batch_size = micro_batch_size * get_num_microbatches() * data_parallel_size
    samples_per_model = batch_size * cfg.model.data.seq_length

    hidden_size = cfg.model.hidden_size
    num_layers = cfg.model.num_layers
    vocab_size = model.padded_vocab_size
    ffn_hidden_size = cfg.model.ffn_hidden_size
    att_heads = cfg.model.num_attention_heads
    model_size = calculate_model_size(vocab_size, cfg.model.data.seq_length, hidden_size, num_layers, ffn_hidden_size, att_heads)

    samples_per_second = batch_size / iteration_time

    #flops calculator
    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # The factor of 4 is when used with activation check-pointing,
    # otherwise it will be 3.
    checkpoint_activations_factor = 3
    if cfg.model.activations_checkpoint_granularity == 'selective':
            checkpoint_activations_factor = 4
    seq_len = cfg.model.data.seq_length
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
    tflops = flops_per_iteration / (iteration_time * world_size * (10**12))
    return samples_per_second, tflops, model_size

# ref: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.logger.html#lightning.pytorch.loggers.logger.Logger

class MetricsLogger(TensorBoardLogger):
    def __init__(self, trainer, model, cfg,
                 train_loss_key='reduced_train_loss', val_loss_key='val_loss',
                 timing_keys=('train_step_timing', 'train_epoch_timing', 'validation_step_timing', 'validation_epoch_timing'),
                 throughput_key='train_epoch_timing'):
        super().__init__()
        self.trainer = trainer
        self.model = model
        self.cfg = cfg
        self.val_loss_key = val_loss_key
        self.train_loss_key = train_loss_key
        self.timing_keys = timing_keys
        self.throughput_key = throughput_key
        self.target_val_log_ppl = 10.82
        self._experiment = None

    def log_metrics(self, metrics: Dict[str, float],
                    step: Optional[int] = None) -> None:
        if "validation_step_timing" in metrics:
            elapsed_time = metrics["validation_step_timing"]
            log_string = ' Step {}/{}: |'.format(step, self.trainer.num_val_batches)
            log_string += ' validation_step_timing {:.2f}: |'.format(metrics["validation_step_timing"])
        elif "train_step_timing" in metrics:
            elapsed_time = metrics["train_step_timing"]
            total_iterations = metrics["global_step"]
            samples_per_sec, tflops, approx_parameters_in_billions = throughput_calculator(self.model, self.cfg, elapsed_time, total_iterations)
            tokens_per_sec = samples_per_sec * self.cfg.model.data.seq_length

            log_string = ' Epoch {}: iteration {}/{} |'.format(metrics["epoch"], self.trainer.global_step, self.trainer.max_steps)
            log_string += ' train_step_timing (s): {:.2f} |'.format(metrics["train_step_timing"])
            # log_string += ' global_step: {} |'.format(metrics["global_step"])
            log_string += ' reduced_train_loss: {:.2f} |'.format(metrics["reduced_train_loss"])
            log_string += ' lr: {:.3E} |'.format(metrics["lr"])
            log_string += ' consumed samples: {} |'.format(metrics["consumed_samples"])
            log_string += ' samples per sec: {:.2f} |'.format(samples_per_sec)
            log_string += ' tokens per sec: {:.2f} |'.format(tokens_per_sec)
            log_string += ' TFLOPs: {:.2f} |'.format(tflops)
            log_string += ' Model Size(B): {:.2f} |'.format(approx_parameters_in_billions)
            log_string += ' Global Batch Size: {} |'.format(self.cfg.model.global_batch_size)
            log_string += ' Micro Batch Size: {}'.format(self.cfg.model.micro_batch_size)
            # log_string += ' loss_scale: {:1f} |'.format(metrics["loss_scale"])
            # log_string += ' train_backward_timing: {:.2f} |'.format(metrics["train_backward_timing"])
            # log_string += ' grad_norm: {:.2f}'.format(metrics["grad_norm"])
        else:
            log_string = str(metrics)
        logging.info(log_string)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace],
                        *args: Any, **kwargs: Any) -> None:
        model_cfg = params.cfg
 
    @property
    def name(self) -> Optional[str]:
        return 'nemo-metrics-logger'

    @property
    def version(self) -> Optional[Union[int, str]]:
        return 1
    
    @property
    def experiment(self):
        """Return the experiment object associated with this logger."""
        return "experiment"

@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model = MegatronGPTModel(cfg.model, trainer)

    trainer.loggers.append(MetricsLogger(trainer, model, cfg))
    trainer.fit(model)


if __name__ == '__main__':
    main()
