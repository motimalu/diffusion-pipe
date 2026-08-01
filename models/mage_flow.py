# Adapted from MAGE (https://github.com/microsoft/Mage)                                                                                                                                                                           
# Original code licensed under MIT: https://opensource.org/licenses/MIT

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import ComfyPipeline
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift
from utils.offloading import ModelOffloader
import comfy.ldm.common_dit

torch.set_float32_matmul_precision('high')

class MageFlowInitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        # DeepSpeed pipeline passes multiple arguments as a single packed tuple
        # Training: (x, timestep, context, attention_mask, ref_latents)
        # Sampling via CommonPipeline.sample(): (x, timestep, context, attention_mask)
        if len(inputs) == 5:
            x, timestep, context, attention_mask, ref_latents = inputs
        else:
            x, timestep, context, attention_mask = inputs
            ref_latents = None

        if attention_mask is not None and not torch.is_floating_point(attention_mask):
            attention_mask = (attention_mask - 1).to(x.dtype) * torch.finfo(x.dtype).max

        hidden_states, img_ids, orig_shape = self.model.process_img(x)
        num_embeds = hidden_states.shape[1]

        ref_num_tokens = []
        if ref_latents is not None:
            index = 0
            for ref in ref_latents:
                index += 1
                kontext, kontext_ids, _ = self.model.process_img(ref, index=index)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                ref_num_tokens.append(kontext.shape[1])

        txt_ids = torch.zeros((x.shape[0], context.shape[1], 3), device=x.device)

        hidden_states = self.model.img_in(hidden_states)
        context = self.model.txt_norm(context)
        context = self.model.txt_in(context)

        temb = self.model.time_text_embed(timestep, hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.model.pe_embedder(ids).contiguous()

        # Encode metadata into tensors to survive DeepSpeed pipeline communication boundaries
        ref_tokens_tensor = torch.tensor(ref_num_tokens, dtype=torch.long, device=x.device)
        orig_shape_tensor = torch.tensor(orig_shape, dtype=torch.long, device=x.device)
        num_embeds_tensor = torch.tensor([num_embeds], dtype=torch.long, device=x.device)

        # Return as a packed tuple for the next pipeline layer
        return (hidden_states, context, attention_mask, temb, image_rotary_emb, ref_tokens_tensor, num_embeds_tensor, orig_shape_tensor)


class MageFlowBlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        # Unpack tuple received from the previous pipeline stage
        hidden_states, context, attention_mask, temb, image_rotary_emb, ref_tokens_tensor, num_embeds_tensor, orig_shape_tensor = inputs

        # Reconstruct transformer_options dictionary from the transmitted tensor
        transformer_options = {}
        if ref_tokens_tensor.numel() > 0:
            transformer_options["reference_image_num_tokens"] = ref_tokens_tensor.tolist()

        context, hidden_states = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=context,
            encoder_hidden_states_mask=attention_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            transformer_options=transformer_options,
        )

        # Return as a packed tuple for the next pipeline layer
        return (hidden_states, context, attention_mask, temb, image_rotary_emb, ref_tokens_tensor, num_embeds_tensor, orig_shape_tensor)


class MageFlowFinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        # Unpack tuple received from the last transformer block
        hidden_states, context, attention_mask, temb, image_rotary_emb, ref_tokens_tensor, num_embeds_tensor, orig_shape_tensor = inputs

        hidden_states = self.model.norm_out(hidden_states, temb)
        hidden_states = self.model.proj_out(hidden_states)

        num_embeds = num_embeds_tensor.item()
        hidden_states = hidden_states[:, :num_embeds]

        h, w = orig_shape_tensor.tolist()

        # Final layer returns just the processed output tensor
        return hidden_states.reshape(hidden_states.shape[0], h, w, self.model.out_channels).movedim(-1, 1)


class MageFlowPipeline(ComfyPipeline):
    name = 'mage_flow'
    checkpointable_layers = ['MageFlowBlockWrapper']
    adapter_target_modules = ['QwenImageTransformerBlock']

    # MageFlow uses a latent-consistency VAE with 128 channels (not 16 like SD/Flux)
    channels = 128
    spatial_compression = 16

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        # MageFlow uses a static shift of 6.0 at inference (FlowMatchEulerDiscreteScheduler).
        self.sample_shift = 6.0

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = []

        layers.append(MageFlowInitialLayer(diffusion_model))

        if hasattr(diffusion_model, 'transformer_blocks'):
            for block in diffusion_model.transformer_blocks:
                layers.append(MageFlowBlockWrapper(block))

        layers.append(MageFlowFinalLayer(diffusion_model))

        return layers

    def get_conds(self, inputs):
        text_embeds = inputs['text_embeds_0']
        attention_mask = inputs['attention_mask_0']

        max_seq_len = max([e.size(0) for e in text_embeds])
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in text_embeds]
        )
        attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attention_mask]
        )
        assert text_embeds.shape[:2] == attention_mask.shape[:2]
        attention_mask = attention_mask.to(torch.bool)
        return text_embeds, attention_mask

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        latents = self.model_patcher.model.process_latent_in(latents)
        mask = inputs['mask']

        conds = self.get_conds(inputs)

        bs, c, h, w = latents.shape
        device = latents.device

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        # MageFlow uses raw t in [0,1] at training time (no shift).
        # The scheduler shift (default 6.0) is applied only at inference by
        # FlowMatchEulerDiscreteScheduler. Applying it here would double-shift
        # timesteps at inference, compressing all signal into the extreme
        # high-noise regime and destroying denoising ability.
        if self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        if 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()
            control_latents = self.model_patcher.model.process_latent_in(control_latents)
            assert control_latents.shape == latents.shape, (
                f"Control latents shape {control_latents.shape} doesn't match latents shape {latents.shape}"
            )
            extra_inputs = ([control_latents],)
        else:
            extra_inputs = (None,)

        return (noisy_latents, t, *conds) + extra_inputs, (target, mask)
