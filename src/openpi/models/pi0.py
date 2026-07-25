import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def _huber(x: at.Array, delta: float = 0.1) -> at.Array:
    """Elementwise Huber / smooth-L1 (robust to fuzzy subgoal-boundary labels)."""
    abs_x = jnp.abs(x)
    return jnp.where(abs_x <= delta, 0.5 * jnp.square(x), delta * (abs_x - 0.5 * delta))


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # --- System1: configurable image keys, anchor role embedding, progress head ---
        base_image_keys = tuple(config.image_keys) if config.image_keys is not None else _model.IMAGE_KEYS
        # When anchors are enabled, the anchor_* groups must ALSO be in the model's
        # image-key set, or preprocess_observation drops them and the backbone never
        # sees the "before" views. (inputs_spec adds them the same way.)
        if config.use_anchor_images:
            self._image_keys = base_image_keys + tuple(f"anchor_{k}" for k in base_image_keys)
        else:
            self._image_keys = base_image_keys
        self._geometric_aug_cameras = (
            tuple(config.geometric_aug_cameras) if config.geometric_aug_cameras is not None else None
        )
        self._use_anchor_images = config.use_anchor_images
        self._flow_loss_real_dim = config.flow_loss_real_dim
        self._use_progress_head = config.use_progress_head
        self._progress_readout = config.progress_readout
        self._progress_stop_gradient = config.progress_stop_gradient
        self._progress_loss_weight = config.progress_loss_weight
        self._progress_k = config.progress_k
        self._progress_mode = config.progress_mode
        self._progress_num_classes = config.progress_num_classes
        self._progress_binary_pos_classes = config.progress_binary_pos_classes
        self._progress_pos_weight = config.progress_pos_weight
        if config.use_anchor_images:
            # Learned role embedding {current, anchor} added to each image group's tokens
            # so the model can distinguish before vs now (the prefix is a bidirectional
            # block; image order alone is a weak cue). Shape (2, paligemma_width).
            self.image_role_embedding = nnx.Param(
                nnx.initializers.normal(0.02)(rngs.params(), (2, paligemma_config.width))
            )
        if config.use_progress_head:
            from openpi.models import progress_head as _progress_head

            self.progress_head = _progress_head.ProgressHead(
                paligemma_config.width,
                readout=config.progress_readout,
                num_layers=config.progress_num_layers,
                num_heads=config.progress_num_heads,
                hidden=config.progress_hidden,
                num_outputs=(config.progress_num_classes if config.progress_mode == "classes" else 1),
                rngs=rngs,
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            # System1: add a learned anchor/current role embedding so the model can
            # tell "before" (anchor_*) views apart from the current views (the prefix
            # is one bidirectional block; token order alone is a weak cue).
            if self._use_anchor_images and hasattr(self, "image_role_embedding"):
                role = 1 if str(name).startswith("anchor_") else 0
                image_tokens = image_tokens + self.image_role_embedding.value[role][None, None, :].astype(
                    image_tokens.dtype
                )

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        return_metrics: bool = False,
    ) -> at.Float[at.Array, "*b ah"] | tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        progress_target = observation.progress
        progress_class_target = observation.progress_class
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            image_keys=self._image_keys,
            geometric_aug_cameras=self._geometric_aug_cameras,
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # Flow-matching loss per (batch, horizon-step).
        sq_err = jnp.square(v_t - u_t)  # [*b, ah, action_dim]
        flow_loss = jnp.mean(sq_err, axis=-1)  # [*b, ah]

        # Per-component metrics for logging (means over the batch). Always includes the
        # flow loss; progress entries are added when the head is active.
        metrics: dict[str, at.Array] = {"flow_loss": jnp.mean(flow_loss)}

        # Optional: flow loss over ONLY the first `flow_loss_real_dim` action dims (the
        # real robot action), excluding augmented dims (progress-as-action) + zero-pad.
        # `flow_loss` above averages over all `action_dim` dims, so it isn't comparable
        # across methods that pad/augment differently; `flow_loss_real` is. Logging-only.
        if self._flow_loss_real_dim is not None:
            metrics["flow_loss_real"] = jnp.mean(sq_err[..., : self._flow_loss_real_dim])

        # System1 progress head: aux state-value regression on frac**k.
        # progress_loss is per-sample [*b]; broadcast over the horizon axis and add.
        # The trainer does jnp.mean over [*b, ah]; broadcasting a per-sample value over
        # ah slots then meaning over ah recovers it exactly, so the realized weight is
        # exactly progress_loss_weight. Do NOT divide by action_horizon — that would
        # attenuate the weight by 1/ah. Progress is a per-sample VLM readout and has
        # nothing to do with the action horizon.
        total_loss = flow_loss
        if self._use_progress_head:
            logits = self._progress_logits(prefix_out, prefix_mask)  # [*b] or [*b, K]
            if self._progress_mode == "classes" and progress_class_target is not None:
                # K-way classification: softmax cross-entropy on the discrete bucket.
                labels = progress_class_target.astype(jnp.int32)
                logp = jax.nn.log_softmax(logits, axis=-1)
                onehot = jax.nn.one_hot(labels, self._progress_num_classes, dtype=logp.dtype)
                progress_loss = -jnp.sum(onehot * logp, axis=-1)  # [*b]
                total_loss = flow_loss + self._progress_loss_weight * progress_loss[..., None]
                pred_class = jnp.argmax(logits, axis=-1)
                metrics["progress_loss"] = jnp.mean(progress_loss)
                metrics["progress_loss_weighted"] = self._progress_loss_weight * jnp.mean(progress_loss)
                metrics["progress_acc"] = jnp.mean((pred_class == labels).astype(jnp.float32))
                # Bucket-distance MAE (how far off the argmax is, in class units).
                metrics["progress_class_mae"] = jnp.mean(jnp.abs(pred_class - labels).astype(jnp.float32))
                # Per-class recall: for each true class c, correct-count and sample-count.
                # Emitted as separate additive scalars (NOT a per-batch ratio) so they
                # average correctly over the wandb log window: train.py forms the ratio
                # progress_acc_class/{c} = mean(num_c) / mean(den_c) AFTER the window
                # reduction, so the 1/n_batches cancels and empty classes don't skew it.
                n_cls = self._progress_num_classes
                lab_flat = labels.reshape(-1)  # [N]
                pred_flat = pred_class.reshape(-1)  # [N]
                lab_oh = jax.nn.one_hot(lab_flat, n_cls, dtype=jnp.float32)  # [N, n_cls]
                correct = (pred_flat == lab_flat).astype(jnp.float32)  # [N]
                den_c = lab_oh.sum(axis=0)  # [n_cls] samples of each true class
                num_c = (lab_oh * correct[:, None]).sum(axis=0)  # [n_cls] correct per class
                for c in range(n_cls):
                    metrics[f"pcls_num/{c}"] = num_c[c]
                    metrics[f"pcls_den/{c}"] = den_c[c]
            elif self._progress_mode == "binary" and progress_class_target is not None:
                # "Is the subgoal finished?" — single logit, pos_weight-weighted BCE.
                # Positive = the top `progress_binary_pos_classes` deciles of the discrete
                # progress class (default classes {K-2,K-1} = frac>=0.8, incl. the settle-pad
                # complete frames whose class is clipped to K-1). This is the only boundary
                # System2 acts on, so we spend all head capacity here rather than on the
                # ambiguous middle deciles that saturate a K-way head at ~65%.
                logit = logits  # [*b] (num_outputs=1 -> squeezed by the head)
                pos_thresh = self._progress_num_classes - self._progress_binary_pos_classes
                labels = (progress_class_target.astype(jnp.int32) >= pos_thresh).astype(logit.dtype)  # [*b]
                # Weighted BCE-with-logits: weight the FINISHED (positive) term by pos_weight
                # to counter the ~4:1 imbalance. Stable form: max(z,0) - z*y + log1p(exp(-|z|)),
                # then scale by a per-sample weight (pos_weight for y=1, else 1).
                z = logit
                bce = jnp.maximum(z, 0) - z * labels + jnp.log1p(jnp.exp(-jnp.abs(z)))  # [*b]
                sample_w = jnp.where(labels > 0.5, self._progress_pos_weight, 1.0)  # [*b]
                progress_loss = sample_w * bce  # [*b]
                total_loss = flow_loss + self._progress_loss_weight * progress_loss[..., None]
                metrics["progress_loss"] = jnp.mean(progress_loss)
                metrics["progress_loss_weighted"] = self._progress_loss_weight * jnp.mean(progress_loss)
                # Confusion counts (additive, window-safe like pcls_num/den): accuracy is
                # useless at 4:1, so we log precision/recall/F1 from tp/fp/fn/tn AFTER the
                # window in train.py. pred = sigmoid(logit) > 0.5.
                pred_pos = (nnx.sigmoid(logit) > 0.5).astype(jnp.float32).reshape(-1)  # [N]
                y = labels.reshape(-1)  # [N]
                metrics["pbin_tp"] = jnp.sum(pred_pos * y)
                metrics["pbin_fp"] = jnp.sum(pred_pos * (1.0 - y))
                metrics["pbin_fn"] = jnp.sum((1.0 - pred_pos) * y)
                metrics["pbin_tn"] = jnp.sum((1.0 - pred_pos) * (1.0 - y))
            elif self._progress_mode == "continuous" and progress_target is not None:
                # Scalar state-value: Huber regression on frac**k.
                progress_pred = nnx.sigmoid(logits)  # [*b], in [0,1]
                target = jnp.clip(progress_target, 0.0, 1.0) ** self._progress_k
                progress_loss = _huber(progress_pred - target)  # [*b]
                total_loss = flow_loss + self._progress_loss_weight * progress_loss[..., None]
                metrics["progress_loss"] = jnp.mean(progress_loss)
                metrics["progress_loss_weighted"] = self._progress_loss_weight * jnp.mean(progress_loss)
                metrics["progress_mae"] = jnp.mean(jnp.abs(progress_pred - target))
                metrics["progress_pred_mean"] = jnp.mean(progress_pred)
                metrics["progress_target_mean"] = jnp.mean(target)
                # Per-bin MAE: bucket samples by the TRUE progress fraction into 10 bins
                # [0,0.1),...,[0.9,1.0], and emit summed abs-error + count per bin. As with
                # the classification per-class recall, these are additive scalars so
                # train.py forms progress_mae_bin/{b} = mean(err_b)/mean(cnt_b) AFTER the
                # window reduction (empty bins in a batch contribute 0/0 that cancels).
                # Bin on the RAW frac (progress_target), not frac**k, so bins are the
                # intuitive [0,0.1)..[0.9,1.0] on the actual progress.
                n_bins = 10
                frac_flat = jnp.clip(progress_target, 0.0, 1.0).reshape(-1)  # [N]
                abserr_flat = jnp.abs(progress_pred - target).reshape(-1)  # [N]
                bin_idx = jnp.clip((frac_flat * n_bins).astype(jnp.int32), 0, n_bins - 1)  # [N]
                bin_oh = jax.nn.one_hot(bin_idx, n_bins, dtype=jnp.float32)  # [N, n_bins]
                cnt_b = bin_oh.sum(axis=0)  # [n_bins]
                err_b = (bin_oh * abserr_flat[:, None]).sum(axis=0)  # [n_bins]
                for b in range(n_bins):
                    metrics[f"pbin_err/{b}"] = err_b[b]
                    metrics[f"pbin_cnt/{b}"] = cnt_b[b]

        if return_metrics:
            return total_loss, metrics
        return total_loss

    def _progress_logits(
        self, prefix_out: at.Float[at.Array, "b s emb"], prefix_mask: at.Bool[at.Array, "b s"]
    ) -> at.Array:
        """Raw progress-head output from the (optionally detached) prefix features.

        Returns [b] (continuous, pre-sigmoid logit) or [b, K] (classes, class logits).
        """
        feats = jax.lax.stop_gradient(prefix_out) if self._progress_stop_gradient else prefix_out
        if self._progress_readout == "shallow_transformer":
            return self.progress_head.from_sequence(feats, prefix_mask)
        if self._progress_readout in ("mean_pool", "prefix_token"):
            # Masked mean over valid prefix tokens.
            m = prefix_mask.astype(feats.dtype)[..., None]
            pooled = (feats * m).sum(axis=1) / jnp.clip(m.sum(axis=1), 1e-6, None)
            return self.progress_head.from_pooled(pooled)
        raise ValueError(f"unknown progress_readout: {self._progress_readout}")

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=self._image_keys)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_with_progress(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, at.Array]:
        """Sample the action chunk AND read the progress head in ONE prefix forward pass.

        sample_actions and predict_progress each run the (expensive) 3B PaliGemma prefix pass;
        called separately that pass is duplicated (~2x cost for head variants). Here the prefix
        runs ONCE — its output feeds the progress head, and its KV cache feeds the flow-sampling
        loop — so head variants cost ~the same as sample_actions alone. Requires use_progress_head
        (progact/no-head variants have no head; use sample_actions and read the action dim).
        Returns (actions [b, ah, ad], progress: [b] sigmoid for continuous | [b, K] softmax for
        classes).
        """
        if not self._use_progress_head:
            raise ValueError("sample_actions_with_progress called but use_progress_head is False")
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=self._image_keys)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # ONE prefix pass: keep BOTH prefix_out (-> progress head) and kv_cache (-> flow loop).
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        # progress from the same prefix features (no second big forward pass)
        logits = self._progress_logits(prefix_out, prefix_mask)
        progress = jax.nn.softmax(logits, axis=-1) if self._progress_mode == "classes" else nnx.sigmoid(logits)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_s = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_s, suffix_attn_mask], axis=-1)
            positions_s = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out_s, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions_s,
                kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
            )
            assert prefix_out_s is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0, progress

    def predict_progress(self, observation: _model.Observation) -> at.Array:
        """Inference-time subgoal-completion progress. Used by the eval server / System2.

        Runs a single prefix forward pass (noise-independent) and reads the progress head.
        Returns a scalar in [0,1] per sample (continuous mode) OR class probabilities
        [b, K] (classes mode; take argmax for the bucket). Requires use_progress_head.
        """
        if not self._use_progress_head:
            raise ValueError("predict_progress called but use_progress_head is False")
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=self._image_keys)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        logits = self._progress_logits(prefix_out, prefix_mask)
        if self._progress_mode == "classes":
            return jax.nn.softmax(logits, axis=-1)  # [b, K]
        # binary + continuous both read a single sigmoid. binary => P(subgoal finished);
        # continuous => the [0,1] state-value. System2 thresholds the binary output.
        return nnx.sigmoid(logits)  # [b]
