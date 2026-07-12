import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # --- System1 additions (RoboCasa subgoal-conditioned pi0.5 + progress head) ---
    # Image keys the model expects, in prefix order. If None, falls back to the
    # global model.IMAGE_KEYS (base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb).
    # For RoboCasa System1: ("scene_left","scene_right","wrist") + optional
    # ("anchor_scene_left","anchor_scene_right","anchor_wrist").
    image_keys: tuple[str, ...] | None = None
    # Cameras that receive geometric augmentation (crop+rotate). If None, falls back
    # to the legacy rule (geom aug iff "wrist" not in key). Anchor views are
    # deliberately excluded so the before/after geometry stays stable.
    geometric_aug_cameras: tuple[str, ...] | None = None
    # Add anchor (before) image groups + a learned anchor/current role embedding.
    use_anchor_images: bool = False

    # Progress head (subgoal-completion prediction).
    use_progress_head: bool = False
    # Prediction target/loss:
    # - "continuous": a scalar state-value in [0,1], Huber-regressed on frac**progress_k
    #   (reads Observation.progress).
    # - "classes": a `progress_num_classes`-way classifier, cross-entropy on the discrete
    #   progress class 0..K-1 (reads Observation.progress_class). System2 reads the
    #   argmax / softmax bucket.
    progress_mode: str = "continuous"
    # Number of buckets when progress_mode == "classes" (system1_full uses 10).
    progress_num_classes: int = 10
    # Readout over the prefix: "shallow_transformer" | "mean_pool" | "prefix_token".
    progress_readout: str = "shallow_transformer"
    # If True, the head reads stop_grad(prefix) so the progress loss never alters the
    # VLM. Default FALSE: the prefix's FINAL-layer output (what the head reads) gets
    # NO gradient from the action loss (the action expert only consumes the prefix via
    # K/V inside attention, not its last-layer output), so insulating it would leave
    # the readout representation stale at pretrained init. Letting progress gradient
    # flow (small weight) trains that representation + co-shapes the backbone.
    progress_stop_gradient: bool = False
    # Aux loss weight (target = frac**progress_k). 0.5: progress is a first-class
    # product (System2 reads it), so it co-trains the VLM substantially — watch
    # flow_loss vs progress_loss in wandb to confirm actions don't regress.
    progress_loss_weight: float = 0.5
    # Progress target = frac**progress_k. k=1 (LINEAR): progress is the literal
    # time-fraction through the subgoal span — uniform gradient across the whole span,
    # no flat/dead zone early, trivially interpretable for System2's hand-off threshold.
    # (k>1 back-loads resolution toward completion — better for pure transit subgoals but
    # understates uniform contact subgoals like grasp; revisit per-primitive if eval
    # curves show transit subgoals need it.)
    progress_k: float = 1.0
    # Shallow-transformer head width / depth / heads. The head down-projects the
    # 2048-d PaliGemma prefix to progress_hidden, then runs the attention readout
    # there (head_dim = progress_hidden / progress_num_heads = 64 by default). 512 is
    # a cheap middle ground (~8M params) between the 256 default and full 2048.
    progress_hidden: int = 512
    progress_num_layers: int = 2
    progress_num_heads: int = 8

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        # Image keys default to the global IMAGE_KEYS; System1 overrides them (and
        # optionally adds anchor_* groups for the progress head).
        keys = list(self.image_keys) if self.image_keys is not None else list(_model.IMAGE_KEYS)
        if self.use_anchor_images:
            keys = keys + [f"anchor_{k}" for k in keys]

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images=dict.fromkeys(keys, image_spec),
                image_masks=dict.fromkeys(keys, image_mask_spec),
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                progress=(
                    jax.ShapeDtypeStruct([batch_size], jnp.float32)
                    if (self.use_progress_head and self.progress_mode == "continuous")
                    else None
                ),
                progress_class=(
                    jax.ShapeDtypeStruct([batch_size], jnp.int32)
                    if (self.use_progress_head and self.progress_mode == "classes")
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
