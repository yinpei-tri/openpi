"""Multi-node JAX distributed initialization.

JAX multi-host works differently from PyTorch: ONE process per node (each process owns
all local GPUs), and the processes form a single global device mesh via
``jax.distributed.initialize(coordinator_address, num_processes, process_id)``. This
must run before any JAX device op. Single-node runs skip it entirely (no-op), so the
existing single-node path is unchanged.

Coordinator/rank info is resolved, in priority order, from:
  1. Explicit env vars (JAX_COORDINATOR_ADDRESS / JAX_NUM_PROCESSES / JAX_PROCESS_ID) —
     lets you force any topology.
  2. SageMaker's /opt/ml/input/config/resourceconfig.json (hosts list + current_host) +
     SM_MASTER_PORT (default 12355). This is what the 2-node SageMaker job uses.
  3. Otherwise: single-node, do nothing.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib

logger = logging.getLogger("openpi")

_SM_RESOURCE_CONFIG = "/opt/ml/input/config/resourceconfig.json"
_DEFAULT_COORDINATOR_PORT = "12355"


def _resolve_topology() -> tuple[str, int, int] | None:
    """Return (coordinator_address, num_processes, process_id), or None for single-node."""
    # 1) Explicit override.
    if os.environ.get("JAX_COORDINATOR_ADDRESS"):
        return (
            os.environ["JAX_COORDINATOR_ADDRESS"],
            int(os.environ["JAX_NUM_PROCESSES"]),
            int(os.environ["JAX_PROCESS_ID"]),
        )

    # 2) SageMaker resourceconfig.json.
    rc_path = pathlib.Path(os.environ.get("SM_RESOURCE_CONFIG", _SM_RESOURCE_CONFIG))
    if rc_path.exists():
        rc = json.loads(rc_path.read_text())
        hosts = sorted(rc["hosts"])  # e.g. ["algo-1", "algo-2"] — sort for a stable order
        current = rc["current_host"]
        if len(hosts) > 1:
            port = os.environ.get("SM_MASTER_PORT", _DEFAULT_COORDINATOR_PORT)
            return (f"{hosts[0]}:{port}", len(hosts), hosts.index(current))

    # 3) Single node.
    return None


def maybe_init_distributed() -> None:
    """Initialize jax.distributed for multi-node; no-op for single-node.

    Safe to call exactly once at process start (before any JAX device op). Logs the
    resolved topology so the SageMaker logs show each node's rank + device counts.
    """
    topo = _resolve_topology()
    if topo is None:
        logger.info("Distributed: single-node (no jax.distributed.initialize).")
        return
    coordinator, num_processes, process_id = topo
    logger.info(
        f"Distributed: initializing jax.distributed | coordinator={coordinator} "
        f"num_processes={num_processes} process_id={process_id}"
    )
    import jax

    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=num_processes,
        process_id=process_id,
    )
    logger.info(
        f"Distributed: initialized. jax.process_index={jax.process_index()} "
        f"jax.process_count={jax.process_count()} global_devices={jax.device_count()}"
    )


def is_primary() -> bool:
    """True on the coordinator process (process 0), or single-node. Use to guard actions
    that only one process should do (e.g. S3 checkpoint upload)."""
    import jax

    return jax.process_index() == 0
