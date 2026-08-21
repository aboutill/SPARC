# SPARC — Architecture Notice

This document describes the four layers the SPARC project is organized into, and traces how a single command propagates through them. Understanding this separation of concerns is the fastest route to understanding the codebase as a whole: most confusion about where a given piece of functionality resides is resolved once the responsibilities of each layer are clear.

```
┌────────────────────────────────────────────────────────────────┐
│  make        ONE-TIME SETUP                        Makefile    │
│              Get `SPARC` onto PATH, pull the Docker image.     │
├────────────────────────────────────────────────────────────────┤
│  bash        ORCHESTRATION                         scripts/    │ 
│              Parse CLI flags, decide what to mount,            │
│              launch Docker. Never does computation.            │
├────────────────────────────────────────────────────────────────┤
│  Docker      ENVIRONMENT                           Dockerfile  │
│              Pinned OS, Python, MRtrix, ITK-SNAP, SVR-lite,    │
│              and the default pretrained model weights.         │
│              One reproducible, versioned image.                │
├────────────────────────────────────────────────────────────────┤
│  Python      THE PIPELINE ITSELF                   src/sparc/  │
│              Preprocessing, segmentation, reconstruction,      │
│              reorientation, postprocessing. The methodology    │
│              is implemented here. Docker exists to access this │
│              part correctly and reproducibly.                  │
└────────────────────────────────────────────────────────────────┘
```

## 1. `make` — One-Time Setup

This layer establishes the preconditions required to run the pipeline.

- `make install`: symlinks the `SPARC` dispatcher onto `PATH` and pulls the Docker image.
- `make pull-image` / `make uninstall`.

This is run once per machine. Every subsequent section assumes this step has already been completed.

## 2. `bash` — Orchestration

**`scripts/SPARC.bash`** serves as the dispatcher: `SPARC <subcommand> [args]` execs into the corresponding script under **`scripts/subcommands/`** (`pipeline.bash`, `train.bash`, `test.bash`, `download-models.bash`, `notebooks.bash`).

Each subcommand script is responsible for translating user-facing flags into a correctly configured `docker run` invocation: determining which host directories to bind-mount (`--input`/`--output`/`--cfg`/`--models_*`), whether GPU or X11 flags are required, and ensuring the container runs under the invoking user's UID rather than root. **No computation is performed at this layer**; any functionality related to numerical processing or image analysis resides elsewhere.

The wrapper script `examples/runSPARC.bash` operates alongside this layer as convenience presets over the same dispatcher.

## 3. `Docker` — The Reproducible Environment

A single `Dockerfile` builds `aboutill/sparc:v1.0.0`, which contains:

- A pinned OS (Ubuntu 20.04), MRtrix3, ITK-SNAP, and a conda-managed Python 3.11 environment.
- SVR-lite, a compiled binary developed as an independent project: https://github.com/baby-MedIA/svr-lite.
- The SPARC Python package itself, installed via `pip install -e .`.
- The default pretrained model weights for all three components, across all four training-domain variants, baked in at build time so that the pipeline is executable with no additional setup.

The container is stateless and ephemeral (`--rm`); no state persists between runs. All run-specific data — input scans, output results, custom configuration files, or model overrides — crosses the host/container boundary through the bind mounts established by the bash layer.

## 4. `Python` — The Pipeline Implementation

Located under `src/sparc/`:

- **`bin/`**: CLI entry points (`pipeline.py`, `train.py`, `test.py`), invoked inside the container by the `docker run` command constructed at the bash layer.
- **`sparc/pipeline/`**: the six pipeline stages (`preprocessing`, `chest_segmentation`, `svr`, `heart_segmentation`, `reorientation`, `postprocessing`), each implemented as an independent subpackage comprising a class definition, the stage logic itself, and, for interactive stages, a GUI review module.
- **`sparc/segmentation/`, `sparc/reorientation/`**: the underlying deep-learning model, training, and testing code, shared between standalone `train`/`test` CLI usage and the deployed pipeline stages.
- **`sparc/utils/`, `sparc/tools/`**: shared utilities (NIfTI I/O, MRtrix/SVR-lite wrappers, logging configuration).
- **`cfg/`**: default YAML configuration: `pipeline_default.yaml` governs pipeline-level behaviour.

This layer contains the scientific and algorithmic core of the project, and is where the majority of ongoing development takes place.

## End-to-End Execution: `SPARC pipeline`

```
SPARC pipeline --input ... --output ...
  |
  +- [make]    already done: SPARC is on PATH, image is pulled
  |
  +- [bash]    SPARC.bash -> scripts/subcommands/pipeline.bash
  |            builds docker run flags (mounts, GPU, UID)
  |
  +- [docker]  docker run ... aboutill/sparc:<tag> pipeline <args>
  |            container starts; Python env, model weights, and
  |            SVR-lite are already present from the image build
  |
  +- [python]  bin/pipeline.py parses args, builds a
               sparc.pipeline.Pipeline, calls .run() -- which runs
               each of the six stages in turn

Results are returned to the host via the bind-mounted --output directory.
```

## Rationale for the Layered Design

Each layer addresses a concern that the others cannot:

- **make** addresses first-time installation overhead.
- **bash** addresses the coordination of host paths, GPU access, and container mounts, without requiring the Python codebase to have any awareness of Docker.
- **Docker** addresses environment consistency across machines and institutions.
- **Python** is the only layer that should change in response to a scientific or algorithmic decision. A fix belonging to this layer will rarely require a corresponding change to the bash or Dockerfile layers.

## Related Documentation

- **README.md**: installation and a public-facing project overview.
- **instructions.txt**: an operational guide covering modes, GUI review, and troubleshooting for clinical routine case processing.
- This document: architectural reference for contributors working on the codebase, as distinct from operating it.
