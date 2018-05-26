# Buildkite & CUDA Example

A minimal example demonstrating CUDA in docker-based buildkite pipelines
using a two step pipeline:

1)  A `nvidia-cuda` & `conda` based ["test
runner"](.buildkite/runner/Dockerfile) is built, including both os-level
and conda-level requirements. This runs within the buildkit-agent
container.

2) The
[docker-buildkite-plugin](https://github.com/uw-ipd/docker-buildkite-plugin)
is used to execute tests the `nvidia` docker runtime within the runner
container.
