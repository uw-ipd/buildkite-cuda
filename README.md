# Buildkite & CUDA Example

A minimal example demonstrating CUDA in docker-based buildkite pipelines using
a two step [pipeline](.buildkite/pipeline.yml):

1)  A `nvidia-cuda` & `conda` based ["test
runner"](.buildkite/runner/Dockerfile) is built, including both os-level
and conda-level requirements. This runs within the buildkit-agent
container.

2) The
[`docker-buildkite-plugin`](https://github.com/uw-ipd/docker-buildkite-plugin)
is used to execute tests via the `nvidia` docker runtime within the runner
container.

## Setup

The example requires an agent host with
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed. The
buildkite agent may be run via the standard
[`buildkite/agent`](https://hub.docker.com/r/buildkite/agent/) image,
bind-mounting `/var/run/docker.sock` to enable support for
`docker-buildkite-plugin`.
