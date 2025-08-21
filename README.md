# local-gpt-oss

This repo introduces a reproducible way to optimize OpenAI's latest open source
model `gpt-oss-20b`  to be able to run locally on a mac mini with 24gb (or even
16gb of ram) quickly and efficiently.

## Goals
* run OpenAI's `gpt-oss-20b` on 24gb mac mini
* benchmark model for peak memory usage and latency
* optimize model to acheive lower peak memory (e.g. pruning weights,
quantization, etc.)

**Requirements**
* optimized model must retain acceptable "performance"
* optimized model's tokens-per-sec metric be at or lower than that of the
original model (i.e. no expensive quantization or large overhead setup at
runtime)

### Stretch goal
Used the pruned model to create a light-weight, tool-equipped agent `Jarvis` --
a helpful assistant on the mac mini that can interact with built-in applications
like chrome (bringing up youtube videos), Spotify (playing a song), or even
handling hands-free usage of the computer for accessibility purposes.
