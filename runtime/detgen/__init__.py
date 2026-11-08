"""Deterministic text generation on top of detrt: the head stage (final norm, lm_head, sampling)
and a reference generator. detrt stays unchanged: its sources are hashed into the registered
runtime_hash of the embedding model; a generation model registers detrt + detgen together."""
