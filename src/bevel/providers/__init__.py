"""Model providers.

Everything that talks to an inference server lives here, behind the
``ModelProvider`` protocol so the agent loop never learns about HTTP.
"""
