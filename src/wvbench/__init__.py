"""Weaviate benchmark environment for the Common Baseline vector database project."""

import warnings

# weaviate-client pulls in authlib, which warns about httpx on import; not relevant here.
# authlib installs its own "always" filter when imported, so import it first and add ours on top.
try:
    import authlib.deprecate  # noqa: F401
except ImportError:
    pass
warnings.filterwarnings("ignore", message=r".*httpx module is deprecated.*")
