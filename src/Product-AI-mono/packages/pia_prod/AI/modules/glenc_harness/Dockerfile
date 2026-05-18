# =============================================================================
# glenc_harness -- Module-Specific Test Dockerfile
# =============================================================================
# Inherits from the shared base image and adds only what glenc_harness needs.
# Dependencies are intentionally NOT installed via setup.py so that missing
# packages fail loudly (the whole point of module-level isolation).
#
# Build context: Product-AI-mono root
#
# Build (recommended via docker-compose, or manually):
#   cd <Product-AI-mono root>
#   docker build -t pia-test-glenc-harness:latest \
#       -f packages/pia_prod/AI/modules/glenc_harness/Dockerfile .
# =============================================================================

ARG BASE_IMAGE=pia-test-base:latest
FROM ${BASE_IMAGE}

WORKDIR /workspace

# -- 1. Install common dependencies (shared across all modules) ----------------
COPY packages/pia_prod/AI/modules/requirements.txt /tmp/requirements-common.txt
RUN pip install --no-cache-dir -r /tmp/requirements-common.txt \
    && rm /tmp/requirements-common.txt

# -- 2. Install glenc_harness-specific dependencies ----------------------------
COPY packages/pia_prod/AI/modules/glenc_harness/requirements.txt \
     /tmp/requirements-module.txt
RUN pip install --no-cache-dir -r /tmp/requirements-module.txt \
    && rm /tmp/requirements-module.txt

# -- 3. Copy project source ----------------------------------------------------
# The full source tree is copied because:
#   - packages/pia/ provides shared utilities (preprocessing, postprocessing)
#   - DTO/stream_params.py imports param/config from ALL modules at parse time
#   - Base classes (service_base.py, event_base.py) are shared infrastructure
#   - conftest.py, thread_stop_signal.py are test utilities
# The .dockerignore at the Product-AI-mono root excludes assets/ (55GB+).
COPY . /workspace/

# -- 4. Make pia + pia_prod importable -----------------------------------------
# Use --no-deps to avoid pulling setup.py's unified dependency list.
# This ensures that ONLY the dependencies explicitly declared in
# requirements.txt are available at runtime.
RUN pip install --no-cache-dir --no-deps -e /workspace/

# -- 5. Path configuration -----------------------------------------------------
ENV PYTHONPATH=/workspace/packages
ENV TEAM=ai

# -- 6. Default command: run glenc_harness tests -------------------------------
CMD ["python3", "-m", "pytest", \
     "packages/pia_prod/AI/tests/modules/test_glenc_harness.py", \
     "-v", "--tb=short"]
