ARG BASE_IMAGE=pia-test-base:latest
FROM ${BASE_IMAGE}

WORKDIR /workspace

# -- 1. Install common dependencies (shared across all modules) ----------------
COPY packages/pia_prod/AI/modules/requirements.txt /tmp/requirements-common.txt
RUN pip install --no-cache-dir -r /tmp/requirements-common.txt \
    && rm /tmp/requirements-common.txt

# -- 2. Install yonsei_walljump-specific dependencies -----------------------------
COPY packages/pia_prod/AI/modules/yonsei_walljump/requirements.txt \
     /tmp/requirements-module.txt
RUN pip install --no-cache-dir -r /tmp/requirements-module.txt \
    && rm /tmp/requirements-module.txt

# -- 3. Copy project source ----------------------------------------------------
COPY . /workspace/

# -- 4. Make pia + pia_prod importable -----------------------------------------
RUN pip install --no-cache-dir --no-deps -e /workspace/

# -- 5. Path configuration -----------------------------------------------------
ENV PYTHONPATH=/workspace/packages
ENV TEAM=ai

# -- 6. Default command: run yonsei_walljump tests ---------------------------------
CMD ["python3", "-m", "pytest", \
     "packages/pia_prod/AI/tests/modules/test_yonsei_walljump.py", \
     "-v", "--tb=short"]
