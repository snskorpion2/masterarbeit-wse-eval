FROM python:3.12-slim@sha256:e5c9fa26ffb76e11e0f054f30dc2523a2f9693f0c36c0cf1e39b27e152d899fc
ARG SOURCE_COMMIT
LABEL org.opencontainers.image.source="https://github.com/snskorpion2/Masterarbeit" org.opencontainers.image.revision="${SOURCE_COMMIT}" org.opencontainers.image.description="Gold-blind paired triple-quality evaluation"
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/results/home TMPDIR=/results/tmp XDG_CACHE_HOME=/results/cache
WORKDIR /app
COPY wse_eval.py /app/wse_eval.py
COPY wse_published_gold_triple_eval.py /app/wse_published_gold_triple_eval.py
COPY NOTICE.md /app/NOTICE.md
COPY MAVEN_ARG_NOTICE.md /app/MAVEN_ARG_NOTICE.md
COPY bundle /app/bundle
RUN chown root:root /tmp /var/tmp && chmod a-w /tmp /var/tmp && chmod -R a-w /app && mkdir -p /results/home /results/tmp /results/cache && chown -R 65532:65532 /results && chmod 700 /results /results/home /results/tmp /results/cache
VOLUME /results
USER 65532:65532
