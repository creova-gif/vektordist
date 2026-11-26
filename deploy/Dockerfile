# syntax=docker/dockerfile:1.7

FROM rust:1.85-bookworm AS builder
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock LICENSE ./
COPY src ./src
COPY crates ./crates

RUN cargo build --release --locked --bin vektordist

# Distroless runtime image running as nonroot:nonroot (UID/GID 65532:65532)
FROM gcr.io/distroless/cc-debian12:nonroot AS runtime
LABEL org.opencontainers.image.title="vektordist" \
      org.opencontainers.image.description="Distributed sharded HNSW vector search engine with quorum WAL durability" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /var/lib/vektordist

COPY --from=builder /workspace/target/release/vektordist /usr/local/bin/vektordist
COPY --from=builder /workspace/LICENSE /usr/share/licenses/vektordist/LICENSE

# 50051: Coordinator / Raft gRPC | 50052: Shard Replica gRPC | 8080: HTTP/1.1 REST & Prometheus /metrics
EXPOSE 50051 50052 8080
USER nonroot:nonroot
ENTRYPOINT ["/usr/local/bin/vektordist"]
CMD ["--help"]
