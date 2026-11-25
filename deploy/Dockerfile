# syntax=docker/dockerfile:1.7

FROM rust:1.85-bookworm AS builder
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY crates ./crates

RUN cargo build --release --locked --bin vektordist

FROM gcr.io/distroless/cc-debian12:nonroot AS runtime
WORKDIR /var/lib/vektordist

COPY --from=builder /workspace/target/release/vektordist /usr/local/bin/vektordist

EXPOSE 50051 50052
USER nonroot:nonroot
ENTRYPOINT ["/usr/local/bin/vektordist"]
CMD ["--help"]
