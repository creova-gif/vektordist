# ADR-012: M12 Atomic Shard Checkpointing (`b"VDCK"` + `b"VDSG"`) and WAL Log Compaction
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
Prior to Milestone 12, each `vektordist` shard appended every `Upsert` and `Delete` frame to an append-only `b"VDWL"` Write-Ahead Log (`WalLog`) indefinitely. Under sustained overwrite and delete churn:
1. **Unbounded Disk Growth:** Historical overwritten vectors and delete tombstones remained in `shard-{id:04}.wal` forever, even when active live cardinality stayed small.
2. **Slow Cold Restarts (`O(|WAL|)`):** Node recovery had to decode, validate, and replay every historical mutation from `seq_no = 1`, allocating and immediately tombstoning superseded slots in `VectorArena` and `HnswIndex`.

We need a crash-safe, single-directory checkpointing and WAL compaction protocol with end-to-end IEEE CRC32 integrity verification and zero silent corruption or mid-compaction data loss.

## Decision
1. **Zero-Tombstone Segment Snapshot (`b"VDSG"`):**
   - `ShardCheckpointManager::create_checkpoint` invokes `TieredCompactor::compact(&[hnsw.arena()])` to strip 100% of tombstoned and overwritten slots (`slot_count == live_len`).
   - The compacted `VectorArena` is written to `shard-{shard_id:04}.vdsg.tmp` via `PersistentVectorSegment::write_from_arena` (which `fsync`s the 8-byte-aligned payload + trailing IEEE CRC32) and atomically `rename`d to `shard-{shard_id:04}.vdsg`.
2. **40-Byte Binary Checkpoint Manifest (`b"VDCK"`):**
   - After `.vdsg` is durable, `ShardCheckpointManager` writes a fixed 40-byte little-endian manifest to `shard-{shard_id:04}.ckpt.tmp` (`magic: b"VDCK"`, `version: 1`, `shard_id: u32`, `dims: u32`, `checkpoint_seq_no: u64`, `live_vectors: u64`, `segment_crc32: u32`, and trailing `ckpt_crc32: u32` over bytes `0..36`), calls `sync_all()`, atomically `rename`s it to `shard-{shard_id:04}.ckpt`, and best-effort syncs the parent directory.
3. **Crash-Safe WAL Compaction (`WalLog::compact_to`):**
   - Only after `.ckpt` is durably renamed does `WalLog::compact_to(retained_tail)` write any caller-retained tail records as `b"VDWL"` CRC32 frames to `<wal_path>.compact.tmp`, call `sync_all()`, atomically `rename` it over `<wal_path>`, and reopen the file handle positioned at `SeekFrom::End(0)`.
4. **Verified Cold Recovery (`ShardCheckpointManager::recover_shard`):**
   - On startup, if `shard-{shard_id:04}.ckpt` exists, `recover_shard` validates its 40-byte `b"VDCK"` header and `ckpt_crc32`, opens `shard-{shard_id:04}.vdsg` (verifying internal segment CRC32 AND cross-checking `seg.crc32() == manifest.segment_crc32`), populates `HnswIndex` without re-normalizing already-normalized `Cosine` vectors, sets `last_seq_no = checkpoint_seq_no`, and replays only post-checkpoint WAL frames (`seq_no > checkpoint_seq_no`). Any `.ckpt` or `.vdsg` bit-rot fails closed with `VektorError::WalCorruption`.

## Alternatives Considered
- **In-Place WAL Truncation (`memmove` + `set_len`):** Rejected because a `SIGKILL` or power loss mid-copy leaves the WAL partially overwritten and unrecoverable. Write-to-temp + `fsync` + POSIX atomic `rename` guarantees the previous WAL remains intact until replacement is durable.
- **Serializing Full HNSW Adjacency Layers to Disk:** Deferred; rebuilding HNSW connectivity over only live compacted vectors avoids tombstoned routing hops (improving graph quality and memory footprint by stripping dead slots) while avoiding complex graph-pointer corruption surfaces.

## Consequences & Failure Modes
- **Crash Window Before `.ckpt` Rename:** `.vdsg.tmp` / `.ckpt.tmp` are ignored on restart; the previous `.ckpt` (or full WAL) recovers zero-loss state.
- **Crash Window After `.ckpt` Rename, Before `compact_to` Rename:** The new `.ckpt` (`checkpoint_seq_no`) is loaded and `recover_shard` idempotently skips all uncompacted WAL records with `seq_no <= checkpoint_seq_no`.
- **Transient Disk Headroom:** Peak disk usage briefly includes `.vdsg.tmp` + uncompacted `.wal` until `compact_to` completes and unlinks the old WAL inode.
