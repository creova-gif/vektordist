//! Declarative zero-dependency node & gateway configuration (`VektorNodeConfig`)
//! supporting TOML/key-value files and `VEKTORDIST_*` environment variable overrides.

use std::path::{Path, PathBuf};

use vektordist_core::hnsw::HnswParams;
use vektordist_core::{ClusterConfig, DistanceMetric, VektorError};

/// Declarative configuration for `vektordist node` and `vektordist gateway`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VektorNodeConfig {
    pub node_id: u32,
    pub listen_addr: String,
    pub coord_addr: String,
    pub gateway_listen_addr: String,
    pub data_dir: PathBuf,
    pub dimensions: usize,
    pub num_shards: u32,
    pub replication_factor: u8,
    pub write_quorum: u8,
    pub metric: DistanceMetric,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub master_secret_hex: Option<String>,
}

impl Default for VektorNodeConfig {
    fn default() -> Self {
        Self {
            node_id: 1,
            listen_addr: "127.0.0.1:50052".to_string(),
            coord_addr: "127.0.0.1:50051".to_string(),
            gateway_listen_addr: "127.0.0.1:8080".to_string(),
            data_dir: PathBuf::from("/tmp/vektordist-node"),
            dimensions: 768,
            num_shards: 16,
            replication_factor: 3,
            write_quorum: 2,
            metric: DistanceMetric::Cosine,
            hnsw_m: 16,
            hnsw_ef_construction: 128,
            hnsw_ef_search: 64,
            master_secret_hex: None,
        }
    }
}

impl VektorNodeConfig {
    /// Parse a TOML / `key = value` configuration string.
    /// Supports `#` comments, blank lines, `[section]` headers, and quoted or unquoted values.
    pub fn from_toml_str(contents: &str) -> Result<Self, VektorError> {
        let mut cfg = Self::default();
        for raw_line in contents.lines() {
            let line = strip_comment(raw_line).trim();
            if line.is_empty() || (line.starts_with('[') && line.ends_with(']')) {
                continue;
            }
            let Some((raw_key, raw_val)) = line.split_once('=') else {
                return Err(VektorError::InvalidConfig(
                    "malformed config line: expected `key = value`",
                ));
            };
            let key = raw_key.trim();
            let val = unquote(raw_val.trim());
            cfg.apply_kv(key, val)?;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load and validate a `VektorNodeConfig` from disk.
    pub fn from_file(path: &Path) -> Result<Self, VektorError> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            VektorError::WalIo(format!("failed to read config {}: {e}", path.display()))
        })?;
        Self::from_toml_str(&text)
    }

    /// Overlay `VEKTORDIST_*` environment variables when set and re-validate.
    pub fn apply_env_overrides(&mut self) -> Result<(), VektorError> {
        const ENV_KEYS: &[(&str, &str)] = &[
            ("VEKTORDIST_NODE_ID", "node_id"),
            ("VEKTORDIST_LISTEN_ADDR", "listen_addr"),
            ("VEKTORDIST_LISTEN", "listen_addr"),
            ("VEKTORDIST_COORD_ADDR", "coord_addr"),
            ("VEKTORDIST_COORD", "coord_addr"),
            ("VEKTORDIST_GATEWAY_LISTEN", "gateway_listen_addr"),
            ("VEKTORDIST_DATA_DIR", "data_dir"),
            ("VEKTORDIST_DIMENSIONS", "dimensions"),
            ("VEKTORDIST_NUM_SHARDS", "num_shards"),
            ("VEKTORDIST_REPLICATION_FACTOR", "replication_factor"),
            ("VEKTORDIST_WRITE_QUORUM", "write_quorum"),
            ("VEKTORDIST_METRIC", "metric"),
            ("VEKTORDIST_HNSW_M", "hnsw_m"),
            ("VEKTORDIST_HNSW_EF_CONSTRUCTION", "hnsw_ef_construction"),
            ("VEKTORDIST_HNSW_EF_SEARCH", "hnsw_ef_search"),
            ("VEKTORDIST_MASTER_SECRET", "master_secret_hex"),
        ];
        for &(env_var, key) in ENV_KEYS {
            if let Ok(val) = std::env::var(env_var) {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    self.apply_kv(key, trimmed)?;
                }
            }
        }
        self.validate()
    }

    /// Validate cluster, HNSW, and node configuration invariants.
    pub fn validate(&self) -> Result<(), VektorError> {
        if self.node_id == 0 {
            return Err(VektorError::InvalidConfig("node_id must be > 0"));
        }
        if self.listen_addr.trim().is_empty() || self.coord_addr.trim().is_empty() {
            return Err(VektorError::InvalidConfig(
                "listen_addr and coord_addr must not be empty",
            ));
        }
        if self.hnsw_m == 0 || self.hnsw_ef_construction == 0 || self.hnsw_ef_search == 0 {
            return Err(VektorError::InvalidConfig(
                "hnsw_m, hnsw_ef_construction, and hnsw_ef_search must be > 0",
            ));
        }
        self.cluster_config().validate()
    }

    /// Convert to `ClusterConfig`.
    #[must_use]
    pub fn cluster_config(&self) -> ClusterConfig {
        ClusterConfig {
            num_shards: self.num_shards,
            replication_factor: self.replication_factor,
            write_quorum: self.write_quorum,
            dimensions: self.dimensions,
            metric: self.metric,
        }
    }

    /// Convert to `HnswParams`.
    #[must_use]
    pub fn hnsw_params(&self) -> HnswParams {
        HnswParams {
            m: self.hnsw_m,
            m_max0: self.hnsw_m.saturating_mul(2),
            ef_construction: self.hnsw_ef_construction,
            ef_search: self.hnsw_ef_search,
            ..HnswParams::default()
        }
    }

    fn apply_kv(&mut self, key: &str, val: &str) -> Result<(), VektorError> {
        match key {
            "node_id" => {
                self.node_id = val
                    .parse::<u32>()
                    .map_err(|_| VektorError::InvalidConfig("invalid u32 for node_id"))?;
            }
            "listen_addr" => self.listen_addr = val.to_string(),
            "coord_addr" => self.coord_addr = val.to_string(),
            "gateway_listen_addr" => self.gateway_listen_addr = val.to_string(),
            "data_dir" => self.data_dir = PathBuf::from(val),
            "dimensions" => {
                self.dimensions = val
                    .parse::<usize>()
                    .map_err(|_| VektorError::InvalidConfig("invalid usize for dimensions"))?;
            }
            "num_shards" => {
                self.num_shards = val
                    .parse::<u32>()
                    .map_err(|_| VektorError::InvalidConfig("invalid u32 for num_shards"))?;
            }
            "replication_factor" => {
                self.replication_factor = val
                    .parse::<u8>()
                    .map_err(|_| VektorError::InvalidConfig("invalid u8 for replication_factor"))?;
            }
            "write_quorum" => {
                self.write_quorum = val
                    .parse::<u8>()
                    .map_err(|_| VektorError::InvalidConfig("invalid u8 for write_quorum"))?;
            }
            "metric" => {
                self.metric = match val.to_ascii_lowercase().as_str() {
                    "cosine" => DistanceMetric::Cosine,
                    "l2" => DistanceMetric::L2,
                    _ => {
                        return Err(VektorError::InvalidConfig(
                            "metric must be 'cosine' or 'l2'",
                        ));
                    }
                };
            }
            "hnsw_m" => {
                self.hnsw_m = val
                    .parse::<usize>()
                    .map_err(|_| VektorError::InvalidConfig("invalid usize for hnsw_m"))?;
            }
            "hnsw_ef_construction" => {
                self.hnsw_ef_construction = val.parse::<usize>().map_err(|_| {
                    VektorError::InvalidConfig("invalid usize for hnsw_ef_construction")
                })?;
            }
            "hnsw_ef_search" => {
                self.hnsw_ef_search = val
                    .parse::<usize>()
                    .map_err(|_| VektorError::InvalidConfig("invalid usize for hnsw_ef_search"))?;
            }
            "master_secret_hex" => {
                self.master_secret_hex = if val.is_empty() {
                    None
                } else {
                    Some(val.to_string())
                };
            }
            _ => {
                return Err(VektorError::InvalidConfig("unknown configuration key"));
            }
        }
        Ok(())
    }
}

/// Deterministically derive a 32-byte `SecurityGate` master secret from either a 64-char hex string
/// or an arbitrary passphrase string using four domain-separated CRC32/SplitMix64 lanes.
#[must_use]
pub fn derive_master_secret(raw: &str) -> [u8; 32] {
    let trimmed = raw.trim();
    if trimmed.len() == 64 {
        if let Some(decoded) = decode_hex_32(trimmed) {
            return decoded;
        }
    }
    let mut out = [0_u8; 32];
    for lane in 0..4_usize {
        let mut hasher_lo = crc32fast::Hasher::new_with_initial(0x5644_4B31 ^ (lane as u32));
        hasher_lo.update(trimmed.as_bytes());
        let lo = u64::from(hasher_lo.finalize());

        let mut hasher_hi =
            crc32fast::Hasher::new_with_initial(0xA5A5_5A5A ^ ((lane as u32) << 16));
        hasher_hi.update(&[lane as u8]);
        hasher_hi.update(trimmed.as_bytes());
        let hi = u64::from(hasher_hi.finalize());

        let mut x = ((hi << 32) | lo).wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        out[lane * 8..(lane + 1) * 8].copy_from_slice(&x.to_le_bytes());
    }
    out
}

fn decode_hex_32(hex: &str) -> Option<[u8; 32]> {
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let mut out = [0_u8; 32];
    for i in 0..32 {
        let hi = hex_nibble(bytes[i * 2])?;
        let lo = hex_nibble(bytes[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

const fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn strip_comment(line: &str) -> &str {
    let mut in_quotes = false;
    for (idx, ch) in line.char_indices() {
        if ch == '"' {
            in_quotes = !in_quotes;
        } else if ch == '#' && !in_quotes {
            return &line[..idx];
        }
    }
    line
}

fn unquote(val: &str) -> &str {
    if val.len() >= 2
        && ((val.starts_with('"') && val.ends_with('"'))
            || (val.starts_with('\'') && val.ends_with('\'')))
    {
        &val[1..val.len() - 1]
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_toml_config_and_rejects_invalid_quorum_or_unknown_key() {
        let toml = r#"
            # VektorDist node configuration
            [node]
            node_id = 3
            listen_addr = "127.0.0.1:50103"
            coord_addr = "127.0.0.1:50051"
            data_dir = "/var/lib/vektordist/node-3"

            [cluster]
            dimensions = 384
            num_shards = 8
            replication_factor = 3
            write_quorum = 2
            metric = "l2"

            [hnsw]
            hnsw_m = 24
            hnsw_ef_construction = 200
            hnsw_ef_search = 96
        "#;

        let cfg = VektorNodeConfig::from_toml_str(toml).unwrap();
        assert_eq!(cfg.node_id, 3);
        assert_eq!(cfg.dimensions, 384);
        assert_eq!(cfg.num_shards, 8);
        assert_eq!(cfg.metric, DistanceMetric::L2);
        assert_eq!(cfg.hnsw_m, 24);
        assert_eq!(cfg.hnsw_params().m_max0, 48);

        assert!(VektorNodeConfig::from_toml_str("unknown_field = 1").is_err());
        assert!(
            VektorNodeConfig::from_toml_str("replication_factor = 2\nwrite_quorum = 3").is_err()
        );
    }

    #[test]
    fn master_secret_derivation_is_deterministic_and_decodes_exact_hex() {
        let s1 = derive_master_secret("prod-passphrase-2026");
        let s2 = derive_master_secret("prod-passphrase-2026");
        let s3 = derive_master_secret("different-passphrase");
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);

        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let decoded = derive_master_secret(hex);
        assert_eq!(decoded[0], 0x01);
        assert_eq!(decoded[7], 0xef);
    }
}
