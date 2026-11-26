#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

//! Milestone 2: Deterministic first-principles Raft topology state machine, leader-lease fencing,
//! and dynamic online shard rebalancing with zero-loss WAL streaming prior to ownership transfer.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};

use tonic::{Request, Response, Status};
use vektordist_core::{ClusterConfig, DistanceMetric, Epoch, NodeId, ShardId, VektorError};
use vektordist_proto::v1::topology_rpc_server::TopologyRpc;
use vektordist_proto::v1::{
    GetTopologyRequest, GetTopologyResponse, WireNodeStatus, WireShardAssignment,
};

use crate::rebalance::push_shard_config_to_nodes;
pub use crate::rebalance::{
    compute_rebalance_plan, execute_online_rebalance, RebalancePlan, ShardMove,
};
use crate::{shard_state_to_u32, ShardAssignment, ShardState, TopologySnapshot, TopologyStore};

/// Monotonic Raft election term.
pub type Term = u64;

/// 1-based monotonic Raft log index (`0` denotes empty log).
pub type LogIndex = u64;

/// Role of a Raft control-plane participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftRole {
    Follower,
    Candidate,
    Leader,
}

/// Replicated state-machine commands for cluster topology transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyCommand {
    /// Promote a new primary for `shard_id` and bump its epoch.
    PromotePrimary {
        shard_id: ShardId,
        new_primary: NodeId,
    },
    /// Replace a single shard's placement and epoch.
    UpdateShard { assignment: ShardAssignment },
    /// Atomically apply a full online rebalance assignment across active nodes.
    ApplyRebalance {
        assignments: Vec<ShardAssignment>,
        active_nodes: Vec<NodeId>,
    },
    /// Register or update a data-plane node address in the topology.
    RegisterNode { node_id: NodeId, addr: String },
    /// Mark a data-plane node offline in the topology.
    RemoveNode { node_id: NodeId },
    /// Replicate or rotate a tenant-scoped API key record across the cluster control plane.
    UpsertApiKey {
        record: crate::security::ScopedApiKey,
    },
    /// Immediately revoke an API key cluster-wide across all synchronized gateways.
    RevokeApiKey { key_id: String },
    /// Configure cluster-wide rate limit and storage quota policy for `org_id`
    /// (`max_qps = max_qps_milli as f64 / 1000.0`, `burst_capacity = burst_milli as f64 / 1000.0`).
    SetTenantQuota {
        org_id: crate::security::OrgId,
        max_qps_milli: u64,
        burst_milli: u64,
        max_vectors: u64,
        max_memory_bytes: u64,
    },
}

/// Single entry in the replicated Raft topology log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEntry {
    pub term: Term,
    pub index: LogIndex,
    pub cmd: TopologyCommand,
}

/// Raft `RequestVote` RPC arguments (§5.2, §5.4.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestVoteArgs {
    pub term: Term,
    pub candidate_id: NodeId,
    pub last_log_index: LogIndex,
    pub last_log_term: Term,
}

/// Raft `RequestVote` RPC reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestVoteReply {
    pub term: Term,
    pub vote_granted: bool,
}

/// Raft `AppendEntries` RPC arguments (§5.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendEntriesArgs {
    pub term: Term,
    pub leader_id: NodeId,
    pub prev_log_index: LogIndex,
    pub prev_log_term: Term,
    pub entries: Vec<LogEntry>,
    pub leader_commit: LogIndex,
}

/// Raft `AppendEntries` RPC reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendEntriesReply {
    pub term: Term,
    pub success: bool,
    pub match_index: LogIndex,
}

/// Deterministic first-principles Raft state machine for a single control-plane node.
#[derive(Debug, Clone)]
pub struct RaftNode {
    pub node_id: NodeId,
    pub peers: Vec<NodeId>,
    pub current_term: Term,
    pub voted_for: Option<NodeId>,
    pub role: RaftRole,
    pub leader_id: Option<NodeId>,
    pub log: Vec<LogEntry>,
    pub commit_index: LogIndex,
    pub last_applied: LogIndex,
    pub next_index: BTreeMap<NodeId, LogIndex>,
    pub match_index: BTreeMap<NodeId, LogIndex>,
    pub lease_acks: BTreeSet<NodeId>,
    pub snapshot: TopologySnapshot,
    pub node_addrs: BTreeMap<NodeId, String>,
    pub alive_nodes: BTreeSet<NodeId>,
    pub api_keys: BTreeMap<String, crate::security::ScopedApiKey>,
    pub revoked_keys: BTreeSet<String>,
    pub tenant_quotas: BTreeMap<crate::security::OrgId, crate::security::TenantQuotaPolicy>,
}

impl RaftNode {
    /// Construct a new `RaftNode` in the `Follower` state at term 0.
    #[must_use]
    pub fn new(node_id: NodeId, peers: Vec<NodeId>, initial_snapshot: TopologySnapshot) -> Self {
        let mut alive_nodes = BTreeSet::new();
        for s in &initial_snapshot.shards {
            alive_nodes.insert(s.primary);
            for &f in &s.followers {
                alive_nodes.insert(f);
            }
        }
        Self {
            node_id,
            peers,
            current_term: 0,
            voted_for: None,
            role: RaftRole::Follower,
            leader_id: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            next_index: BTreeMap::new(),
            match_index: BTreeMap::new(),
            lease_acks: BTreeSet::new(),
            snapshot: initial_snapshot,
            node_addrs: BTreeMap::new(),
            alive_nodes,
            api_keys: BTreeMap::new(),
            revoked_keys: BTreeSet::new(),
            tenant_quotas: BTreeMap::new(),
        }
    }

    /// Strict majority quorum count (`peers.len() / 2 + 1`).
    #[must_use]
    pub fn majority_quorum(&self) -> usize {
        (self.peers.len() / 2) + 1
    }

    /// Index of the highest log entry (`0` if the log is empty).
    #[must_use]
    pub fn last_log_index(&self) -> LogIndex {
        self.log.len() as LogIndex
    }

    /// Term of the highest log entry (`0` if the log is empty).
    #[must_use]
    pub fn last_log_term(&self) -> Term {
        self.log.last().map_or(0, |e| e.term)
    }

    /// Term at `index` (`Some(0)` at `0`, `Some(entry.term)` if in bounds, `None` if beyond log end).
    #[must_use]
    pub fn term_at(&self, index: LogIndex) -> Option<Term> {
        if index == 0 {
            return Some(0);
        }
        self.log.get((index - 1) as usize).map(|e| e.term)
    }

    /// Transition to `Candidate`, increment `current_term`, vote for self, and build `RequestVoteArgs`.
    pub fn start_election(&mut self) -> RequestVoteArgs {
        self.current_term += 1;
        self.role = RaftRole::Candidate;
        self.voted_for = Some(self.node_id);
        self.leader_id = None;
        self.lease_acks.clear();
        self.lease_acks.insert(self.node_id);

        if self.peers.len() <= 1 {
            self.become_leader();
        }

        RequestVoteArgs {
            term: self.current_term,
            candidate_id: self.node_id,
            last_log_index: self.last_log_index(),
            last_log_term: self.last_log_term(),
        }
    }

    /// Transition from `Candidate` to `Leader` after winning a majority of votes in `current_term`.
    pub fn become_leader(&mut self) {
        self.role = RaftRole::Leader;
        self.leader_id = Some(self.node_id);
        let next = self.last_log_index() + 1;
        self.next_index.clear();
        self.match_index.clear();
        for &peer in &self.peers {
            self.next_index.insert(peer, next);
            if peer == self.node_id {
                self.match_index.insert(peer, self.last_log_index());
            } else {
                self.match_index.insert(peer, 0);
            }
        }
        self.lease_acks.clear();
        self.lease_acks.insert(self.node_id);
    }

    /// Evaluate a `RequestVote` RPC using Raft §5.2 term checks and §5.4.1 log up-to-date rules.
    pub fn handle_request_vote(&mut self, args: &RequestVoteArgs) -> RequestVoteReply {
        if args.term < self.current_term {
            return RequestVoteReply {
                term: self.current_term,
                vote_granted: false,
            };
        }

        if args.term > self.current_term {
            self.current_term = args.term;
            self.role = RaftRole::Follower;
            self.voted_for = None;
            self.leader_id = None;
            self.lease_acks.clear();
        }

        let my_last_term = self.last_log_term();
        let my_last_index = self.last_log_index();
        let log_up_to_date = args.last_log_term > my_last_term
            || (args.last_log_term == my_last_term && args.last_log_index >= my_last_index);

        let can_vote = self.voted_for.is_none() || self.voted_for == Some(args.candidate_id);
        let vote_granted = can_vote && log_up_to_date;
        if vote_granted {
            self.voted_for = Some(args.candidate_id);
            self.role = RaftRole::Follower;
        }

        RequestVoteReply {
            term: self.current_term,
            vote_granted,
        }
    }

    /// Append log entries locally as leader before replicating to followers.
    pub fn append_leader_entry(&mut self, cmd: TopologyCommand) -> Result<LogEntry, VektorError> {
        if self.role != RaftRole::Leader {
            return Err(VektorError::QuorumUnavailable {
                shard_id: 0,
                required: self.majority_quorum() as u8,
                alive: 0,
            });
        }
        let index = self.last_log_index() + 1;
        let entry = LogEntry {
            term: self.current_term,
            index,
            cmd,
        };
        self.log.push(entry.clone());
        self.match_index.insert(self.node_id, index);
        self.next_index.insert(self.node_id, index + 1);
        Ok(entry)
    }

    /// Handle `AppendEntries` RPC from a leader (§5.3).
    pub fn handle_append_entries(&mut self, args: &AppendEntriesArgs) -> AppendEntriesReply {
        if args.term < self.current_term {
            return AppendEntriesReply {
                term: self.current_term,
                success: false,
                match_index: self.commit_index,
            };
        }

        if args.term > self.current_term {
            self.current_term = args.term;
            self.voted_for = None;
        }
        self.role = RaftRole::Follower;
        self.leader_id = Some(args.leader_id);
        self.lease_acks.clear();

        // Log consistency check at prev_log_index / prev_log_term.
        let Some(local_prev_term) = self.term_at(args.prev_log_index) else {
            return AppendEntriesReply {
                term: self.current_term,
                success: false,
                match_index: self.last_log_index(),
            };
        };
        if local_prev_term != args.prev_log_term {
            return AppendEntriesReply {
                term: self.current_term,
                success: false,
                match_index: args.prev_log_index.saturating_sub(1),
            };
        }

        // Append any new entries and overwrite conflicting suffixes.
        for entry in &args.entries {
            if let Some(existing_term) = self.term_at(entry.index) {
                if existing_term != entry.term {
                    self.log.truncate((entry.index - 1) as usize);
                    self.log.push(entry.clone());
                }
            } else if entry.index == self.last_log_index() + 1 {
                self.log.push(entry.clone());
            }
        }

        if args.leader_commit > self.commit_index {
            self.commit_index = args.leader_commit.min(self.last_log_index());
            self.apply_committed();
        }

        AppendEntriesReply {
            term: self.current_term,
            success: true,
            match_index: self.last_log_index(),
        }
    }

    /// Recompute the leader's `commit_index` from `match_index` across peers (only committing
    /// entries from `self.current_term` per Raft §5.4.2) and apply newly committed commands.
    pub fn advance_leader_commit_index(&mut self) -> LogIndex {
        if self.role != RaftRole::Leader {
            return self.commit_index;
        }
        let quorum = self.majority_quorum();
        let max_idx = self.last_log_index();
        for n in ((self.commit_index + 1)..=max_idx).rev() {
            if self.term_at(n) != Some(self.current_term) {
                continue;
            }
            let replicated = self
                .peers
                .iter()
                .filter(|&&p| self.match_index.get(&p).copied().unwrap_or(0) >= n)
                .count();
            if replicated >= quorum {
                self.commit_index = n;
                break;
            }
        }
        self.apply_committed();
        self.commit_index
    }

    /// Validate that this node is the active leader for `expected_term` and holds a valid
    /// majority lease, fencing stale leaders after a partition heals.
    pub fn validate_leader_lease(&self, expected_term: Term) -> Result<(), VektorError> {
        if self.current_term != expected_term {
            return Err(VektorError::WrongEpoch {
                current: self.current_term,
                received: expected_term,
            });
        }
        let required = self.majority_quorum();
        if self.role != RaftRole::Leader || self.lease_acks.len() < required {
            return Err(VektorError::QuorumUnavailable {
                shard_id: 0,
                required: required as u8,
                alive: self.lease_acks.len() as u8,
            });
        }
        Ok(())
    }

    /// Apply all committed log entries in `(last_applied..=commit_index]` to `self.snapshot`.
    pub fn apply_committed(&mut self) {
        while self.last_applied < self.commit_index {
            let next_idx = self.last_applied + 1;
            let Some(entry) = self.log.get((next_idx - 1) as usize).cloned() else {
                break;
            };
            self.apply_command(&entry.cmd);
            self.last_applied = next_idx;
        }
    }

    fn apply_command(&mut self, cmd: &TopologyCommand) {
        match cmd {
            TopologyCommand::PromotePrimary {
                shard_id,
                new_primary,
            } => {
                if let Some(shard) = self
                    .snapshot
                    .shards
                    .iter_mut()
                    .find(|s| s.shard_id == *shard_id)
                {
                    shard.followers.retain(|&id| id != *new_primary);
                    if shard.primary != *new_primary {
                        let old_primary = shard.primary;
                        shard.primary = *new_primary;
                        if self.alive_nodes.contains(&old_primary)
                            && !shard.followers.contains(&old_primary)
                        {
                            shard.followers.push(old_primary);
                        }
                    }
                    shard.epoch += 1;
                }
            }
            TopologyCommand::UpdateShard { assignment } => {
                if let Some(existing) = self
                    .snapshot
                    .shards
                    .iter_mut()
                    .find(|s| s.shard_id == assignment.shard_id)
                {
                    *existing = assignment.clone();
                } else {
                    self.snapshot.shards.push(assignment.clone());
                }
            }
            TopologyCommand::ApplyRebalance {
                assignments,
                active_nodes,
            } => {
                self.snapshot.shards = assignments.clone();
                self.alive_nodes = active_nodes.iter().copied().collect();
            }
            TopologyCommand::RegisterNode { node_id, addr } => {
                self.node_addrs.insert(*node_id, addr.clone());
                self.alive_nodes.insert(*node_id);
            }
            TopologyCommand::RemoveNode { node_id } => {
                self.alive_nodes.remove(node_id);
            }
            TopologyCommand::UpsertApiKey { record } => {
                if record.revoked {
                    self.revoked_keys.insert(record.key_id.clone());
                } else {
                    self.revoked_keys.remove(&record.key_id);
                }
                self.api_keys.insert(record.key_id.clone(), record.clone());
            }
            TopologyCommand::RevokeApiKey { key_id } => {
                self.revoked_keys.insert(key_id.clone());
                if let Some(existing) = self.api_keys.get_mut(key_id) {
                    existing.revoked = true;
                }
            }
            TopologyCommand::SetTenantQuota {
                org_id,
                max_qps_milli,
                burst_milli,
                max_vectors,
                max_memory_bytes,
            } => {
                let policy = crate::security::TenantQuotaPolicy {
                    max_qps: (*max_qps_milli as f64) / 1000.0,
                    burst_capacity: (*burst_milli as f64) / 1000.0,
                    max_vectors: *max_vectors,
                    max_memory_bytes: *max_memory_bytes,
                };
                self.tenant_quotas.insert(org_id.clone(), policy);
            }
        }
    }
}

impl TopologyStore for RaftNode {
    fn snapshot(&self) -> TopologySnapshot {
        self.snapshot.clone()
    }

    fn promote_primary(
        &mut self,
        shard_id: ShardId,
        new_primary: NodeId,
    ) -> Result<Epoch, VektorError> {
        self.validate_leader_lease(self.current_term)?;
        if !self.snapshot.shards.iter().any(|s| s.shard_id == shard_id) {
            return Err(VektorError::InvalidConfig("unknown shard_id"));
        }
        let entry = self.append_leader_entry(TopologyCommand::PromotePrimary {
            shard_id,
            new_primary,
        })?;
        if self.peers.len() <= 1 {
            self.commit_index = entry.index;
            self.apply_committed();
        }
        self.snapshot
            .shards
            .iter()
            .find(|s| s.shard_id == shard_id)
            .map(|s| s.epoch)
            .ok_or(VektorError::InvalidConfig("unknown shard_id"))
    }
}

/// Deterministic multi-node Raft cluster controller with partition/isolation simulation,
/// leader-lease fencing, and `TopologyStore` integration.
#[derive(Debug, Clone)]
pub struct RaftCluster {
    nodes: BTreeMap<NodeId, RaftNode>,
    isolated: BTreeSet<NodeId>,
}

impl RaftCluster {
    /// Create a Raft cluster of `node_ids` initialized with `initial_snapshot`.
    pub fn new(
        node_ids: Vec<NodeId>,
        initial_snapshot: TopologySnapshot,
    ) -> Result<Self, VektorError> {
        initial_snapshot.config.validate()?;
        if node_ids.is_empty() {
            return Err(VektorError::InvalidConfig(
                "at least 1 Raft node is required",
            ));
        }
        let mut nodes = BTreeMap::new();
        for &id in &node_ids {
            nodes.insert(
                id,
                RaftNode::new(id, node_ids.clone(), initial_snapshot.clone()),
            );
        }
        Ok(Self {
            nodes,
            isolated: BTreeSet::new(),
        })
    }

    /// Create a Raft cluster from data-plane `(node_id, addr)` descriptors and build the initial shard ring.
    pub fn from_initial_nodes(
        config: ClusterConfig,
        initial_nodes: Vec<(NodeId, String)>,
    ) -> Result<Self, VektorError> {
        config.validate()?;
        if initial_nodes.is_empty() {
            return Err(VektorError::InvalidConfig("at least 1 node required"));
        }
        let node_ids: Vec<NodeId> = initial_nodes.iter().map(|(id, _)| *id).collect();
        let rf = (config.replication_factor as usize).min(node_ids.len());
        let mut shards = Vec::with_capacity(config.num_shards as usize);
        for shard_id in 0..config.num_shards {
            let primary_idx = (shard_id as usize) % node_ids.len();
            let primary = node_ids[primary_idx];
            let mut followers = Vec::with_capacity(rf.saturating_sub(1));
            for offset in 1..rf {
                followers.push(node_ids[(primary_idx + offset) % node_ids.len()]);
            }
            let state = if rf < (config.write_quorum as usize) {
                ShardState::UnderQuorum
            } else if rf < (config.replication_factor as usize) {
                ShardState::DegradedOk
            } else {
                ShardState::Healthy
            };
            shards.push(ShardAssignment {
                shard_id,
                epoch: 1,
                primary,
                followers,
                state,
            });
        }

        let snapshot = TopologySnapshot { config, shards };
        let mut cluster = Self::new(node_ids, snapshot)?;
        for (id, addr) in initial_nodes {
            for node in cluster.nodes.values_mut() {
                node.node_addrs.insert(id, addr.clone());
                node.alive_nodes.insert(id);
            }
        }
        Ok(cluster)
    }

    /// Return immutable reference to `node_id`.
    #[must_use]
    pub fn node(&self, node_id: NodeId) -> Option<&RaftNode> {
        self.nodes.get(&node_id)
    }

    /// Return mutable reference to `node_id`.
    pub fn node_mut(&mut self, node_id: NodeId) -> Option<&mut RaftNode> {
        self.nodes.get_mut(&node_id)
    }

    /// All voting Raft member IDs.
    #[must_use]
    pub fn member_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }

    /// Isolate `node_id` from all network communication (simulating asymmetric/symmetric partition or kill).
    pub fn isolate_node(&mut self, node_id: NodeId) {
        self.isolated.insert(node_id);
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.lease_acks.clear();
            node.lease_acks.insert(node_id);
        }
    }

    /// Heal network partition for `node_id` and synchronize it with the current cluster term/log.
    pub fn heal_node(&mut self, node_id: NodeId) {
        self.isolated.remove(&node_id);
        if let Some(leader_id) = self.leader_id() {
            if leader_id != node_id {
                let _ = self.replicate_from_leader(leader_id);
            }
        }
    }

    /// Check whether `node_id` is currently network-isolated.
    #[must_use]
    pub fn is_isolated(&self, node_id: NodeId) -> bool {
        self.isolated.contains(&node_id)
    }

    fn can_communicate(&self, a: NodeId, b: NodeId) -> bool {
        if a == b {
            return true;
        }
        !self.isolated.contains(&a) && !self.isolated.contains(&b)
    }

    /// Return the current active leader with the highest term, if one exists and holds a valid lease.
    #[must_use]
    pub fn leader_id(&self) -> Option<NodeId> {
        self.nodes
            .values()
            .filter(|n| {
                n.role == RaftRole::Leader
                    && !self.isolated.contains(&n.node_id)
                    && n.lease_acks.len() >= n.majority_quorum()
            })
            .max_by_key(|n| n.current_term)
            .map(|n| n.node_id)
    }

    /// Dynamically add a voting node (and optional data-plane address) to the Raft cluster.
    pub fn add_node(
        &mut self,
        node_id: NodeId,
        addr: Option<String>,
    ) -> Result<(Term, LogIndex), VektorError> {
        if !self.nodes.contains_key(&node_id) {
            let base_snap = self.snapshot();
            let mut all_peers: Vec<NodeId> = self.nodes.keys().copied().collect();
            all_peers.push(node_id);
            all_peers.sort_unstable();

            for existing in self.nodes.values_mut() {
                existing.peers = all_peers.clone();
                if existing.role == RaftRole::Leader {
                    existing
                        .next_index
                        .insert(node_id, existing.last_log_index() + 1);
                    existing.match_index.insert(node_id, 0);
                }
            }

            let mut new_node = RaftNode::new(node_id, all_peers, base_snap);
            if let Some(leader_id) = self.leader_id() {
                if let Some(leader) = self.nodes.get(&leader_id) {
                    new_node.node_addrs = leader.node_addrs.clone();
                    new_node.alive_nodes = leader.alive_nodes.clone();
                }
            }
            self.nodes.insert(node_id, new_node);
        }

        if let Some(a) = addr {
            self.propose(TopologyCommand::RegisterNode { node_id, addr: a })
        } else if let Some(leader_id) = self.leader_id() {
            self.replicate_from_leader(leader_id)?;
            let leader = self
                .nodes
                .get(&leader_id)
                .ok_or(VektorError::InvalidConfig("missing leader"))?;
            Ok((leader.current_term, leader.commit_index))
        } else {
            let (_, term) = self.elect_any_leader()?;
            Ok((term, 0))
        }
    }

    /// Trigger an election for `candidate_id` across all reachable peers.
    pub fn start_election(&mut self, candidate_id: NodeId) -> Result<Term, VektorError> {
        let peer_ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        let req = {
            let candidate = self
                .nodes
                .get_mut(&candidate_id)
                .ok_or(VektorError::InvalidConfig("unknown candidate node_id"))?;
            candidate.start_election()
        };

        let mut granted_voters = BTreeSet::new();
        granted_voters.insert(candidate_id);
        let mut max_seen_term = req.term;

        for peer_id in peer_ids {
            if peer_id == candidate_id || !self.can_communicate(candidate_id, peer_id) {
                continue;
            }
            if let Some(peer) = self.nodes.get_mut(&peer_id) {
                let reply = peer.handle_request_vote(&req);
                if reply.term > max_seen_term {
                    max_seen_term = reply.term;
                }
                if reply.vote_granted && reply.term == req.term {
                    granted_voters.insert(peer_id);
                }
            }
        }

        let quorum = {
            let candidate = self
                .nodes
                .get_mut(&candidate_id)
                .ok_or(VektorError::InvalidConfig("unknown candidate node_id"))?;
            if max_seen_term > candidate.current_term {
                candidate.current_term = max_seen_term;
                candidate.role = RaftRole::Follower;
                candidate.voted_for = None;
                candidate.leader_id = None;
                candidate.lease_acks.clear();
                return Err(VektorError::WrongEpoch {
                    current: max_seen_term,
                    received: req.term,
                });
            }
            candidate.majority_quorum()
        };

        if granted_voters.len() < quorum {
            return Err(VektorError::QuorumUnavailable {
                shard_id: 0,
                required: quorum as u8,
                alive: granted_voters.len() as u8,
            });
        }

        if let Some(candidate) = self.nodes.get_mut(&candidate_id) {
            candidate.become_leader();
            candidate.lease_acks = granted_voters;
        }
        self.replicate_from_leader(candidate_id)?;
        Ok(req.term)
    }

    /// Deterministically elect the reachable node with the most up-to-date log.
    pub fn elect_any_leader(&mut self) -> Result<(NodeId, Term), VektorError> {
        let best_candidate = self
            .nodes
            .values()
            .filter(|n| !self.isolated.contains(&n.node_id))
            .max_by_key(|n| {
                (
                    n.last_log_term(),
                    n.last_log_index(),
                    std::cmp::Reverse(n.node_id),
                )
            })
            .map(|n| n.node_id)
            .ok_or(VektorError::QuorumUnavailable {
                shard_id: 0,
                required: 1,
                alive: 0,
            })?;

        let term = self.start_election(best_candidate)?;
        Ok((best_candidate, term))
    }

    /// Replicate log entries / heartbeats from `leader_id` to all reachable followers.
    pub fn replicate_from_leader(&mut self, leader_id: NodeId) -> Result<LogIndex, VektorError> {
        let (leader_term, peers, quorum) = {
            let leader = self
                .nodes
                .get(&leader_id)
                .ok_or(VektorError::InvalidConfig("unknown leader node_id"))?;
            if leader.role != RaftRole::Leader {
                return Err(VektorError::QuorumUnavailable {
                    shard_id: 0,
                    required: leader.majority_quorum() as u8,
                    alive: 0,
                });
            }
            (
                leader.current_term,
                leader.peers.clone(),
                leader.majority_quorum(),
            )
        };

        let mut active_acks = BTreeSet::new();
        active_acks.insert(leader_id);

        for peer_id in &peers {
            let peer_id = *peer_id;
            if peer_id == leader_id || !self.can_communicate(leader_id, peer_id) {
                continue;
            }

            // Catch up follower log from next_index, backing up if prev_log doesn't match.
            loop {
                let args = {
                    let Some(leader) = self.nodes.get(&leader_id) else {
                        break;
                    };
                    let next_idx = leader.next_index.get(&peer_id).copied().unwrap_or(1).max(1);
                    let prev_log_index = next_idx - 1;
                    let prev_log_term = leader.term_at(prev_log_index).unwrap_or(0);
                    let entries: Vec<LogEntry> = leader
                        .log
                        .iter()
                        .skip(prev_log_index as usize)
                        .cloned()
                        .collect();
                    AppendEntriesArgs {
                        term: leader_term,
                        leader_id,
                        prev_log_index,
                        prev_log_term,
                        entries,
                        leader_commit: leader.commit_index,
                    }
                };

                let reply = {
                    let Some(follower) = self.nodes.get_mut(&peer_id) else {
                        break;
                    };
                    follower.handle_append_entries(&args)
                };

                if reply.term > leader_term {
                    if let Some(leader) = self.nodes.get_mut(&leader_id) {
                        leader.current_term = reply.term;
                        leader.role = RaftRole::Follower;
                        leader.voted_for = None;
                        leader.leader_id = None;
                        leader.lease_acks.clear();
                    }
                    return Err(VektorError::WrongEpoch {
                        current: reply.term,
                        received: leader_term,
                    });
                }

                if reply.success {
                    if let Some(leader) = self.nodes.get_mut(&leader_id) {
                        leader.match_index.insert(peer_id, reply.match_index);
                        leader.next_index.insert(peer_id, reply.match_index + 1);
                    }
                    active_acks.insert(peer_id);
                    break;
                }

                if let Some(leader) = self.nodes.get_mut(&leader_id) {
                    let cur_next = leader.next_index.get(&peer_id).copied().unwrap_or(1);
                    if cur_next <= 1 {
                        break;
                    }
                    leader
                        .next_index
                        .insert(peer_id, (reply.match_index + 1).min(cur_next - 1).max(1));
                } else {
                    break;
                }
            }
        }

        let new_commit = {
            let leader = self
                .nodes
                .get_mut(&leader_id)
                .ok_or(VektorError::InvalidConfig("unknown leader node_id"))?;
            leader.lease_acks = active_acks.clone();
            if active_acks.len() < quorum {
                return Err(VektorError::QuorumUnavailable {
                    shard_id: 0,
                    required: quorum as u8,
                    alive: active_acks.len() as u8,
                });
            }
            leader.advance_leader_commit_index()
        };

        // Second pass: notify reachable followers of the updated leader_commit so their state
        // machines are synchronously up to date.
        for peer_id in peers {
            if peer_id == leader_id || !self.can_communicate(leader_id, peer_id) {
                continue;
            }
            let args = {
                let Some(leader) = self.nodes.get(&leader_id) else {
                    continue;
                };
                let prev_log_index = leader.last_log_index();
                let prev_log_term = leader.last_log_term();
                AppendEntriesArgs {
                    term: leader_term,
                    leader_id,
                    prev_log_index,
                    prev_log_term,
                    entries: Vec::new(),
                    leader_commit: new_commit,
                }
            };
            if let Some(follower) = self.nodes.get_mut(&peer_id) {
                let _ = follower.handle_append_entries(&args);
            }
        }

        Ok(new_commit)
    }

    /// Propose a `TopologyCommand` on a specific `leader_id` (useful for testing split-brain fencing
    /// when an isolated old leader attempts to propose).
    pub fn propose_on(
        &mut self,
        leader_id: NodeId,
        cmd: TopologyCommand,
    ) -> Result<(Term, LogIndex), VektorError> {
        let entry = {
            let leader = self
                .nodes
                .get_mut(&leader_id)
                .ok_or(VektorError::InvalidConfig("unknown leader node_id"))?;
            leader.append_leader_entry(cmd)?
        };
        let commit_idx = self.replicate_from_leader(leader_id)?;
        if commit_idx < entry.index {
            return Err(VektorError::QuorumUnavailable {
                shard_id: 0,
                required: 2,
                alive: 1,
            });
        }
        Ok((entry.term, entry.index))
    }

    /// Propose a `TopologyCommand` on the current active cluster leader.
    pub fn propose(&mut self, cmd: TopologyCommand) -> Result<(Term, LogIndex), VektorError> {
        let leader_id = self.leader_id().ok_or_else(|| {
            let required = self
                .nodes
                .values()
                .next()
                .map_or(1, |n| n.majority_quorum() as u8);
            VektorError::QuorumUnavailable {
                shard_id: 0,
                required,
                alive: 0,
            }
        })?;
        self.propose_on(leader_id, cmd)
    }

    /// Validate that `node_id` holds a valid leader lease for `expected_term` and has not been
    /// superseded by a newer term in the cluster.
    pub fn validate_leader_lease(
        &self,
        node_id: NodeId,
        expected_term: Term,
    ) -> Result<(), VektorError> {
        let highest_cluster_term = self
            .nodes
            .values()
            .map(|n| n.current_term)
            .max()
            .unwrap_or(0);
        if expected_term < highest_cluster_term {
            return Err(VektorError::WrongEpoch {
                current: highest_cluster_term,
                received: expected_term,
            });
        }
        let node = self
            .nodes
            .get(&node_id)
            .ok_or(VektorError::InvalidConfig("unknown node_id"))?;
        if self.isolated.contains(&node_id) {
            return Err(VektorError::QuorumUnavailable {
                shard_id: 0,
                required: node.majority_quorum() as u8,
                alive: 1,
            });
        }
        node.validate_leader_lease(expected_term)
    }

    /// Push all committed shard configurations from the active Raft leader to live data-plane nodes.
    pub async fn push_all_shard_configs(&self) -> Result<(), VektorError> {
        let leader_id = self.leader_id().ok_or(VektorError::QuorumUnavailable {
            shard_id: 0,
            required: 2,
            alive: 0,
        })?;
        let leader = self
            .nodes
            .get(&leader_id)
            .ok_or(VektorError::InvalidConfig("missing leader"))?;
        for shard in &leader.snapshot.shards {
            push_shard_config_to_nodes(
                &leader.snapshot.config,
                &leader.node_addrs,
                &leader.alive_nodes,
                shard,
                None,
            )
            .await;
        }
        Ok(())
    }

    /// Build a `GetTopologyResponse` from the authoritative leader snapshot.
    #[must_use]
    pub fn build_topology_response(&self) -> GetTopologyResponse {
        let Some(ref_node) = self
            .leader_id()
            .and_then(|id| self.nodes.get(&id))
            .or_else(|| self.nodes.values().max_by_key(|n| n.commit_index))
        else {
            return GetTopologyResponse::default();
        };

        let mut owned_counts: BTreeMap<NodeId, u32> = BTreeMap::new();
        let mut wire_shards = Vec::with_capacity(ref_node.snapshot.shards.len());
        for s in &ref_node.snapshot.shards {
            *owned_counts.entry(s.primary).or_insert(0) += 1;
            for &f in &s.followers {
                *owned_counts.entry(f).or_insert(0) += 1;
            }
            let primary_addr = ref_node
                .node_addrs
                .get(&s.primary)
                .cloned()
                .unwrap_or_default();
            let follower_addrs = s
                .followers
                .iter()
                .filter_map(|id| ref_node.node_addrs.get(id).cloned())
                .collect();
            wire_shards.push(WireShardAssignment {
                shard_id: s.shard_id,
                epoch: s.epoch,
                primary_node_id: s.primary,
                primary_addr,
                follower_node_ids: s.followers.clone(),
                follower_addrs,
                state: shard_state_to_u32(s.state),
            });
        }

        let wire_nodes = ref_node
            .node_addrs
            .iter()
            .map(|(&node_id, addr)| {
                let alive = ref_node.alive_nodes.contains(&node_id);
                WireNodeStatus {
                    node_id,
                    addr: addr.clone(),
                    alive,
                    memory_bytes: 0,
                    live_vectors: 0,
                    owned_shards: if alive {
                        owned_counts.get(&node_id).copied().unwrap_or(0)
                    } else {
                        0
                    },
                }
            })
            .collect();

        GetTopologyResponse {
            num_shards: ref_node.snapshot.config.num_shards,
            replication_factor: u32::from(ref_node.snapshot.config.replication_factor),
            write_quorum: u32::from(ref_node.snapshot.config.write_quorum),
            dimensions: ref_node.snapshot.config.dimensions as u32,
            metric: match ref_node.snapshot.config.metric {
                DistanceMetric::Cosine => 0,
                DistanceMetric::L2 => 1,
            },
            shards: wire_shards,
            nodes: wire_nodes,
        }
    }
}

impl TopologyStore for RaftCluster {
    fn snapshot(&self) -> TopologySnapshot {
        if let Some(leader_id) = self.leader_id() {
            if let Some(leader) = self.nodes.get(&leader_id) {
                return leader.snapshot.clone();
            }
        }
        self.nodes
            .values()
            .max_by_key(|n| (n.commit_index, n.current_term))
            .map_or_else(
                || TopologySnapshot {
                    config: ClusterConfig::default(),
                    shards: Vec::new(),
                },
                |n| n.snapshot.clone(),
            )
    }

    fn promote_primary(
        &mut self,
        shard_id: ShardId,
        new_primary: NodeId,
    ) -> Result<Epoch, VektorError> {
        let exists = self
            .snapshot()
            .shards
            .iter()
            .any(|s| s.shard_id == shard_id);
        if !exists {
            return Err(VektorError::InvalidConfig("unknown shard_id"));
        }
        self.propose(TopologyCommand::PromotePrimary {
            shard_id,
            new_primary,
        })?;
        self.snapshot()
            .shards
            .iter()
            .find(|s| s.shard_id == shard_id)
            .map(|s| s.epoch)
            .ok_or(VektorError::InvalidConfig("unknown shard_id"))
    }
}

/// Thread-safe gRPC `TopologyRpc` wrapper around `RaftCluster` so `ClusterClient` can query
/// a Raft-backed control plane transparently.
#[derive(Debug, Clone)]
pub struct RaftTopologyService {
    pub cluster: Arc<RwLock<RaftCluster>>,
}

impl RaftTopologyService {
    #[must_use]
    pub fn new(cluster: Arc<RwLock<RaftCluster>>) -> Self {
        Self { cluster }
    }
}

#[tonic::async_trait]
impl TopologyRpc for RaftTopologyService {
    async fn get_topology(
        &self,
        _request: Request<GetTopologyRequest>,
    ) -> Result<Response<GetTopologyResponse>, Status> {
        let guard = self
            .cluster
            .read()
            .map_err(|_| Status::internal("raft cluster lock poisoned"))?;
        Ok(Response::new(guard.build_topology_response()))
    }
}
