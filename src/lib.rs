//! VektorDist CLI, declarative node configuration, multi-process cluster supervisor,
//! and interactive `ratatui` chaos dashboard.

#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod cli;
pub mod config;
pub mod tui;
