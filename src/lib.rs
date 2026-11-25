#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod cli;
pub mod tui;

pub fn healthcheck() -> &'static str {
    "ok"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_test() {
        assert_eq!(healthcheck(), "ok");
    }
}
