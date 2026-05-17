# =============================================================================
# ATCS - Adaptive Traffic Control System
# shared/version.py | Version tracking & dev sync check
#
# Rules:
#   - Bump SCHEMA_VERSION whenever schemas.py wire format changes.
#     Both Edge and Server must be on the same SCHEMA_VERSION or they
#     will refuse to run (assert_compatible() raises RuntimeError).
#   - Bump APP_VERSION freely for any other code change (follows semver).
#   - This file has ZERO imports — not even stdlib. It must be importable
#     on any device before any dependency is installed.
# =============================================================================


# -----------------------------------------------------------------------------
# 1. Application Version (semver)
#    Bump on every meaningful code change.
#    MAJOR.MINOR.PATCH
#      MAJOR → breaking architecture change
#      MINOR → new feature, backward-compatible
#      PATCH → bug fix, refactor, doc update
# -----------------------------------------------------------------------------

APP_VERSION: str = "0.1.0"


# -----------------------------------------------------------------------------
# 2. Schema Version (integer, monotonically increasing)
#    Bump ONLY when the FramePayload or PhaseDecision wire format changes
#    (fields added, removed, renamed, or type-changed).
#    Edge and Server MUST share the same SCHEMA_VERSION at runtime.
# -----------------------------------------------------------------------------

SCHEMA_VERSION: int = 1


# -----------------------------------------------------------------------------
# 3. Minimum Compatible Schema Version
#    If you add an optional field to schemas.py (backward-compatible),
#    you can bump SCHEMA_VERSION but keep MIN_COMPATIBLE_SCHEMA_VERSION
#    at the previous value — older nodes can still parse the new payload.
#    If you remove or rename a field (breaking), bump both to the same value.
# -----------------------------------------------------------------------------

MIN_COMPATIBLE_SCHEMA_VERSION: int = 1


# -----------------------------------------------------------------------------
# 4. Compatibility Check
#    Call assert_compatible(remote_schema_version) on first MQTT handshake
#    or at startup when both sides are known.
# -----------------------------------------------------------------------------

def assert_compatible(remote_schema_version: int, remote_label: str = "remote") -> None:
    """
    Verify that a remote node's schema version is compatible with ours.

    Parameters
    ----------
    remote_schema_version : int
        SCHEMA_VERSION reported by the other side (Edge or Server).
    remote_label : str
        Human-readable label for the remote node (used in error messages).

    Raises
    ------
    RuntimeError
        If the remote schema version is incompatible with this node's
        MIN_COMPATIBLE_SCHEMA_VERSION or SCHEMA_VERSION.
    """
    if remote_schema_version < MIN_COMPATIBLE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Schema version mismatch: {remote_label} is running schema v"
            f"{remote_schema_version}, but this node requires at least v"
            f"{MIN_COMPATIBLE_SCHEMA_VERSION}. "
            f"Update {remote_label} to schema v{SCHEMA_VERSION}."
        )

    if remote_schema_version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Schema version mismatch: {remote_label} is running schema v"
            f"{remote_schema_version}, but this node only understands up to v"
            f"{SCHEMA_VERSION}. "
            f"Update this node to match {remote_label}."
        )


def is_compatible(remote_schema_version: int) -> bool:
    """
    Non-raising version of assert_compatible(). Returns True if compatible.

    Useful for logging a warning instead of hard-failing.
    """
    return MIN_COMPATIBLE_SCHEMA_VERSION <= remote_schema_version <= SCHEMA_VERSION


def version_banner() -> str:
    """
    Returns a formatted one-line version string for startup logs.

    Example output:
        ATCS v0.1.0 | Schema v1 (min-compat: v1)
    """
    return (
        f"ATCS v{APP_VERSION} | "
        f"Schema v{SCHEMA_VERSION} "
        f"(min-compat: v{MIN_COMPATIBLE_SCHEMA_VERSION})"
    )


# =============================================================================
# USAGE EXAMPLE (run directly to verify)
# =============================================================================

if __name__ == "__main__":
    print("=== ATCS Version Smoke Test ===\n")

    print(f"[Version Banner]")
    print(f"  {version_banner()}\n")

    print(f"[Compatibility Checks]")

    # Same version → OK
    try:
        assert_compatible(SCHEMA_VERSION, remote_label="Server")
        print(f"  ✅ Same version (v{SCHEMA_VERSION}) → compatible")
    except RuntimeError as e:
        print(f"  ❌ {e}")

    # Older but still within min-compat → OK (if range allows)
    older = MIN_COMPATIBLE_SCHEMA_VERSION
    if is_compatible(older):
        print(f"  ✅ Older version (v{older}) → compatible via min-compat")
    else:
        print(f"  ⚠️  Older version (v{older}) → NOT compatible")

    # Too old → should fail
    too_old = MIN_COMPATIBLE_SCHEMA_VERSION - 1
    try:
        assert_compatible(too_old, remote_label="OldEdge")
        print(f"  ❌ Should have raised for v{too_old}")
    except RuntimeError as e:
        print(f"  ✅ Correctly rejected too-old version (v{too_old}): {e}")

    # Too new → should fail
    too_new = SCHEMA_VERSION + 1
    try:
        assert_compatible(too_new, remote_label="NewServer")
        print(f"  ❌ Should have raised for v{too_new}")
    except RuntimeError as e:
        print(f"  ✅ Correctly rejected too-new version (v{too_new}): {e}")

    print("\n=== All version checks passed ===")
