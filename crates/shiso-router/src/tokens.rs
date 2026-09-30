use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Context, Result};
use rand::RngCore;
use rusqlite::Connection;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::limits::ResetInterval;

/// Prefix prepended to generated API tokens.
pub const TOKEN_PREFIX: &str = "sk-";

/// Identity the admin chat console dispatches under. It has no row in
/// `tokens` and no token hash, so it can never authenticate anything — it
/// exists only to give console traffic a name in the logs and analytics.
/// Real key ids are UUIDs, so this can never collide with one.
pub const ADMIN_KEY_ID: &str = "admin";
/// Synthetic display name for the admin console key.
pub const ADMIN_KEY_NAME: &str = "admin console";

/// Display name for a token id with no `tokens` row. Only the console's
/// synthetic id has one; a genuinely deleted key stays null.
pub fn synthetic_key_name(token_id: &str) -> Option<&'static str> {
    (token_id == ADMIN_KEY_ID).then_some(ADMIN_KEY_NAME)
}

/// Current schema version, tracked via `PRAGMA user_version`. Version 2
/// added key limits/controls columns and the `request_logs` table; version 3
/// moved the backend/model catalog out of the config file into the
/// `backends` / `models` / `model_targets` tables; version 4 added
/// `models.aliases` (extra inbound names for a route); version 5 added
/// `idx_logs_backend_time`, which the per-provider analytics queries need;
/// version 6 added device-aware scheduling — the `exclusive_groups` table
/// plus the per-backend scheduling class / concurrency / token-cap columns,
/// the per-model QoS deadline columns, and `model_targets.fallback_only`;
/// version 7 added `request_logs.service_tier`, the client's
/// `x-service-tier` hint as it was forwarded.
const SCHEMA_VERSION: i64 = 7;

/// A client API key as stored — everything except the secret, which exists
/// only as a SHA-256 hash. Records are immutable snapshots: admin updates
/// build a new record and swap the cache entry wholesale.
#[derive(Debug, Clone, Serialize)]
pub struct KeyRecord {
    /// Unique key identifier.
    pub id: String,
    /// Human-readable key name.
    pub name: String,
    /// Creation timestamp in RFC3339 format.
    pub created_at: String,
    /// Whether key is disabled.
    pub disabled: bool,
    /// USD budget per reset window; `None` = unlimited.
    pub credit_limit: Option<f64>,
    /// Reset interval for credit budget.
    pub credit_limit_reset: ResetInterval,
    /// Total-token budget per reset window; `None` = unlimited.
    pub token_limit: Option<i64>,
    /// Reset interval for token budget.
    pub token_limit_reset: ResetInterval,
    /// Models this key may call; `None` = all configured models.
    pub allowed_models: Option<Vec<String>>,
}

impl KeyRecord {
    /// Constructs a new key record with the given name.
    pub fn new(name: String) -> Self {
        KeyRecord {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            created_at: now_rfc3339(),
            disabled: false,
            credit_limit: None,
            credit_limit_reset: ResetInterval::Never,
            token_limit: None,
            token_limit_reset: ResetInterval::Never,
            allowed_models: None,
        }
    }

    /// The record the admin console runs as: unlimited and unrestricted,
    /// because the admin token already authorizes it and a setup test must
    /// not be blocked by a quota it does not own.
    pub fn admin_console() -> Self {
        KeyRecord {
            id: ADMIN_KEY_ID.to_string(),
            ..KeyRecord::new(ADMIN_KEY_NAME.to_string())
        }
    }

    /// Checks whether the key permits calling the specified model.
    pub fn allows_model(&self, model: &str) -> bool {
        match &self.allowed_models {
            None => true,
            Some(list) => list.iter().any(|m| m == model),
        }
    }

    /// Checks whether the key has credit or token limits configured.
    pub fn has_limits(&self) -> bool {
        self.credit_limit.is_some() || self.token_limit.is_some()
    }
}

/// Computes the SHA-256 hash of a plaintext token.
pub fn hash_token(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

/// Generates a cryptographically random plaintext API token.
pub fn generate_token() -> String {
    let mut buf = [0u8; 32];
    rand::rng().fill_bytes(&mut buf);
    format!("{TOKEN_PREFIX}{}", hex::encode(buf))
}

/// SQLite-backed store for keys and request logs. Key writes only happen on
/// admin operations and log writes go through one batching task, so a
/// single write connection behind a mutex (used via `spawn_blocking`) is
/// plenty. A second read-only connection serves admin analytics queries so
/// they never contend with the writer (WAL allows concurrent readers).
#[derive(Debug, Clone)]
pub struct Db {
    conn: Arc<Mutex<Connection>>,
    read_conn: Arc<Mutex<Connection>>,
}

impl Db {
    /// Opens or creates the SQLite database at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open sqlite database at {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        migrate(&conn)?;
        let read_conn =
            Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
                .with_context(|| format!("failed to open read connection at {}", path.display()))?;
        Ok(Db {
            conn: Arc::new(Mutex::new(conn)),
            read_conn: Arc::new(Mutex::new(read_conn)),
        })
    }

    /// Runs a closure with the write connection locked.
    /// Runs a closure with the write connection locked.
    pub fn with_write<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        let conn = self.conn.lock().unwrap();
        f(&conn)
    }

    /// Runs a closure with the read-only connection locked.
    pub fn with_read<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        let conn = self.read_conn.lock().unwrap();
        f(&conn)
    }

    /// Inserts a new key record and its token hash.
    pub fn insert(&self, record: &KeyRecord, token_hash: &[u8; 32]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO tokens (id, name, token_hash, created_at, disabled,
                                 credit_limit, credit_limit_reset,
                                 token_limit, token_limit_reset, allowed_models)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                record.id,
                record.name,
                hex::encode(token_hash),
                record.created_at,
                record.disabled,
                record.credit_limit,
                record.credit_limit_reset.as_str(),
                record.token_limit,
                record.token_limit_reset.as_str(),
                allowed_models_to_sql(&record.allowed_models)?,
            ],
        )?;
        Ok(())
    }

    /// Lists all stored key records.
    pub fn list(&self) -> Result<Vec<KeyRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT {RECORD_COLUMNS} FROM tokens ORDER BY created_at, id"
        ))?;
        let rows = stmt
            .query_map([], record_from_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Fetches a key record by id.
    ///
    /// Returns `Ok(None)` if no key with the given id exists.
    pub fn get(&self, id: &str) -> Result<Option<KeyRecord>> {
        let conn = self.conn.lock().unwrap();
        let record = conn
            .query_row(
                &format!("SELECT {RECORD_COLUMNS} FROM tokens WHERE id = ?1"),
                [id],
                record_from_row,
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                e => Err(e),
            })?;
        Ok(record)
    }

    /// Overwrites the mutable columns of a key.
    ///
    /// Returns `Ok(None)` if the row did not exist.
    pub fn update(&self, record: &KeyRecord) -> Result<Option<[u8; 32]>> {
        let conn = self.conn.lock().unwrap();
        let hash = lookup_hash(&conn, &record.id)?;
        let Some(hash) = hash else { return Ok(None) };
        conn.execute(
            "UPDATE tokens SET name = ?2, disabled = ?3,
                    credit_limit = ?4, credit_limit_reset = ?5,
                    token_limit = ?6, token_limit_reset = ?7,
                    allowed_models = ?8
             WHERE id = ?1",
            rusqlite::params![
                record.id,
                record.name,
                record.disabled,
                record.credit_limit,
                record.credit_limit_reset.as_str(),
                record.token_limit,
                record.token_limit_reset.as_str(),
                allowed_models_to_sql(&record.allowed_models)?,
            ],
        )?;
        Ok(Some(hash))
    }

    /// Deletes a key by id.
    ///
    /// Returns `Ok(None)` if the key did not exist.
    pub fn delete(&self, id: &str) -> Result<Option<[u8; 32]>> {
        let conn = self.conn.lock().unwrap();
        let hash = lookup_hash(&conn, id)?;
        let Some(hash) = hash else { return Ok(None) };
        conn.execute("DELETE FROM tokens WHERE id = ?1", [id])?;
        Ok(Some(hash))
    }

    /// Loads all key hashes and records into memory.
    pub fn load_all(&self) -> Result<Vec<([u8; 32], KeyRecord)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!("SELECT token_hash, {RECORD_COLUMNS} FROM tokens"))?;
        let rows = stmt
            .query_map([], |row| {
                let hash: String = row.get(0)?;
                let record = record_from_offset_row(row, 1)?;
                Ok((hash, record))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        rows.into_iter()
            .map(|(hash, record)| Ok((decode_hash(&hash)?, record)))
            .collect()
    }
}

const RECORD_COLUMNS: &str = "id, name, created_at, disabled, credit_limit, \
     credit_limit_reset, token_limit, token_limit_reset, allowed_models";

fn migrate(conn: &Connection) -> Result<()> {
    let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version >= SCHEMA_VERSION {
        return Ok(());
    }

    // Fresh databases get the full current schema; databases created before
    // schema versioning existed (version 0 with a populated tokens table)
    // are brought up additively — ALTERs below only run for columns that
    // are actually missing, so this block is idempotent either way.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS tokens (
            id                 TEXT PRIMARY KEY,
            name               TEXT NOT NULL,
            token_hash         TEXT NOT NULL UNIQUE,
            created_at         TEXT NOT NULL,
            disabled           INTEGER NOT NULL DEFAULT 0,
            credit_limit       REAL,
            credit_limit_reset TEXT NOT NULL DEFAULT 'never',
            token_limit        INTEGER,
            token_limit_reset  TEXT NOT NULL DEFAULT 'never',
            allowed_models     TEXT
        );",
    )?;

    let existing = table_columns(conn, "tokens")?;
    for (col, ddl) in [
        (
            "disabled",
            "ALTER TABLE tokens ADD COLUMN disabled INTEGER NOT NULL DEFAULT 0",
        ),
        (
            "credit_limit",
            "ALTER TABLE tokens ADD COLUMN credit_limit REAL",
        ),
        (
            "credit_limit_reset",
            "ALTER TABLE tokens ADD COLUMN credit_limit_reset TEXT NOT NULL DEFAULT 'never'",
        ),
        (
            "token_limit",
            "ALTER TABLE tokens ADD COLUMN token_limit INTEGER",
        ),
        (
            "token_limit_reset",
            "ALTER TABLE tokens ADD COLUMN token_limit_reset TEXT NOT NULL DEFAULT 'never'",
        ),
        (
            "allowed_models",
            "ALTER TABLE tokens ADD COLUMN allowed_models TEXT",
        ),
    ] {
        if !existing.contains(col) {
            conn.execute(ddl, [])?;
        }
    }

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS request_logs (
            id                 TEXT PRIMARY KEY,
            created_at         INTEGER NOT NULL,
            token_id           TEXT NOT NULL,
            model              TEXT NOT NULL,
            backend            TEXT,
            upstream_model     TEXT,
            path               TEXT NOT NULL,
            status             INTEGER,
            streamed           INTEGER NOT NULL DEFAULT 0,
            ttfb_ms            INTEGER,
            duration_ms        INTEGER,
            prompt_tokens      INTEGER,
            completion_tokens  INTEGER,
            cached_tokens      INTEGER,
            cache_write_tokens INTEGER,
            total_tokens       INTEGER,
            cost               REAL,
            finish_reason      TEXT,
            error              TEXT,
            service_tier       TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_logs_token_time ON request_logs(token_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_logs_created    ON request_logs(created_at);
        CREATE INDEX IF NOT EXISTS idx_logs_model_time ON request_logs(model, created_at);
        CREATE INDEX IF NOT EXISTS idx_logs_backend_time ON request_logs(backend, created_at);",
    )?;

    // The routing catalog. Empty after this migration; `catalog::
    // seed_from_config` imports the config file's sections on first start.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS exclusive_groups (
            id               TEXT PRIMARY KEY,
            min_hold_ms      INTEGER NOT NULL,
            max_hold_ms      INTEGER NOT NULL,
            drain_timeout_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS backends (
            name               TEXT PRIMARY KEY,
            kinds              TEXT NOT NULL,
            base_url           TEXT NOT NULL,
            api_key            TEXT,
            api_key_env        TEXT,
            anthropic_version  TEXT,
            inject_usage       INTEGER NOT NULL DEFAULT 1,
            scheduling_class   TEXT NOT NULL DEFAULT 'unrestricted',
            exclusive_group_id TEXT,
            max_concurrency    INTEGER,
            max_tokens_cap     INTEGER,
            tok_s_estimate     REAL
        );
        CREATE TABLE IF NOT EXISTS models (
            name                    TEXT PRIMARY KEY,
            aliases                 TEXT,
            pricing                 TEXT,
            qos_max_wait_ms         INTEGER,
            qos_background_max_wait_ms INTEGER,
            on_shed_without_fallback   TEXT NOT NULL DEFAULT 'wait'
        );
        CREATE TABLE IF NOT EXISTS model_targets (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            model         TEXT NOT NULL,
            position      INTEGER NOT NULL,
            backend       TEXT NOT NULL,
            rename        TEXT,
            weight        INTEGER NOT NULL DEFAULT 1,
            pricing       TEXT,
            fallback_only INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_targets_model   ON model_targets(model, position);
        CREATE INDEX IF NOT EXISTS idx_targets_backend ON model_targets(backend);",
    )?;

    // v3 databases already have a `models` table without the aliases column.
    if !table_columns(conn, "models")?.contains("aliases") {
        conn.execute("ALTER TABLE models ADD COLUMN aliases TEXT", [])?;
    }

    // v3–v5 databases have the catalog tables without the scheduling
    // columns. Every added column is either nullable or carries a default
    // that reproduces the pre-scheduling behavior, so existing rows keep
    // working untouched.
    let existing = table_columns(conn, "backends")?;
    for (col, ddl) in [
        (
            "scheduling_class",
            "ALTER TABLE backends ADD COLUMN scheduling_class TEXT NOT NULL DEFAULT 'unrestricted'",
        ),
        (
            "exclusive_group_id",
            "ALTER TABLE backends ADD COLUMN exclusive_group_id TEXT",
        ),
        (
            "max_concurrency",
            "ALTER TABLE backends ADD COLUMN max_concurrency INTEGER",
        ),
        (
            "max_tokens_cap",
            "ALTER TABLE backends ADD COLUMN max_tokens_cap INTEGER",
        ),
        (
            "tok_s_estimate",
            "ALTER TABLE backends ADD COLUMN tok_s_estimate REAL",
        ),
    ] {
        if !existing.contains(col) {
            conn.execute(ddl, [])?;
        }
    }

    let existing = table_columns(conn, "models")?;
    for (col, ddl) in [
        (
            "qos_max_wait_ms",
            "ALTER TABLE models ADD COLUMN qos_max_wait_ms INTEGER",
        ),
        (
            "qos_background_max_wait_ms",
            "ALTER TABLE models ADD COLUMN qos_background_max_wait_ms INTEGER",
        ),
        (
            "on_shed_without_fallback",
            "ALTER TABLE models ADD COLUMN on_shed_without_fallback TEXT NOT NULL DEFAULT 'wait'",
        ),
    ] {
        if !existing.contains(col) {
            conn.execute(ddl, [])?;
        }
    }

    if !table_columns(conn, "model_targets")?.contains("fallback_only") {
        conn.execute(
            "ALTER TABLE model_targets ADD COLUMN fallback_only INTEGER NOT NULL DEFAULT 0",
            [],
        )?;
    }

    // v2–v6 databases have request_logs without the service tier. Nullable
    // with no default, so rows written before the column existed read back
    // as "no tier stated", which is exactly what they were.
    if !table_columns(conn, "request_logs")?.contains("service_tier") {
        conn.execute("ALTER TABLE request_logs ADD COLUMN service_tier TEXT", [])?;
    }

    conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    Ok(())
}

fn table_columns(conn: &Connection, table: &str) -> Result<std::collections::HashSet<String>> {
    let mut stmt = conn.prepare("SELECT name FROM pragma_table_info(?1)")?;
    let cols = stmt
        .query_map([table], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<_, _>>()?;
    Ok(cols)
}

/// Returns `Ok(None)` if no token with the given id exists.
fn lookup_hash(conn: &Connection, id: &str) -> Result<Option<[u8; 32]>> {
    let hash: Option<String> = conn
        .query_row("SELECT token_hash FROM tokens WHERE id = ?1", [id], |row| {
            row.get(0)
        })
        .map(Some)
        .or_else(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => Ok(None),
            e => Err(e),
        })?;
    hash.map(|h| decode_hash(&h)).transpose()
}

fn decode_hash(hash: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(hash).context("corrupt token_hash in database")?;
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("corrupt token_hash length"))
}

/// Returns `Ok(None)` when no model filter is specified.
fn allowed_models_to_sql(models: &Option<Vec<String>>) -> Result<Option<String>> {
    models
        .as_ref()
        .map(|m| serde_json::to_string(m).context("serializing allowed_models"))
        .transpose()
}

fn record_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<KeyRecord> {
    record_from_offset_row(row, 0)
}

fn record_from_offset_row(row: &rusqlite::Row<'_>, offset: usize) -> rusqlite::Result<KeyRecord> {
    let parse_reset = |idx: usize, s: String| {
        ResetInterval::parse(&s).ok_or_else(|| {
            rusqlite::Error::FromSqlConversionFailure(
                idx,
                rusqlite::types::Type::Text,
                format!("invalid reset interval {s:?}").into(),
            )
        })
    };
    let allowed: Option<String> = row.get(offset + 8)?;
    let allowed_models = allowed
        .map(|json| {
            serde_json::from_str::<Vec<String>>(&json).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    offset + 8,
                    rusqlite::types::Type::Text,
                    format!("invalid allowed_models JSON: {e}").into(),
                )
            })
        })
        .transpose()?;
    Ok(KeyRecord {
        id: row.get(offset)?,
        name: row.get(offset + 1)?,
        created_at: row.get(offset + 2)?,
        disabled: row.get(offset + 3)?,
        credit_limit: row.get(offset + 4)?,
        credit_limit_reset: parse_reset(offset + 5, row.get(offset + 5)?)?,
        token_limit: row.get(offset + 6)?,
        token_limit_reset: parse_reset(offset + 7, row.get(offset + 7)?)?,
        allowed_models,
    })
}

/// In-memory map of valid token hashes to their key records; the hot path
/// never touches SQLite. Loaded fully at startup and kept in sync by the
/// admin handlers (single process, so cache and DB cannot drift).
#[derive(Default, Debug)]
pub struct TokenCache {
    map: RwLock<HashMap<[u8; 32], Arc<KeyRecord>>>,
}

impl TokenCache {
    /// Constructs a new token cache from existing entries.
    pub fn new(entries: Vec<([u8; 32], KeyRecord)>) -> Self {
        TokenCache {
            map: RwLock::new(
                entries
                    .into_iter()
                    .map(|(hash, rec)| (hash, Arc::new(rec)))
                    .collect(),
            ),
        }
    }

    /// Lookups a key record by its SHA-256 token hash.
    pub fn get(&self, hash: &[u8; 32]) -> Option<Arc<KeyRecord>> {
        self.map.read().unwrap().get(hash).cloned()
    }

    /// Inserts or replaces a key record in the cache.
    pub fn insert(&self, hash: [u8; 32], record: Arc<KeyRecord>) {
        self.map.write().unwrap().insert(hash, record);
    }

    /// Removes a token from the cache.
    pub fn remove(&self, hash: &[u8; 32]) {
        self.map.write().unwrap().remove(hash);
    }
}

/// Returns the current UTC timestamp formatted as RFC 3339.
pub fn now_rfc3339() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .expect("RFC3339 formatting cannot fail for UTC")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (tempfile::TempDir, Db) {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("t.db")).unwrap();
        (dir, db)
    }

    fn record(id: &str, name: &str) -> KeyRecord {
        KeyRecord {
            id: id.into(),
            ..KeyRecord::new(name.into())
        }
    }

    #[test]
    fn token_generation_format() {
        let t = generate_token();
        assert!(t.starts_with(TOKEN_PREFIX));
        assert_eq!(t.len(), TOKEN_PREFIX.len() + 64);
        assert_ne!(t, generate_token());
    }

    #[test]
    fn round_trip_and_hash_only_storage() {
        let (_dir, db) = temp_db();
        let token = generate_token();
        let hash = hash_token(&token);
        db.insert(&record("id1", "ci"), &hash).unwrap();

        let listed = db.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "id1");
        assert_eq!(listed[0].name, "ci");
        assert!(!listed[0].disabled);
        assert_eq!(listed[0].credit_limit, None);
        assert_eq!(listed[0].credit_limit_reset, ResetInterval::Never);

        // Plaintext token must not appear anywhere in the database.
        let conn = db.conn.lock().unwrap();
        let stored: String = conn
            .query_row("SELECT token_hash FROM tokens WHERE id = 'id1'", [], |r| {
                r.get(0)
            })
            .unwrap();
        drop(conn);
        assert_eq!(stored, hex::encode(hash));
        assert!(!stored.contains(&token));

        let loaded = db.load_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, hash);
        assert_eq!(loaded[0].1.id, "id1");
    }

    #[test]
    fn limits_and_allowlist_round_trip() {
        let (_dir, db) = temp_db();
        let mut rec = record("id1", "limited");
        rec.disabled = true;
        rec.credit_limit = Some(12.5);
        rec.credit_limit_reset = ResetInterval::Monthly;
        rec.token_limit = Some(1_000_000);
        rec.token_limit_reset = ResetInterval::Daily;
        rec.allowed_models = Some(vec!["gpt-4o".into(), "claude-sonnet".into()]);
        db.insert(&rec, &hash_token("tok")).unwrap();

        let got = db.get("id1").unwrap().unwrap();
        assert!(got.disabled);
        assert_eq!(got.credit_limit, Some(12.5));
        assert_eq!(got.credit_limit_reset, ResetInterval::Monthly);
        assert_eq!(got.token_limit, Some(1_000_000));
        assert_eq!(got.token_limit_reset, ResetInterval::Daily);
        assert_eq!(
            got.allowed_models,
            Some(vec!["gpt-4o".to_string(), "claude-sonnet".to_string()])
        );
        assert!(got.allows_model("gpt-4o"));
        assert!(!got.allows_model("o3"));
        assert!(db.get("missing").unwrap().is_none());
    }

    #[test]
    fn update_overwrites_and_returns_hash() {
        let (_dir, db) = temp_db();
        let hash = hash_token("tok");
        db.insert(&record("id1", "before"), &hash).unwrap();

        let mut rec = db.get("id1").unwrap().unwrap();
        rec.name = "after".into();
        rec.disabled = true;
        rec.credit_limit = Some(5.0);
        assert_eq!(db.update(&rec).unwrap(), Some(hash));

        let got = db.get("id1").unwrap().unwrap();
        assert_eq!(got.name, "after");
        assert!(got.disabled);
        assert_eq!(got.credit_limit, Some(5.0));

        let missing = record("nope", "x");
        assert_eq!(db.update(&missing).unwrap(), None);
    }

    #[test]
    fn delete_returns_hash_and_removes() {
        let (_dir, db) = temp_db();
        let hash = hash_token("tok");
        db.insert(&record("id1", "a"), &hash).unwrap();
        assert_eq!(db.delete("id1").unwrap(), Some(hash));
        assert_eq!(db.delete("id1").unwrap(), None);
        assert!(db.list().unwrap().is_empty());
    }

    #[test]
    fn cache_operations() {
        let hash = hash_token("tok");
        let cache = TokenCache::new(vec![(hash, record("id1", "a"))]);
        assert_eq!(cache.get(&hash).unwrap().id, "id1");
        cache.remove(&hash);
        assert!(cache.get(&hash).is_none());
        cache.insert(hash, Arc::new(record("id1", "a")));
        assert!(cache.get(&hash).is_some());
    }

    /// A v6 database predates `request_logs.service_tier`; the column has
    /// to appear on open, with rows written before it reading back as
    /// "no tier stated".
    #[test]
    fn migrates_a_v6_database_to_the_service_tier_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        {
            // Full current schema, then the column dropped and the version
            // wound back — a v6 database in every way that matters here.
            let db = Db::open(&path).unwrap();
            db.with_write(|conn| {
                conn.execute(
                    "INSERT INTO request_logs (id, created_at, token_id, model, path)
                     VALUES ('pre', 1, 'tok', 'm', '/v1/chat/completions')",
                    [],
                )?;
                conn.execute("ALTER TABLE request_logs DROP COLUMN service_tier", [])?;
                conn.pragma_update(None, "user_version", 6i64)?;
                Ok(())
            })
            .unwrap();
        }

        let db = Db::open(&path).unwrap();
        let (version, tier): (i64, Option<String>) = db
            .with_read(|conn| {
                Ok((
                    conn.query_row("PRAGMA user_version", [], |r| r.get(0))?,
                    conn.query_row(
                        "SELECT service_tier FROM request_logs WHERE id = 'pre'",
                        [],
                        |r| r.get(0),
                    )?,
                ))
            })
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        assert_eq!(tier, None, "a pre-v7 row must read back with no tier");
    }

    #[test]
    fn migrates_legacy_v1_database() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        // Hand-create the pre-versioning schema with one row.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE tokens (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    token_hash  TEXT NOT NULL UNIQUE,
                    created_at  TEXT NOT NULL
                );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO tokens (id, name, token_hash, created_at) VALUES (?1, ?2, ?3, ?4)",
                (
                    "old-id",
                    "old-key",
                    hex::encode(hash_token("old")),
                    "2025-01-01T00:00:00Z",
                ),
            )
            .unwrap();
        }

        let db = Db::open(&path).unwrap();
        let got = db.get("old-id").unwrap().unwrap();
        assert_eq!(got.name, "old-key");
        assert!(!got.disabled);
        assert_eq!(got.credit_limit, None);
        assert_eq!(got.credit_limit_reset, ResetInterval::Never);
        assert_eq!(got.allowed_models, None);

        let version: i64 = db
            .with_write(|conn| Ok(conn.query_row("PRAGMA user_version", [], |r| r.get(0))?))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);

        // request_logs exists and is queryable after migration.
        let count: i64 = db
            .with_read(|conn| {
                Ok(conn.query_row("SELECT COUNT(*) FROM request_logs", [], |r| r.get(0))?)
            })
            .unwrap();
        assert_eq!(count, 0);

        // Reopening an already-migrated database is a no-op.
        drop(db);
        let db = Db::open(&path).unwrap();
        assert_eq!(db.list().unwrap().len(), 1);
    }

    #[test]
    fn migrates_v3_catalog_by_adding_model_aliases() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        // A v3 database: `models` exists without the aliases column.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE models (name TEXT PRIMARY KEY, pricing TEXT);
                 INSERT INTO models (name, pricing) VALUES ('gpt-4o', NULL);",
            )
            .unwrap();
            conn.pragma_update(None, "user_version", 3).unwrap();
        }

        let db = Db::open(&path).unwrap();
        let aliases: Option<String> = db
            .with_read(|conn| {
                Ok(conn.query_row(
                    "SELECT aliases FROM models WHERE name = 'gpt-4o'",
                    [],
                    |r| r.get(0),
                )?)
            })
            .unwrap();
        assert_eq!(aliases, None, "existing routes keep no aliases");
        let version: i64 = db
            .with_write(|conn| Ok(conn.query_row("PRAGMA user_version", [], |r| r.get(0))?))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
    }

    #[test]
    fn migrates_v5_catalog_by_adding_scheduling_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        // A v5 database: the catalog tables exist in their pre-scheduling
        // shape, with rows in them.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE backends (
                    name TEXT PRIMARY KEY, kinds TEXT NOT NULL, base_url TEXT NOT NULL,
                    api_key TEXT, api_key_env TEXT, anthropic_version TEXT,
                    inject_usage INTEGER NOT NULL DEFAULT 1);
                 CREATE TABLE models (name TEXT PRIMARY KEY, aliases TEXT, pricing TEXT);
                 CREATE TABLE model_targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT NOT NULL,
                    position INTEGER NOT NULL, backend TEXT NOT NULL, rename TEXT,
                    weight INTEGER NOT NULL DEFAULT 1, pricing TEXT);
                 INSERT INTO backends (name, kinds, base_url)
                   VALUES ('openai', '[\"openai\"]', 'https://api.openai.com');
                 INSERT INTO models (name) VALUES ('gpt-4o');
                 INSERT INTO model_targets (model, position, backend)
                   VALUES ('gpt-4o', 0, 'openai');",
            )
            .unwrap();
            conn.pragma_update(None, "user_version", 5).unwrap();
        }

        let db = Db::open(&path).unwrap();
        let version: i64 = db
            .with_write(|conn| Ok(conn.query_row("PRAGMA user_version", [], |r| r.get(0))?))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);

        // Existing rows keep working and pick up defaults that reproduce
        // the pre-scheduling behavior: unscheduled, uncapped, no fallback.
        let (class, group, cap): (String, Option<String>, Option<i64>) = db
            .with_read(|conn| {
                Ok(conn.query_row(
                    "SELECT scheduling_class, exclusive_group_id, max_tokens_cap
                     FROM backends WHERE name = 'openai'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )?)
            })
            .unwrap();
        assert_eq!(class, "unrestricted");
        assert_eq!(group, None);
        assert_eq!(cap, None);

        let (wait, on_shed): (Option<i64>, String) = db
            .with_read(|conn| {
                Ok(conn.query_row(
                    "SELECT qos_max_wait_ms, on_shed_without_fallback
                     FROM models WHERE name = 'gpt-4o'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )?)
            })
            .unwrap();
        assert_eq!(wait, None);
        assert_eq!(on_shed, "wait");

        let fallback_only: bool = db
            .with_read(|conn| {
                Ok(conn.query_row(
                    "SELECT fallback_only FROM model_targets WHERE model = 'gpt-4o'",
                    [],
                    |r| r.get(0),
                )?)
            })
            .unwrap();
        assert!(!fallback_only);

        // The new group table exists and starts empty.
        let groups: i64 = db
            .with_read(|conn| {
                Ok(conn.query_row("SELECT COUNT(*) FROM exclusive_groups", [], |r| r.get(0))?)
            })
            .unwrap();
        assert_eq!(groups, 0);
    }
}
