-- =============================================================================
-- 001_pgvector.sql — Millenia Capital System: Production pgvector schema
--
-- Run once against the target database:
--   psql $DATABASE_URL -f migrations/001_pgvector.sql
--
-- Idempotent — safe to re-run.
-- =============================================================================

-- Enable pgvector extension (requires PostgreSQL 11+ with pgvector installed)
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- investors — one row per investor/partner, 4 vector columns (BGE-M3, 1024-dim)
-- =============================================================================
CREATE TABLE IF NOT EXISTS investors (
    -- Primary key matches ingestion pipeline's investor_id (e.g. "pb_inv123")
    investor_id              TEXT PRIMARY KEY,

    -- ── Identity ──────────────────────────────────────────────────────────────
    full_name                TEXT,
    firm                     TEXT,
    title                    TEXT,
    investor_type            TEXT,          -- 'individual' | 'fund' | 'family_office' | etc.
    pitchbook_id             TEXT,          -- original PitchBook identifier
    hq_location              TEXT,
    hq_state                 TEXT,

    -- ── Contact ───────────────────────────────────────────────────────────────
    email                    TEXT,
    phone                    TEXT,
    linkedin_url             TEXT,
    twitter_handle           TEXT,
    website                  TEXT,

    -- ── Fund profile ──────────────────────────────────────────────────────────
    fund_id                  TEXT,
    fund_name                TEXT,
    fund_status              TEXT NOT NULL DEFAULT 'active',  -- 'active' | 'inactive' | 'closed'
    stage_focus              TEXT[],        -- ['seed', 'series_a', ...]
    check_size_min           BIGINT,        -- USD
    check_size_max           BIGINT,        -- USD
    geography                TEXT[],        -- ['United States', 'Texas', ...]
    industry_focus           TEXT[],        -- ['Fintech', 'AI', ...]
    sector_exclusions        TEXT[],
    lead_preference          TEXT,          -- 'lead' | 'follow' | 'either'
    recent_investment_count_12m INTEGER DEFAULT 0,
    aum_millions             NUMERIC(12,2),
    dry_powder_millions      NUMERIC(12,2),
    preferred_verticals      TEXT[],
    thesis_text              TEXT,          -- full-text fund thesis (used for embedding)

    -- ── Matching metadata ─────────────────────────────────────────────────────
    investment_thesis        TEXT,          -- shorter thesis / partner-level thesis
    portfolio_companies      TEXT[],
    check_size_range         TEXT,          -- human-readable e.g. "$250K-$2M"
    data_quality_confidence  NUMERIC(4,3) DEFAULT 0.5,  -- 0.0-1.0
    source_tags              TEXT[],        -- ['pitchbook', 'apollo', ...]
    warm_intro_paths         TEXT[],
    last_verified_at         TEXT,

    -- ── Full JSON blob ────────────────────────────────────────────────────────
    -- Stores the full InvestorCandidate JSON for retrieval without re-joining
    metadata                 JSONB DEFAULT '{}',

    -- ── Embeddings (BGE-M3 1024-dim, L2-normalised) ───────────────────────────
    fund_thesis_embedding    vector(1024),  -- embed(fund.thesis_text)
    partner_bio_embedding    vector(1024),  -- embed("{name} {title} at {firm}. {investment_thesis}")
    portfolio_embedding      vector(1024),  -- embed(join(portfolio_companies))
    activity_embedding       vector(1024),  -- embed(recent signals / public content)

    -- ── Audit ─────────────────────────────────────────────────────────────────
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- HNSW indexes for approximate nearest-neighbour search (cosine distance)
--
-- HNSW is preferred over IVFFlat because:
--   - No need to pre-cluster (CLUSTER/VACUUM) before the index is useful
--   - Better recall vs IVFFlat at same ef_construction
--   - Queries don't require probes tuning
--
-- m=16, ef_construction=64 are good defaults for 50K-100K rows.
-- Increase ef_construction (e.g. 128) for higher recall at build-time cost.
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_investors_fund_thesis_hnsw
    ON investors USING hnsw (fund_thesis_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_investors_partner_bio_hnsw
    ON investors USING hnsw (partner_bio_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_investors_portfolio_hnsw
    ON investors USING hnsw (portfolio_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_investors_activity_hnsw
    ON investors USING hnsw (activity_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- GIN indexes for array containment queries (fund_status filter, stage, geo)
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_investors_stage_focus_gin
    ON investors USING gin (stage_focus);

CREATE INDEX IF NOT EXISTS idx_investors_geography_gin
    ON investors USING gin (geography);

CREATE INDEX IF NOT EXISTS idx_investors_industry_focus_gin
    ON investors USING gin (industry_focus);

CREATE INDEX IF NOT EXISTS idx_investors_source_tags_gin
    ON investors USING gin (source_tags);

-- =============================================================================
-- B-tree indexes for equality filters used in WHERE clauses
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_investors_fund_status
    ON investors (fund_status);

CREATE INDEX IF NOT EXISTS idx_investors_hq_state
    ON investors (hq_state);

CREATE INDEX IF NOT EXISTS idx_investors_pitchbook_id
    ON investors (pitchbook_id);

CREATE INDEX IF NOT EXISTS idx_investors_data_quality
    ON investors (data_quality_confidence DESC);

-- =============================================================================
-- vector_records — generic fallback table for non-investor namespaces
-- (company profiles, deal summaries, etc.)
-- =============================================================================
CREATE TABLE IF NOT EXISTS vector_records (
    namespace   TEXT NOT NULL,
    record_id   TEXT NOT NULL,
    embedding   vector(1024),
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (namespace, record_id)
);

CREATE INDEX IF NOT EXISTS idx_vector_records_embedding_hnsw
    ON vector_records USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- updated_at trigger — auto-stamp on row update
-- =============================================================================
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_investors_updated_at ON investors;
CREATE TRIGGER trg_investors_updated_at
    BEFORE UPDATE ON investors
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_vector_records_updated_at ON vector_records;
CREATE TRIGGER trg_vector_records_updated_at
    BEFORE UPDATE ON vector_records
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =============================================================================
-- Useful views
-- =============================================================================

-- Active investors with at least one embedding populated
CREATE OR REPLACE VIEW v_indexed_investors AS
SELECT
    investor_id,
    full_name,
    firm,
    title,
    fund_status,
    data_quality_confidence,
    (fund_thesis_embedding IS NOT NULL)::int +
    (partner_bio_embedding IS NOT NULL)::int +
    (portfolio_embedding IS NOT NULL)::int +
    (activity_embedding IS NOT NULL)::int AS embedded_namespace_count,
    created_at,
    updated_at
FROM investors
WHERE fund_status = 'active';

-- Embedding coverage report (useful for monitoring precompute progress)
CREATE OR REPLACE VIEW v_embedding_coverage AS
SELECT
    COUNT(*) AS total_investors,
    COUNT(fund_thesis_embedding) AS fund_thesis_embedded,
    COUNT(partner_bio_embedding) AS partner_bio_embedded,
    COUNT(portfolio_embedding) AS portfolio_embedded,
    COUNT(activity_embedding) AS activity_embedded,
    ROUND(100.0 * COUNT(fund_thesis_embedding) / NULLIF(COUNT(*), 0), 1) AS pct_fund_thesis,
    ROUND(100.0 * COUNT(partner_bio_embedding) / NULLIF(COUNT(*), 0), 1) AS pct_partner_bio,
    ROUND(100.0 * COUNT(portfolio_embedding) / NULLIF(COUNT(*), 0), 1) AS pct_portfolio,
    ROUND(100.0 * COUNT(activity_embedding) / NULLIF(COUNT(*), 0), 1) AS pct_activity
FROM investors
WHERE fund_status = 'active';
