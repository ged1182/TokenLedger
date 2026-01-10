-- TokenLedger Initial Migration
-- Creates the events table with all necessary indexes

-- Create the events table
CREATE TABLE IF NOT EXISTS token_ledger_events (
    -- Identifiers
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID,
    span_id UUID,
    parent_span_id UUID,
    
    -- Timing
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duration_ms DOUBLE PRECISION,
    
    -- Provider & Model
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    
    -- Token counts
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER DEFAULT 0,
    
    -- Cost
    cost_usd DECIMAL(12, 8),
    
    -- Request details
    endpoint VARCHAR(255),
    request_type VARCHAR(50) DEFAULT 'chat',
    
    -- User & context
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    organization_id VARCHAR(255),
    
    -- Application context
    app_name VARCHAR(100),
    environment VARCHAR(50),
    
    -- Status
    status VARCHAR(20) DEFAULT 'success',
    error_type VARCHAR(100),
    error_message TEXT,
    
    -- Custom metadata (JSONB for flexible querying)
    metadata JSONB,
    
    -- Request/Response previews (for debugging)
    request_preview TEXT,
    response_preview TEXT
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_token_ledger_timestamp 
    ON token_ledger_events (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_token_ledger_user 
    ON token_ledger_events (user_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_token_ledger_model 
    ON token_ledger_events (model, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_token_ledger_app 
    ON token_ledger_events (app_name, environment, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_token_ledger_trace 
    ON token_ledger_events (trace_id);

CREATE INDEX IF NOT EXISTS idx_token_ledger_status 
    ON token_ledger_events (status, timestamp DESC);

-- Composite index for common dashboard queries
CREATE INDEX IF NOT EXISTS idx_token_ledger_dashboard 
    ON token_ledger_events (timestamp DESC, model, user_id) 
    INCLUDE (cost_usd, total_tokens);

-- GIN index for metadata queries
CREATE INDEX IF NOT EXISTS idx_token_ledger_metadata 
    ON token_ledger_events USING GIN (metadata);


-- Helper views for common queries

-- Daily cost summary view
CREATE OR REPLACE VIEW token_ledger_daily_costs AS
SELECT
    DATE(timestamp) as date,
    provider,
    model,
    COUNT(*) as request_count,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    SUM(total_tokens) as total_tokens,
    SUM(cost_usd) as total_cost,
    AVG(duration_ms) as avg_latency_ms
FROM token_ledger_events
GROUP BY DATE(timestamp), provider, model;


-- User cost summary view
CREATE OR REPLACE VIEW token_ledger_user_costs AS
SELECT
    COALESCE(user_id, 'anonymous') as user_id,
    COUNT(*) as request_count,
    SUM(total_tokens) as total_tokens,
    SUM(cost_usd) as total_cost,
    MIN(timestamp) as first_request,
    MAX(timestamp) as last_request
FROM token_ledger_events
GROUP BY user_id;
