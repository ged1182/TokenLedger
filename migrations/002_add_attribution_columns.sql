-- TokenLedger Migration 002: Add Attribution Columns
-- Adds attribution context columns to support feature, team, and cost center tracking

-- Add attribution columns to support AttributionContext fields
ALTER TABLE token_ledger_events ADD COLUMN IF NOT EXISTS feature VARCHAR(100);
ALTER TABLE token_ledger_events ADD COLUMN IF NOT EXISTS page VARCHAR(255);
ALTER TABLE token_ledger_events ADD COLUMN IF NOT EXISTS component VARCHAR(100);
ALTER TABLE token_ledger_events ADD COLUMN IF NOT EXISTS team VARCHAR(100);
ALTER TABLE token_ledger_events ADD COLUMN IF NOT EXISTS project VARCHAR(100);
ALTER TABLE token_ledger_events ADD COLUMN IF NOT EXISTS cost_center VARCHAR(100);

-- Add indexes for common attribution queries
CREATE INDEX IF NOT EXISTS idx_token_ledger_feature
    ON token_ledger_events (feature, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_token_ledger_team
    ON token_ledger_events (team, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_token_ledger_cost_center
    ON token_ledger_events (cost_center, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_token_ledger_project
    ON token_ledger_events (project, timestamp DESC);

-- Update the user cost summary view to include team attribution
CREATE OR REPLACE VIEW token_ledger_team_costs AS
SELECT
    COALESCE(team, 'unassigned') as team,
    COUNT(*) as request_count,
    SUM(total_tokens) as total_tokens,
    SUM(cost_usd) as total_cost,
    MIN(timestamp) as first_request,
    MAX(timestamp) as last_request
FROM token_ledger_events
GROUP BY team;

-- Cost center summary view
CREATE OR REPLACE VIEW token_ledger_cost_center_costs AS
SELECT
    COALESCE(cost_center, 'unassigned') as cost_center,
    COUNT(*) as request_count,
    SUM(total_tokens) as total_tokens,
    SUM(cost_usd) as total_cost,
    MIN(timestamp) as first_request,
    MAX(timestamp) as last_request
FROM token_ledger_events
GROUP BY cost_center;
