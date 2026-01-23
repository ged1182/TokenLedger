import React, { useState, useEffect } from 'react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import {
  DollarSign,
  TrendingUp,
  Zap,
  Users,
  AlertTriangle,
  Clock,
  RefreshCw,
  Settings,
  ChevronDown,
} from 'lucide-react';

// API base URL - configure for your setup
const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';

// Color palette
const COLORS = ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e'];
const MODEL_COLORS = {
  // OpenAI - Greens
  'gpt-4.5-preview': '#047857',
  'gpt-4o': '#10b981',
  'gpt-4o-mini': '#34d399',
  'o1': '#065f46',
  'o1-mini': '#059669',
  'o3-mini': '#6ee7b7',
  // Anthropic - Purples
  'claude-opus-4': '#7c3aed',
  'claude-sonnet-4': '#8b5cf6',
  'claude-3.5-sonnet': '#a855f7',
  'claude-3.5-haiku': '#c084fc',
  'claude-3-opus': '#6d28d9',
  'claude-3-haiku': '#ddd6fe',
  // Google - Blues
  'gemini-2.0-flash': '#3b82f6',
  'gemini-2.0-pro': '#1d4ed8',
  'gemini-1.5-pro': '#2563eb',
  'gemini-1.5-flash': '#60a5fa',
};

// Format currency
const formatCurrency = (value) => {
  if (value >= 1000) return `$${(value / 1000).toFixed(1)}k`;
  if (value >= 1) return `$${value.toFixed(2)}`;
  return `$${value.toFixed(4)}`;
};

// Format large numbers
const formatNumber = (value) => {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return value.toString();
};

// Stat Card Component
function StatCard({ title, value, subtitle, icon: Icon, trend, color = 'indigo' }) {
  const colorClasses = {
    indigo: 'bg-indigo-50 text-indigo-600',
    green: 'bg-green-50 text-green-600',
    purple: 'bg-purple-50 text-purple-600',
    amber: 'bg-amber-50 text-amber-600',
    rose: 'bg-rose-50 text-rose-600',
  };

  return (
    <div className="stat-card">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-slate-500">{title}</p>
          <p className="mt-1 text-2xl font-semibold text-slate-900">{value}</p>
          {subtitle && (
            <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
          )}
          {trend !== undefined && (
            <div className={`mt-2 flex items-center text-sm ${trend >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              <TrendingUp className={`h-4 w-4 mr-1 ${trend < 0 ? 'rotate-180' : ''}`} />
              {Math.abs(trend).toFixed(1)}% vs last period
            </div>
          )}
        </div>
        <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
    </div>
  );
}

// Main Dashboard Component
export default function App() {
  const [days, setDays] = useState(30);
  const [summary, setSummary] = useState(null);
  const [dailyCosts, setDailyCosts] = useState([]);
  const [modelCosts, setModelCosts] = useState([]);
  const [userCosts, setUserCosts] = useState([]);
  const [errorStats, setErrorStats] = useState(null);
  const [latencyStats, setLatencyStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters
  const [selectedProvider, setSelectedProvider] = useState('all');
  const [selectedModel, setSelectedModel] = useState('all');
  const [selectedUser, setSelectedUser] = useState('all');

  // Reset model when provider changes
  const handleProviderChange = (provider) => {
    setSelectedProvider(provider);
    setSelectedModel('all'); // Reset model since it may not belong to new provider
  };

  // Available filter options (populated from data)
  const [providers, setProviders] = useState([]);
  const [models, setModels] = useState([]);
  const [users, setUsers] = useState([]);

  // Build query string with filters
  const buildQueryString = (baseParams = {}) => {
    const params = new URLSearchParams({ days: days.toString(), ...baseParams });
    if (selectedModel !== 'all') params.append('model', selectedModel);
    if (selectedUser !== 'all') params.append('user_id', selectedUser);
    return params.toString();
  };

  // Fetch all dashboard data
  const fetchData = async () => {
    setLoading(true);
    setError(null);

    try {
      const queryString = buildQueryString();
      const [summaryRes, dailyRes, modelRes, userRes, errorRes, latencyRes] = await Promise.all([
        fetch(`${API_BASE}/summary?${queryString}`),
        fetch(`${API_BASE}/costs/daily?${buildQueryString({ user_id: selectedUser !== 'all' ? selectedUser : '' })}`),
        fetch(`${API_BASE}/costs/by-model?days=${days}`),
        fetch(`${API_BASE}/costs/by-user?days=${days}`),
        fetch(`${API_BASE}/errors?days=${Math.min(days, 7)}`),
        fetch(`${API_BASE}/latency?days=${Math.min(days, 7)}`),
      ]);

      if (!summaryRes.ok) throw new Error('Failed to fetch data');

      const summaryData = await summaryRes.json();
      const modelData = await modelRes.json();
      const userData = await userRes.json();

      setSummary(summaryData);
      setDailyCosts(await dailyRes.json());
      setModelCosts(modelData);
      setUserCosts(userData);
      setErrorStats(await errorRes.json());
      setLatencyStats(await latencyRes.json());

      // Populate filter options
      const uniqueProviders = [...new Set(modelData.map(m => m.provider))];
      const uniqueModels = modelData.map(m => ({ model: m.model, provider: m.provider }));
      const uniqueUsers = userData.map(u => u.user_id);

      setProviders(uniqueProviders);
      setModels(uniqueModels);
      setUsers(uniqueUsers);
    } catch (err) {
      setError(err.message);
      // Use demo data if API is not available
      useDemoData();
    } finally {
      setLoading(false);
    }
  };

  // Demo data for preview
  const useDemoData = () => {
    setSummary({
      total_cost: 1247.83,
      total_tokens: 45678912,
      total_input_tokens: 32456789,
      total_output_tokens: 13222123,
      total_requests: 15234,
      avg_cost_per_request: 0.082,
      avg_tokens_per_request: 2998,
      projected_monthly_cost: 1567.45,
    });

    // Generate daily costs
    const daily = [];
    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      daily.push({
        date: date.toISOString().split('T')[0],
        total_cost: 30 + Math.random() * 50,
        total_requests: 400 + Math.floor(Math.random() * 200),
        total_tokens: 1000000 + Math.floor(Math.random() * 500000),
      });
    }
    setDailyCosts(daily);

    setModelCosts([
      { model: 'gpt-4o', provider: 'openai', total_cost: 523.45, total_requests: 4521, percentage_of_total: 42 },
      { model: 'claude-3-5-sonnet-20241022', provider: 'anthropic', total_cost: 312.67, total_requests: 3102, percentage_of_total: 25 },
      { model: 'gpt-4o-mini', provider: 'openai', total_cost: 189.23, total_requests: 5432, percentage_of_total: 15 },
      { model: 'claude-3-haiku-20240307', provider: 'anthropic', total_cost: 122.48, total_requests: 2179, percentage_of_total: 10 },
      { model: 'gpt-3.5-turbo', provider: 'openai', total_cost: 100.00, total_requests: 4000, percentage_of_total: 8 },
    ]);

    setUserCosts([
      { user_id: 'user_001', total_cost: 234.56, total_requests: 2341 },
      { user_id: 'user_002', total_cost: 189.34, total_requests: 1892 },
      { user_id: 'user_003', total_cost: 156.78, total_requests: 1567 },
      { user_id: 'user_004', total_cost: 123.45, total_requests: 1234 },
      { user_id: 'anonymous', total_cost: 543.70, total_requests: 8200 },
    ]);

    setErrorStats({
      total_requests: 15234,
      errors: 127,
      error_rate: 0.0083,
    });

    setLatencyStats({
      p50_ms: 423,
      p90_ms: 1245,
      p95_ms: 1876,
      p99_ms: 3421,
      avg_ms: 567,
    });
  };

  useEffect(() => {
    fetchData();
  }, [days, selectedProvider, selectedModel, selectedUser]);

  // Filter data based on selections
  const filteredModelCosts = modelCosts.filter(m => {
    if (selectedProvider !== 'all' && m.provider !== selectedProvider) return false;
    if (selectedModel !== 'all' && m.model !== selectedModel) return false;
    return true;
  });

  const filteredUserCosts = userCosts.filter(u => {
    if (selectedUser !== 'all' && u.user_id !== selectedUser) return false;
    return true;
  });

  // Compute filtered summary
  const filteredSummary = {
    ...summary,
    total_cost: selectedProvider === 'all' && selectedModel === 'all'
      ? summary?.total_cost || 0
      : filteredModelCosts.reduce((sum, m) => sum + m.total_cost, 0),
    total_requests: selectedProvider === 'all' && selectedModel === 'all'
      ? summary?.total_requests || 0
      : filteredModelCosts.reduce((sum, m) => sum + m.total_requests, 0),
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <RefreshCw className="h-8 w-8 animate-spin text-indigo-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="h-8 w-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Zap className="h-5 w-5 text-white" />
              </div>
              <h1 className="text-xl font-bold gradient-text">TokenLedger</h1>
            </div>

            <div className="flex items-center space-x-3">
              {/* Provider Filter */}
              <div className="relative">
                <select
                  value={selectedProvider}
                  onChange={(e) => handleProviderChange(e.target.value)}
                  className="appearance-none bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 pr-8 text-sm font-medium text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="all">All Providers</option>
                  {providers.map(p => (
                    <option key={p} value={p}>{p.charAt(0).toUpperCase() + p.slice(1)}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none" />
              </div>

              {/* Model Filter */}
              <div className="relative">
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="appearance-none bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 pr-8 text-sm font-medium text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 max-w-[160px]"
                >
                  <option value="all">All Models</option>
                  {models
                    .filter(m => selectedProvider === 'all' || m.provider === selectedProvider)
                    .map(m => (
                      <option key={m.model} value={m.model}>{m.model}</option>
                    ))}
                </select>
                <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none" />
              </div>

              {/* User Filter */}
              <div className="relative">
                <select
                  value={selectedUser}
                  onChange={(e) => setSelectedUser(e.target.value)}
                  className="appearance-none bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 pr-8 text-sm font-medium text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="all">All Users</option>
                  {users.map(u => (
                    <option key={u} value={u}>{u}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none" />
              </div>

              {/* Time Range Selector */}
              <div className="relative">
                <select
                  value={days}
                  onChange={(e) => setDays(Number(e.target.value))}
                  className="appearance-none bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 pr-8 text-sm font-medium text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value={7}>7 days</option>
                  <option value={14}>14 days</option>
                  <option value={30}>30 days</option>
                  <option value={90}>90 days</option>
                </select>
                <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none" />
              </div>

              <button
                onClick={fetchData}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <RefreshCw className="h-5 w-5 text-slate-500" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-lg text-amber-800 text-sm">
            ⚠️ Using demo data. Connect to the API at {API_BASE} to see real metrics.
          </div>
        )}

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard
            title="Total Cost"
            value={formatCurrency(filteredSummary?.total_cost || 0)}
            subtitle={`Last ${days} days${selectedProvider !== 'all' ? ` · ${selectedProvider}` : ''}`}
            icon={DollarSign}
            color="indigo"
          />
          <StatCard
            title="Projected Monthly"
            value={formatCurrency((filteredSummary?.total_cost || 0) / days * 30)}
            subtitle="Based on current usage"
            icon={TrendingUp}
            color="purple"
          />
          <StatCard
            title="Total Requests"
            value={formatNumber(filteredSummary?.total_requests || 0)}
            subtitle={filteredSummary?.total_requests ? `${formatCurrency(filteredSummary.total_cost / filteredSummary.total_requests)} avg/request` : '-'}
            icon={Zap}
            color="green"
          />
          <StatCard
            title="Error Rate"
            value={`${((errorStats?.error_rate || 0) * 100).toFixed(2)}%`}
            subtitle={`${errorStats?.errors || 0} errors`}
            icon={AlertTriangle}
            color={errorStats?.error_rate > 0.01 ? 'rose' : 'green'}
          />
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Daily Cost Chart */}
          <div className="lg:col-span-2 card">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Daily Cost Trend</h3>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={dailyCosts}>
                  <defs>
                    <linearGradient id="costGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                    stroke="#94a3b8"
                    fontSize={12}
                  />
                  <YAxis
                    tickFormatter={(value) => `$${value}`}
                    stroke="#94a3b8"
                    fontSize={12}
                  />
                  <Tooltip
                    contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px' }}
                    formatter={(value) => [`${formatCurrency(value)}`, 'Cost']}
                    labelFormatter={(date) => new Date(date).toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
                  />
                  <Area
                    type="monotone"
                    dataKey="total_cost"
                    stroke="#6366f1"
                    strokeWidth={2}
                    fill="url(#costGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Model Distribution */}
          <div className="card">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Cost by Model</h3>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={filteredModelCosts}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={2}
                    dataKey="total_cost"
                    nameKey="model"
                  >
                    {filteredModelCosts.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={MODEL_COLORS[entry.model] || COLORS[index % COLORS.length]}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value) => formatCurrency(value)}
                    contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            {/* Legend */}
            <div className="mt-4 space-y-2">
              {filteredModelCosts.slice(0, 4).map((model, index) => (
                <div key={model.model} className="flex items-center justify-between text-sm">
                  <div className="flex items-center">
                    <div
                      className="w-3 h-3 rounded-full mr-2"
                      style={{ backgroundColor: MODEL_COLORS[model.model] || COLORS[index % COLORS.length] }}
                    />
                    <span className="text-slate-600 truncate max-w-[120px]" title={model.model}>
                      {model.model.replace(/-\d{8}$/, '')}
                    </span>
                  </div>
                  <span className="font-medium text-slate-900">{formatCurrency(model.total_cost)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Top Users */}
          <div className="card">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Top Users by Cost</h3>
            <div className="space-y-3">
              {filteredUserCosts.slice(0, 5).map((user, index) => {
                const maxCost = filteredUserCosts[0]?.total_cost || 1;
                const percentage = (user.total_cost / maxCost) * 100;

                return (
                  <div key={user.user_id}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center">
                        <div className="w-6 h-6 rounded-full bg-slate-100 flex items-center justify-center text-xs font-medium text-slate-600 mr-2">
                          {index + 1}
                        </div>
                        <span className="text-sm font-medium text-slate-700">{user.user_id}</span>
                      </div>
                      <span className="text-sm font-semibold text-slate-900">
                        {formatCurrency(user.total_cost)}
                      </span>
                    </div>
                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Latency Stats */}
          <div className="card">
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Response Latency</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={[
                    { name: 'P50', value: latencyStats?.p50_ms || 0 },
                    { name: 'P90', value: latencyStats?.p90_ms || 0 },
                    { name: 'P95', value: latencyStats?.p95_ms || 0 },
                    { name: 'P99', value: latencyStats?.p99_ms || 0 },
                  ]}
                  layout="vertical"
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                  <XAxis type="number" tickFormatter={(v) => `${v}ms`} stroke="#94a3b8" fontSize={12} />
                  <YAxis type="category" dataKey="name" stroke="#94a3b8" fontSize={12} width={40} />
                  <Tooltip
                    formatter={(value) => [`${value.toFixed(0)}ms`, 'Latency']}
                    contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px' }}
                  />
                  <Bar dataKey="value" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 flex items-center justify-center text-sm text-slate-500">
              <Clock className="h-4 w-4 mr-1" />
              Average: {latencyStats?.avg_ms?.toFixed(0) || 0}ms
            </div>
          </div>
        </div>

        {/* Token Usage Summary */}
        <div className="mt-6 card">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">Token Usage Breakdown</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-slate-50 rounded-lg">
              <p className="text-3xl font-bold text-slate-900">{formatNumber(summary?.total_tokens || 0)}</p>
              <p className="text-sm text-slate-500 mt-1">Total Tokens</p>
            </div>
            <div className="text-center p-4 bg-indigo-50 rounded-lg">
              <p className="text-3xl font-bold text-indigo-600">{formatNumber(summary?.total_input_tokens || 0)}</p>
              <p className="text-sm text-slate-500 mt-1">Input Tokens</p>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <p className="text-3xl font-bold text-purple-600">{formatNumber(summary?.total_output_tokens || 0)}</p>
              <p className="text-sm text-slate-500 mt-1">Output Tokens</p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 mt-12 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-slate-500">
          TokenLedger — Open source LLM cost analytics for Postgres
        </div>
      </footer>
    </div>
  );
}
