import dbData from '../../../db.json';

export interface Tool {
  name: string;
  slug: string;
  tagline: string;
  description: string;
  category: string;
  sub_category: string;
  logo_url: string;
  website: string;
  repo_url: string;
  license: string;
  license_type: string;
  language: string[];
  framework: string[];
  api_type: string[];
  min_ram_mb: number;
  min_cpu_cores: number;
  scaling_pattern: string;
  data_model: string[];
  protocols: string[];
  deployment_methods: string[];
  self_hostable: boolean;
  k8s_native: boolean;
  offline_capable: boolean;
  integrates_with: string[];
  complements: string[];
  replaces: string[];
  similar_to: string[];
  conflicts_with: string[];
  sdk_languages: string[];
  plugin_ecosystem: string;
  github_stars: number;
  contributors_count: number;
  commit_frequency: string;
  first_release_year: number;
  latest_version: string;
  last_release_date: string;
  backing_org: string;
  funding_model: string;
  docs_quality: string;
  maturity: string;
  complexity_level: string;
  team_size_fit: string[];
  industry_verticals: string[];
  performance_tier: string;
  vendor_lockin_risk: string;
  pricing_model: string;
  tags: string[];
  problem_domains: string[];
  use_cases_detailed: string[];
  anti_patterns: string[];
  stack_layer: string[];
}

export const tools: Tool[] = dbData as Tool[];

export function getToolBySlug(slug: string): Tool | undefined {
  return tools.find((t) => t.slug === slug);
}

export function getCategories(): string[] {
  return [...new Set(tools.map((t) => t.category))].sort();
}

export function getToolsByCategory(category: string): Tool[] {
  return tools.filter((t) => t.category === category);
}

export function getSubCategories(category: string): string[] {
  return [
    ...new Set(
      tools.filter((t) => t.category === category).map((t) => t.sub_category)
    ),
  ].sort();
}

export interface ComparisonPair {
  slugA: string;
  slugB: string;
  toolA: Tool;
  toolB: Tool;
}

export function getComparisons(): ComparisonPair[] {
  const seen = new Set<string>();
  const pairs: ComparisonPair[] = [];

  for (const tool of tools) {
    for (const similarSlug of tool.similar_to ?? []) {
      const a = tool.slug < similarSlug ? tool.slug : similarSlug;
      const b = tool.slug < similarSlug ? similarSlug : tool.slug;
      const key = `${a}___${b}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const toolA = getToolBySlug(a);
      const toolB = getToolBySlug(b);
      if (toolA && toolB) {
        pairs.push({ slugA: a, slugB: b, toolA, toolB });
      }
    }
  }

  return pairs;
}

export function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export function formatRam(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(0)} GB`;
  return `${mb} MB`;
}

export const CATEGORY_COLORS: Record<string, string> = {
  'AI / ML': '#7c3aed',
  'LLMs & AI Infra': '#6d28d9',
  'Databases': '#0369a1',
  'DevOps & Infra': '#b45309',
  'Monitoring': '#065f46',
  'CRM & ERP': '#1e40af',
  'Communication': '#0f766e',
  'Project Mgmt': '#7e22ce',
  'Knowledge & Docs': '#15803d',
  'Automation': '#b91c1c',
  'Low-Code': '#c2410c',
  'Analytics': '#1d4ed8',
  'Email Marketing': '#be185d',
  'Security & Auth': '#dc2626',
  'Web & CMS': '#0891b2',
  'Media & Files': '#d97706',
  'Scheduling': '#6d28d9',
  'Dev Tools': '#0d9488',
  'DNS & Networking': '#374151',
  'Embeddable': '#9333ea',
};

export function getCategoryColor(category: string): string {
  return CATEGORY_COLORS[category] ?? '#4b5563';
}
