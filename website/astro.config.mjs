import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://ossneuralmatch.com',
  integrations: [sitemap()],
  output: 'static',
});
