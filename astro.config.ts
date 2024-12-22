import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import react from "@astrojs/react";
import remarkToc from "remark-toc";
import remarkCollapse from "remark-collapse";
import sitemap from "@astrojs/sitemap";
import { SITE } from "./src/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";


// https://astro.build/config
export default defineConfig({
  site: SITE.website,
  integrations: [
    tailwind({
      applyBaseStyles: false,
    }),
    react(),
    sitemap(),
  ],
  markdown: {
    remarkPlugins: [
      remarkMath,
      remarkToc,
      [
        remarkCollapse,
        {
          test: "Table of contents",
        },
      ],
    ],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      // For more themes, visit https://shiki.style/themes
      themes: { light: "min-light", dark: "night-owl" },
      wrap: true,
    },
  },
  vite: {
    optimizeDeps: {
      exclude: ["@resvg/resvg-js"],
    },
  },
  scopedStyleStrategy: "where",
  experimental: {
    contentLayer: true,
  },
  redirects: {
    '/cinder-experiments/rendering-julia-set-fractal/main.html': '/posts/rendering-julia-set-fractal/',
    '/random-blogs/building-an-autograd-library-from-scratch-in-c-for-simple-neural-networks/main.html': '/posts/building-an-autograd-library-from-scratch-in-c-for-simple-neural-networks/',
    '/random-blogs/optimizing-cpu-matrix-multiplication/main.html': '/posts/optimizing-cpu-matrix-multiplication/',
    '/cinder-experiments/simulating_fluid/main.html': '/posts/simulating-fluid/',
    '/cinder-experiments/building_a_dynamic_particle_system/main.html': '/posts/building-a-dynamic-particle-system/',
  }
});
