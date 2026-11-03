import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        navy: "#0b1f3a",
        ink: "#172033",
        canvas: "#f5f7fb",
        line: "#e5eaf1",
      },
      boxShadow: {
        card: "0 8px 24px rgba(15, 35, 65, 0.05)",
      },
    },
  },
  plugins: [],
};

export default config;
