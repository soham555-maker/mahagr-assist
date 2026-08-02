import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // deck palette — navy dominant, teal accent
        navy: { DEFAULT: "#0B2545", 700: "#13315C", 600: "#16345B" },
        teal: { DEFAULT: "#1C7293", bright: "#2CA6A4" },
        ink: "#102A43",
        slate2: "#627D98",
        iceblue: "#9FB3C8",
        ice: "#EEF3F8",
        ice2: "#E1EAF3",
        line: "#D4DEEA",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "var(--font-deva)", "system-ui", "sans-serif"],
        serif: ["var(--font-serif)", "Cambria", "Georgia", "serif"],
      },
    },
  },
  plugins: [],
};
export default config;
