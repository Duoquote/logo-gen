/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Geist", "system-ui", "sans-serif"],
        body: ["DM Sans", "system-ui", "sans-serif"],
        mono: ["Geist Mono", "ui-monospace", "monospace"],
      },
      colors: {
        surface: {
          1000: "#05060a",
          950: "#0a0c12",
          900: "#10121a",
          850: "#161924",
          800: "#1c2030",
          750: "#23283a",
          700: "#2c3245",
          600: "#3b4258",
          500: "#565d76",
          400: "#878fa8",
          300: "#b1b8cd",
          200: "#d6dae6",
          100: "#eef0f7",
          50: "#f8f9fc",
        },
        glass: {
          DEFAULT: "rgba(22, 25, 36, 0.55)",
          strong: "rgba(22, 25, 36, 0.72)",
          soft: "rgba(22, 25, 36, 0.38)",
          border: "rgba(255, 255, 255, 0.06)",
          "border-strong": "rgba(255, 255, 255, 0.10)",
        },
        accent: {
          DEFAULT: "#7cc7ff",
          hover: "#a3d6ff",
          muted: "rgba(124, 199, 255, 0.14)",
          dim: "#4d8cc4",
          ink: "#06121f",
        },
        glow: {
          cyan: "#3aa9ff",
          violet: "#8a6dff",
          pink: "#e36ad6",
        },
      },
      borderRadius: {
        panel: "18px",
        field: "10px",
        pill: "9999px",
      },
      boxShadow: {
        glass:
          "0 1px 0 0 rgba(255,255,255,0.06) inset, 0 18px 60px -20px rgba(0,0,0,0.55), 0 2px 6px -1px rgba(0,0,0,0.4)",
        "glass-sm":
          "0 1px 0 0 rgba(255,255,255,0.05) inset, 0 6px 24px -8px rgba(0,0,0,0.5)",
        glow:
          "0 0 0 1px rgba(124,199,255,0.35), 0 0 28px -4px rgba(124,199,255,0.35)",
      },
      backdropBlur: {
        panel: "18px",
        rail: "24px",
        modal: "28px",
      },
      animation: {
        "pulse-slow": "pulse 3s ease-in-out infinite",
        "fade-in": "fadeIn 0.3s ease-out",
        "slide-up": "slideUp 0.3s ease-out",
        "palette-in": "paletteIn 0.18s ease-out",
      },
      keyframes: {
        fadeIn: {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        slideUp: {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        paletteIn: {
          from: { opacity: "0", transform: "translate(-50%, -52%) scale(0.98)" },
          to: { opacity: "1", transform: "translate(-50%, -50%) scale(1)" },
        },
      },
    },
  },
  plugins: [require("@tailwindcss/forms")],
};
