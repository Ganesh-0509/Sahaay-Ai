import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Calming Mental Health Theme - Ocean & Sunset
        'theme-bg': '#0A1929',          // Deep Navy (peaceful night)
        'theme-bg-alt': '#132F4C',      // Lighter Navy
        'theme-panel': 'rgba(19, 47, 76, 0.85)', // Glass Panel
        'theme-hover': '#1E4976',       // Hover state

        'theme-primary': '#5BA3A3',     // Soft Teal (calm, trust)
        'theme-primary-light': '#7CC4C4', // Light Teal
        'theme-primary-dark': '#3D7A7A',  // Dark Teal

        'theme-secondary': '#FF9B85',   // Warm Coral (comfort, hope)
        'theme-secondary-light': '#FFB8A3',
        'theme-secondary-dark': '#E67A63',

        'theme-accent': '#FFD89B',      // Soft Gold (warmth)
        'theme-success': '#7BC67E',     // Soft Green (healing)
        'theme-warning': '#F9C74F',     // Warm Yellow
        'theme-danger': '#F28B82',      // Soft Red

        'theme-text-main': '#E8F4F8',   // Almost White
        'theme-text-subtle': '#B0C4DE',  // Light Steel Blue
        'theme-text-muted': '#7A96B0',   // Muted Blue

        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        '4xl': '2rem',
      },
      boxShadow: {
        'primary-glow': '0 0 25px 0 rgba(91, 163, 163, 0.4)',
        'secondary-glow': '0 0 30px 0 rgba(255, 155, 133, 0.3)',
        'card': '0 8px 20px rgba(0, 0, 0, 0.25)',
        'card-hover': '0 15px 40px rgba(91, 163, 163, 0.3)',
        'soft': '0 4px 12px rgba(0, 0, 0, 0.1)',
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 15px 0 rgba(91, 163, 163, 0.3)' },
          '50%': { boxShadow: '0 0 35px 5px rgba(91, 163, 163, 0.6)' },
        },
        animatedGradient: {
          '0%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
          '100%': { backgroundPosition: '0% 50%' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        }
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.6s ease-out forwards',
        'pulse-glow': 'pulseGlow 2.5s infinite ease-in-out',
        'animated-gradient': 'animatedGradient 20s ease infinite',
        'float': 'float 3s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
      },
      backdropBlur: {
        'xs': '2px',
      }
    },
  },
  plugins: [],
};

export default config;
