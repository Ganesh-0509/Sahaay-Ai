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
        'theme-bg': '#1A103C',
        'theme-panel': 'rgba(45, 33, 89, 0.85)',
        'theme-hover': '#3E2D7A',

        'theme-primary': '#8A6CFF',
        'theme-primary-light': '#A187FF',
        'theme-primary-dark': '#6A4FE0',

        'theme-secondary': '#C576FF',
        'theme-secondary-light': '#D395FF',
        'theme-secondary-dark': '#A85FE0',

        'theme-accent': '#FFD166',
        'theme-success': '#7BC67E',
        'theme-warning': '#F9C74F',
        'theme-danger': '#FF5C8D',
        'theme-pulse': '#FF5C8D',

        'theme-text-main': '#F0EBFF',
        'theme-text-subtle': '#A99BDB',
        'theme-text-muted': '#7F73B0',

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
        'primary-glow': '0 0 25px 0 rgba(138, 108, 255, 0.5)',
        'secondary-glow': '0 0 30px 0 rgba(197, 118, 255, 0.4)',
        'card': '0 8px 20px rgba(0, 0, 0, 0.25)',
        'card-hover': '0 15px 40px rgba(138, 108, 255, 0.35)',
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
