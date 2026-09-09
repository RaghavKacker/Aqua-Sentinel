/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ocean: {
          950: '#030a16',
          900: '#07152b',
          800: '#0c2445',
          700: '#133966',
          600: '#1b528f',
          500: '#2570be',
          400: '#3b92eb',
          300: '#6db2f2',
          200: '#a3d1f8',
          100: '#d5eafc',
        },
        sonar: {
          highlight: '#00f0ff',
          shadow: '#050c1a',
          grid: '#1a365d',
          gold: '#ffd166',
          danger: '#ff4d6d',
          success: '#06d6a0',
          warning: '#f77f00'
        }
      },
      fontFamily: {
        mono: ['Courier New', 'Consolas', 'monospace'],
      }
    },
  },
  plugins: [],
}
