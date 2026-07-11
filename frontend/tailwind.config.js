/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Outfit', 'sans-serif'],
      },
      colors: {
        brand: {
          mint: '#F0FDF4',   // background mint
          teal: '#0D9488',   // primary teal
          green: '#22C55E',  // accent green
        }
      }
    },
  },
  plugins: [],
}
