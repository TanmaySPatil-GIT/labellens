import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

import en from './locales/en.json';
import hi from './locales/hi.json';
import mr from './locales/mr.json';

// Supported language codes
export const SUPPORTED_LANGS = ['en', 'hi', 'mr'] as const;
export type LangCode = typeof SUPPORTED_LANGS[number];

// Persist user preference in localStorage so it survives refresh
const STORAGE_KEY = 'labellens_lang';
const storedLang = (localStorage.getItem(STORAGE_KEY) as LangCode) || 'en';

i18n
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      hi: { translation: hi },
      mr: { translation: mr },
    },
    lng: storedLang,
    fallbackLng: 'en',
    interpolation: {
      // React already handles XSS escaping
      escapeValue: false,
    },
  });

/** Change language and persist choice to localStorage. */
export function setLanguage(lang: LangCode): void {
  localStorage.setItem(STORAGE_KEY, lang);
  i18n.changeLanguage(lang);
}

export default i18n;
