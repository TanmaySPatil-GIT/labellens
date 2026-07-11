import { useState, useEffect, useRef, DragEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { setLanguage, SUPPORTED_LANGS, type LangCode } from './i18n';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';

interface HealthStatus {
  apiConnected: boolean;
  dbConnected: boolean;
  loading: boolean;
  error: string | null;
}

interface AnalysisState {
  analyzing: boolean;
  rawText: string | null;
  error: string | null;
}

interface IngredientDetails {
  ingredient: string;
  safety_status: 'safe' | 'moderate' | 'unsafe';
  reason: string;
  simple_explanation: string;
  allergen: string;
  safe_frequency: string;
  source: string;
}

interface AlternativeSuggestion {
  alternative_name: string;
  reasons: string[];
}

const LANG_LABELS: Record<LangCode, string> = {
  en: 'English',
  hi: 'हिंदी',
  mr: 'मराठी',
};

function App() {
  const { t, i18n } = useTranslation();
  const currentLang = i18n.language as LangCode;

  // System Health States
  const [health, setHealth] = useState<HealthStatus>({
    apiConnected: false,
    dbConnected: false,
    loading: true,
    error: null,
  });

  // Upload & Analysis States
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisState>({
    analyzing: false,
    rawText: null,
    error: null,
  });
  const [isDragActive, setIsDragActive] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [parsedIngredients, setParsedIngredients] = useState<IngredientDetails[]>([]);
  const [parsing, setParsing] = useState<boolean>(false);
  const [overallScore, setOverallScore] = useState<number | null>(null);
  const [scoreLabel, setScoreLabel] = useState<string>('');
  const [productGuess, setProductGuess] = useState<string>('');

  const [alternative, setAlternative] = useState<AlternativeSuggestion | null>(null);
  const [loadingAlternative, setLoadingAlternative] = useState<boolean>(false);

  // Phase 8 & 9 States
  const [activeTab, setActiveTab] = useState<'scan' | 'search' | 'history' | 'favorites'>('scan');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  const [selectedCategory, setSelectedCategory] = useState('');
  const [leaderboard, setLeaderboard] = useState<any[]>([]);
  const [loadingLeaderboard, setLoadingLeaderboard] = useState(false);
  const [leaderboardError, setLeaderboardError] = useState<string | null>(null);

  // Phase 9 Auth & Data States
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [userName, setUserName] = useState<string | null>(null);
  const [showAuthModal, setShowAuthModal] = useState<boolean>(false);
  const [authTab, setAuthTab] = useState<'login' | 'signup'>('login');
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authName, setAuthName] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);
  const [userHistory, setUserHistory] = useState<any[]>([]);
  const [userFavorites, setUserFavorites] = useState<any[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [loadingFavorites, setLoadingFavorites] = useState(false);

  // Language selector visibility
  const [langOpen, setLangOpen] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  // System Health Check
  const checkSystemHealth = async () => {
    setHealth((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const response = await fetch(`${apiUrl}/api/health`);
      if (!response.ok) {
        throw new Error(`HTTP status ${response.status}`);
      }
      const data = await response.json();
      setHealth({
        apiConnected: true,
        dbConnected: data.db_connected === true,
        loading: false,
        error: null,
      });
    } catch (err: any) {
      setHealth({
        apiConnected: false,
        dbConnected: false,
        loading: false,
        error: err.message || 'Failed to reach API server',
      });
    }
  };

  useEffect(() => {
    checkSystemHealth();
  }, []);

  // Handle File Input Selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processSelectedFile(files[0]);
    }
  };

  // Drag and Drop Handlers
  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragActive(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragActive(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      processSelectedFile(files[0]);
    }
  };

  // Validate and Stage File for Upload
  const processSelectedFile = (file: File) => {
    setAnalysis({ analyzing: false, rawText: null, error: null });

    const allowedTypes = ['image/jpeg', 'image/png'];
    if (!allowedTypes.includes(file.type)) {
      setAnalysis((prev) => ({ ...prev, error: t('errors.invalidType') }));
      setSelectedFile(null);
      setPreviewUrl(null);
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setAnalysis((prev) => ({ ...prev, error: t('errors.fileTooLarge') }));
      setSelectedFile(null);
      setPreviewUrl(null);
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  // Trigger File Input Click
  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  // Clear Staged Image
  const clearStagedImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnalysis({ analyzing: false, rawText: null, error: null });
    setParsedIngredients([]);
    setParsing(false);
    setOverallScore(null);
    setScoreLabel('');
    setProductGuess('');
    setAlternative(null);
    setLoadingAlternative(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Copy to Clipboard Utility
  const copyToClipboard = () => {
    if (analysis.rawText) {
      navigator.clipboard.writeText(analysis.rawText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // Fetch a healthier alternative suggestion after ingredients are resolved
  const fetchAlternative = async (productGuessArg: string, ingredients: IngredientDetails[]) => {
    setLoadingAlternative(true);
    try {
      const ingredientNames = ingredients.map((i) => i.ingredient);
      const response = await fetch(`${apiUrl}/api/suggest-alternative`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          product_guess: productGuessArg,
          ingredients: ingredientNames,
          language: currentLang,
        }),
      });
      const data = await response.json();
      setAlternative({ alternative_name: data.alternative_name, reasons: data.reasons || [] });
    } catch {
      setAlternative({
        alternative_name: t('alternative.loading'),
        reasons: [],
      });
    } finally {
      setLoadingAlternative(false);
    }
  };

  // Parse raw text into structured safety ingredients profiles
  const parseRawIngredients = async (rawText: string) => {
    setParsing(true);
    try {
      const response = await fetch(`${apiUrl}/api/parse-ingredients`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
        },
        body: JSON.stringify({ raw_text: rawText, language: currentLang }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to analyze ingredients safety.');
      }

      const resolvedIngredients: IngredientDetails[] = data.ingredients || [];
      const resolvedProductGuess: string = data.product_guess || '';

      setParsedIngredients(resolvedIngredients);
      setOverallScore(data.overall_score !== undefined ? data.overall_score : null);
      setScoreLabel(data.score_label || '');
      setProductGuess(resolvedProductGuess);

      // Kick off alternative suggestion as a separate async step (non-blocking)
      if (resolvedIngredients.length > 0) {
        fetchAlternative(resolvedProductGuess, resolvedIngredients);
      }

      // Refresh history if logged in
      if (authToken) {
        fetchHistory(authToken);
      }
    } catch (err: any) {
      setAnalysis((prev) => ({
        ...prev,
        error: `Vision extraction succeeded, but safety parsing failed: ${err.message || 'Failed to parse ingredients.'}`
      }));
    } finally {
      setParsing(false);
    }
  };

  // Trigger API Analysis Request
  const analyzeIngredients = async () => {
    if (!selectedFile) return;

    setAnalysis({ analyzing: true, rawText: null, error: null });
    setParsedIngredients([]);
    setParsing(false);
    setOverallScore(null);
    setScoreLabel('');
    setProductGuess('');
    setAlternative(null);
    setLoadingAlternative(false);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${apiUrl}/api/analyze-image`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'An error occurred during ingredient analysis.');
      }

      setAnalysis({
        analyzing: false,
        rawText: data.raw_text,
        error: null,
      });

      // Call the secondary parsing endpoint to assess ingredient safety
      await parseRawIngredients(data.raw_text);

    } catch (err: any) {
      setAnalysis({
        analyzing: false,
        rawText: null,
        error: err.message || 'Server connection timed out.',
      });
    }
  };

  // Phase 8 logic functions
  const searchProducts = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;

    setSearching(true);
    setSearchError(null);
    setSearchResults([]);

    try {
      const response = await fetch(`${apiUrl}/api/search-product?query=${encodeURIComponent(searchQuery.trim())}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to search products.');
      }
      setSearchResults(data.products || []);
      if ((data.products || []).length === 0) {
        setSearchError(t('search.noResults'));
      }
    } catch (err: any) {
      setSearchError(err.message || 'Error searching products.');
    } finally {
      setSearching(false);
    }
  };

  const analyzeBarcode = async (code: string, productName?: string, image?: string) => {
    // Clear previous states
    setSelectedFile(null);
    setPreviewUrl(image || null);
    setAnalysis({ analyzing: false, rawText: null, error: null });
    setParsedIngredients([]);
    setParsing(true);
    setOverallScore(null);
    setScoreLabel('');
    setProductGuess(productName || '');
    setAlternative(null);
    setLoadingAlternative(false);

    try {
      const response = await fetch(`${apiUrl}/api/analyze-barcode?code=${encodeURIComponent(code)}&language=${currentLang}`, {
        headers: {
          ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
        }
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to analyze barcode.');
      }

      const resolvedIngredients: IngredientDetails[] = data.ingredients || [];
      const resolvedProductGuess: string = data.product_guess || productName || '';

      setParsedIngredients(resolvedIngredients);
      setOverallScore(data.overall_score !== undefined ? data.overall_score : null);
      setScoreLabel(data.score_label || '');
      setProductGuess(resolvedProductGuess);
      
      if (data.image_url) {
        setPreviewUrl(data.image_url);
      }

      if (resolvedIngredients.length > 0) {
        fetchAlternative(resolvedProductGuess, resolvedIngredients);
      }

      // Refresh history if logged in
      if (authToken) {
        fetchHistory(authToken);
      }
    } catch (err: any) {
      setAnalysis({
        analyzing: false,
        rawText: `Barcode: ${code}`,
        error: `Barcode analysis failed: ${err.message || 'Could not parse ingredients.'}`
      });
    } finally {
      setParsing(false);
    }
  };

  const fetchCategoryLeaderboard = async (category: string) => {
    if (!category) {
      setLeaderboard([]);
      return;
    }
    setLoadingLeaderboard(true);
    setLeaderboardError(null);
    setLeaderboard([]);

    try {
      const response = await fetch(`${apiUrl}/api/category-best?category=${encodeURIComponent(category)}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch category rankings.');
      }
      setLeaderboard(data.products || []);
    } catch (err: any) {
      setLeaderboardError(err.message || 'Error loading leaderboard.');
    } finally {
      setLoadingLeaderboard(false);
    }
  };

  const handleCategoryChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    setSelectedCategory(val);
    fetchCategoryLeaderboard(val);
  };

  // Phase 9 Auth & Data Logic
  const fetchHistory = async (token: string) => {
    setLoadingHistory(true);
    try {
      const response = await fetch(`${apiUrl}/api/history`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await response.json();
      if (response.ok) {
        setUserHistory(data.history || []);
      }
    } catch (err) {
      console.error('Failed to fetch history', err);
    } finally {
      setLoadingHistory(false);
    }
  };

  const fetchFavorites = async (token: string) => {
    setLoadingFavorites(true);
    try {
      const response = await fetch(`${apiUrl}/api/favorites`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await response.json();
      if (response.ok) {
        setUserFavorites(data.favorites || []);
      }
    } catch (err) {
      console.error('Failed to fetch favorites', err);
    } finally {
      setLoadingFavorites(false);
    }
  };

  const addFavorite = async () => {
    if (!authToken) {
      setShowAuthModal(true);
      return;
    }
    if (!productGuess || parsedIngredients.length === 0) return;

    try {
      const response = await fetch(`${apiUrl}/api/favorites`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({
          product_name: productGuess,
          product_guess: productGuess,
          overall_score: overallScore,
          ingredients_data: parsedIngredients
        })
      });
      if (response.ok) {
        fetchFavorites(authToken);
      }
    } catch (err) {
      console.error('Failed to add favorite', err);
    }
  };

  const deleteFavorite = async (favId: number) => {
    if (!authToken) return;
    try {
      const response = await fetch(`${apiUrl}/api/favorites/${favId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${authToken}` }
      });
      if (response.ok) {
        fetchFavorites(authToken);
      }
    } catch (err) {
      console.error('Failed to delete favorite', err);
    }
  };

  const authSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!authEmail.trim() || !authPassword) {
      setAuthError('Email and password are required.');
      return;
    }
    setAuthLoading(true);
    setAuthError(null);

    const isLogin = authTab === 'login';
    const endpoint = isLogin ? '/api/auth/login' : '/api/auth/signup';
    const payload = isLogin 
      ? { email: authEmail.trim(), password: authPassword }
      : { name: authName.strip ? authName.strip() : authName, email: authEmail.trim(), password: authPassword };

    try {
      const response = await fetch(`${apiUrl}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Authentication failed.');
      }
      
      setAuthToken(data.access_token);
      setUserName(data.name);
      setShowAuthModal(false);
      
      // Clear inputs
      setAuthEmail('');
      setAuthPassword('');
      setAuthName('');
      
      // Load user data
      fetchHistory(data.access_token);
      fetchFavorites(data.access_token);
    } catch (err: any) {
      setAuthError(err.message || 'Authentication error.');
    } finally {
      setAuthLoading(false);
    }
  };

  const logout = () => {
    setAuthToken(null);
    setUserName(null);
    setUserHistory([]);
    setUserFavorites([]);
    if (activeTab === 'history' || activeTab === 'favorites') {
      setActiveTab('scan');
    }
  };

  // Handle language switch
  const handleLangChange = (lang: LangCode) => {
    setLanguage(lang);
    setLangOpen(false);
  };

  return (
    <div className="min-h-screen bg-brand-mint flex flex-col font-sans">
      {/* Navigation Header */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-teal-100 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-teal to-brand-green flex items-center justify-center text-white font-bold text-xl shadow-md shadow-teal-100">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <span className="font-extrabold text-2xl tracking-tight bg-gradient-to-r from-brand-teal to-brand-green bg-clip-text text-transparent">
                LabelLens
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {/* System Health Indicators */}
            <div className="flex items-center space-x-4 bg-emerald-50/50 border border-teal-100/60 px-4 py-1.5 rounded-full text-xs">
              <div className="flex items-center space-x-2">
                <span className="text-slate-500 font-medium">{t('nav.api')}:</span>
                {health.loading ? (
                  <span className="w-2.5 h-2.5 rounded-full bg-yellow-400 animate-pulse" />
                ) : health.apiConnected ? (
                  <span className="w-2.5 h-2.5 rounded-full bg-brand-green" />
                ) : (
                  <span className="w-2.5 h-2.5 rounded-full bg-slate-400" />
                )}
                <span className="font-semibold text-slate-700">
                  {health.loading ? t('nav.checking') : health.apiConnected ? t('nav.online') : t('nav.offline')}
                </span>
              </div>

              <div className="w-px h-3 bg-teal-200" />

              <div className="flex items-center space-x-2">
                <span className="text-slate-500 font-medium">{t('nav.db')}:</span>
                {health.loading ? (
                  <span className="w-2.5 h-2.5 rounded-full bg-yellow-400 animate-pulse" />
                ) : health.dbConnected ? (
                  <span className="w-2.5 h-2.5 rounded-full bg-brand-green" />
                ) : (
                  <span className="w-2.5 h-2.5 rounded-full bg-slate-400" />
                )}
                <span className="font-semibold text-slate-700">
                  {health.loading ? t('nav.checking') : health.dbConnected ? t('nav.connected') : t('nav.disconnected')}
                </span>
              </div>

              {!health.loading && (health.error || !health.dbConnected || !health.apiConnected) && (
                <button
                  onClick={checkSystemHealth}
                  className="ml-2 text-brand-teal hover:text-teal-700 font-bold hover:underline focus:outline-none flex items-center space-x-0.5"
                  title="Retry Health Check"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.21 8H18.5" />
                  </svg>
                </button>
              )}
            </div>

            {/* ── Language Selector ── */}
            <div className="relative">
              <button
                id="lang-selector-btn"
                onClick={() => setLangOpen((o) => !o)}
                className="flex items-center space-x-1.5 px-3 py-1.5 bg-white border border-teal-200 hover:border-brand-teal rounded-full text-xs font-semibold text-slate-700 hover:text-brand-teal transition focus:outline-none shadow-sm"
                aria-haspopup="listbox"
                aria-expanded={langOpen}
              >
                <svg className="w-3.5 h-3.5 text-brand-teal" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                </svg>
                <span>{LANG_LABELS[currentLang]}</span>
                <svg className={`w-3 h-3 transition-transform ${langOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {langOpen && (
                <div className="absolute right-0 mt-2 w-36 bg-white border border-teal-100 rounded-xl shadow-lg py-1 z-50 animate-fadeIn">
                  {SUPPORTED_LANGS.map((lang) => (
                    <button
                      key={lang}
                      id={`lang-option-${lang}`}
                      onClick={() => handleLangChange(lang)}
                      className={`w-full text-left px-4 py-2 text-sm font-medium transition-colors ${
                        currentLang === lang
                          ? 'bg-teal-50 text-brand-teal font-bold'
                          : 'text-slate-700 hover:bg-teal-50/50 hover:text-brand-teal'
                      }`}
                    >
                      {LANG_LABELS[lang]}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* User Profile & Authentication */}
            <div className="flex items-center space-x-2 pl-2 border-l border-teal-100">
              {authToken && userName ? (
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-bold text-slate-700 bg-teal-50/50 px-3 py-1.5 rounded-full border border-teal-100">
                    👤 {userName}
                  </span>
                  <button
                    onClick={logout}
                    className="px-3 py-1.5 bg-rose-50 hover:bg-rose-100 border border-rose-200 hover:border-rose-300 rounded-full text-xs font-black text-rose-600 transition focus:outline-none shadow-sm"
                  >
                    Logout
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => {
                    setAuthTab('login');
                    setShowAuthModal(true);
                  }}
                  className="px-4 py-1.5 bg-gradient-to-tr from-brand-teal to-brand-green hover:opacity-90 text-white rounded-full text-xs font-black transition focus:outline-none shadow-sm"
                >
                  Login / Signup
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Body */}
      <main className="flex-grow max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 flex flex-col justify-center">
        {/* Banner Alert for connection problems */}
        {!health.loading && (health.error || !health.dbConnected) && (
          <div className="mb-8 p-4 bg-amber-50 border-l-4 border-amber-500 rounded-r-xl shadow-sm text-sm text-amber-800 flex items-start space-x-3">
            <svg className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <p className="font-bold">{t('banner.title')}</p>
              <p className="mt-0.5 opacity-90">
                {health.error
                  ? t('banner.apiError', { error: health.error })
                  : t('banner.dbError')}
              </p>
            </div>
          </div>
        )}

        {/* Hero Section */}
        <div className="text-center max-w-3xl mx-auto mb-12">
          <div className="inline-flex items-center space-x-1.5 px-3 py-1 bg-teal-50 border border-teal-100 rounded-full text-xs font-semibold text-brand-teal mb-4 uppercase tracking-wider">
            <span>{t('hero.badge')}</span>
          </div>
          <h1 className="text-5xl sm:text-6xl font-extrabold text-slate-900 tracking-tight leading-none mb-6">
            {t('hero.headline1')}{' '}
            <span className="bg-gradient-to-r from-brand-teal to-brand-green bg-clip-text text-transparent">
              {t('hero.headline2')}
            </span>
          </h1>
          <p className="text-lg sm:text-xl text-slate-600 font-normal leading-relaxed">
            {t('hero.tagline')}
          </p>
        </div>

        {/* Analysis & Actions Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Main Upload Control Panel */}
          <div className="lg:col-span-7 bg-white/70 backdrop-blur-md rounded-3xl border border-teal-100/50 shadow-xl shadow-teal-900/5 p-8 flex flex-col">
            
            {/* Tab Selector */}
            <div className="flex flex-wrap bg-slate-100/80 p-1 rounded-2xl mb-6 max-w-lg border border-slate-200/40 gap-1 sm:gap-0">
              <button
                type="button"
                onClick={() => setActiveTab('scan')}
                className={`flex-1 min-w-[70px] py-1.5 px-3 sm:px-4 rounded-xl text-[10px] sm:text-xs font-bold transition-all duration-200 ${
                  activeTab === 'scan' ? 'bg-white text-brand-teal shadow-sm' : 'text-slate-500 hover:text-slate-800'
                }`}
              >
                📷 {t('search.tabScan')}
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('search')}
                className={`flex-1 min-w-[70px] py-1.5 px-3 sm:px-4 rounded-xl text-[10px] sm:text-xs font-bold transition-all duration-200 ${
                  activeTab === 'search' ? 'bg-white text-brand-teal shadow-sm' : 'text-slate-500 hover:text-slate-800'
                }`}
              >
                🔍 {t('search.tabSearch')}
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('history')}
                className={`flex-1 min-w-[70px] py-1.5 px-3 sm:px-4 rounded-xl text-[10px] sm:text-xs font-bold transition-all duration-200 ${
                  activeTab === 'history' ? 'bg-white text-brand-teal shadow-sm' : 'text-slate-500 hover:text-slate-800'
                }`}
              >
                📜 History
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('favorites')}
                className={`flex-1 min-w-[70px] py-1.5 px-3 sm:px-4 rounded-xl text-[10px] sm:text-xs font-bold transition-all duration-200 ${
                  activeTab === 'favorites' ? 'bg-white text-brand-teal shadow-sm' : 'text-slate-500 hover:text-slate-800'
                }`}
              >
                ❤️ Favorites
              </button>
            </div>

            {activeTab === 'scan' && (
              <div className="flex flex-col">
                <h2 className="text-2xl font-bold text-slate-800 mb-2">{t('upload.title')}</h2>
                <p className="text-sm text-slate-500 mb-6">{t('upload.subtitle')}</p>
                
                {/* Hidden native input */}
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept="image/jpeg, image/png"
                  className="hidden"
                />

                {/* Error alerts during staging/analysis */}
                {analysis.error && (
                  <div className="mb-6 p-4 bg-amber-50 border border-amber-100 rounded-2xl text-sm text-amber-800 flex items-start space-x-3">
                    <svg className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div className="flex-1">
                      <p className="font-semibold">{t('errors.title')}</p>
                      <p className="opacity-95 mt-0.5">{analysis.error}</p>
                    </div>
                  </div>
                )}

                {/* Staged Image Preview vs Upload Area */}
                {previewUrl ? (
                  <div className="w-full flex flex-col items-center">
                    <div className="relative w-full max-h-[350px] overflow-hidden rounded-2xl border border-slate-200 shadow-inner bg-slate-900/5 flex items-center justify-center mb-6">
                      <img
                        src={previewUrl}
                        alt="Food packaging preview"
                        className="max-h-[350px] object-contain"
                      />
                      
                      {/* Reset overlay button */}
                      {!analysis.analyzing && (
                        <button
                          onClick={clearStagedImage}
                          className="absolute top-3 right-3 bg-slate-900/70 hover:bg-slate-900/90 text-white rounded-full p-2 transition shadow-md"
                          title="Clear image"
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      )}
                    </div>

                    {/* Staged State Action Buttons */}
                    <div className="w-full flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4">
                      <button
                        type="button"
                        disabled={analysis.analyzing}
                        onClick={analyzeIngredients}
                        className="flex-1 px-6 py-3 bg-brand-teal hover:bg-teal-700 disabled:bg-teal-400 text-white font-bold rounded-xl transition shadow-md hover:shadow-lg focus:outline-none flex items-center justify-center space-x-2"
                      >
                        {analysis.analyzing ? (
                          <>
                            <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                            <span>{t('upload.analyzing')}</span>
                          </>
                        ) : (
                          <>
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                            </svg>
                            <span>{t('upload.analyzeButton')}</span>
                          </>
                        )}
                      </button>
                      
                      {!analysis.analyzing && (
                        <button
                          type="button"
                          onClick={triggerFileSelect}
                          className="px-6 py-3 bg-white border border-teal-200 text-brand-teal hover:bg-teal-50 font-bold rounded-xl transition focus:outline-none"
                        >
                          {t('upload.changePhoto')}
                        </button>
                      )}
                    </div>
                  </div>
                ) : (
                  /* Drag and Drop staging frame */
                  <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={triggerFileSelect}
                    className={`w-full border-2 border-dashed transition-all duration-300 rounded-2xl p-12 flex flex-col items-center justify-center cursor-pointer group ${
                      isDragActive
                        ? 'border-brand-teal bg-teal-50/50'
                        : 'border-teal-200 bg-teal-50/10 hover:border-brand-teal hover:bg-teal-50/30'
                    }`}
                  >
                    <div className="w-16 h-16 rounded-2xl bg-teal-50 text-brand-teal group-hover:bg-brand-teal group-hover:text-white transition-all duration-300 flex items-center justify-center mb-4 shadow-sm">
                      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                    </div>
                    <p className="text-base font-semibold text-slate-700 mb-1 group-hover:text-brand-teal transition-colors">
                      {isDragActive ? t('upload.dropPrompt') : t('upload.dragPrompt')}
                    </p>
                    <p className="text-xs text-slate-400 mb-6">{t('upload.hint')}</p>
                    
                    <button
                      type="button"
                      className="px-6 py-2.5 bg-brand-teal hover:bg-teal-700 text-white font-bold rounded-xl transition shadow-md hover:shadow-lg focus:outline-none"
                    >
                      {t('upload.button')}
                    </button>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'search' && (
              /* Brand/Product Name Search Interface */
              <div className="flex flex-col h-full">
                <h2 className="text-2xl font-bold text-slate-800 mb-2">{t('search.tabSearch')}</h2>
                <p className="text-sm text-slate-500 mb-6">{t('search.placeholder')}</p>
                
                <form onSubmit={searchProducts} className="flex space-x-3 mb-6">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder={t('search.placeholder')}
                    className="flex-grow px-4 py-2.5 bg-white border border-teal-200 focus:border-brand-teal rounded-xl outline-none text-slate-800 font-medium placeholder-slate-400 transition"
                  />
                  <button
                    type="submit"
                    disabled={searching}
                    className="px-6 py-2.5 bg-brand-teal hover:bg-teal-700 text-white font-bold rounded-xl transition shadow-md hover:shadow-lg focus:outline-none flex items-center justify-center space-x-2"
                  >
                    {searching ? (
                      <>
                        <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        <span>{t('search.searching')}</span>
                      </>
                    ) : (
                      <span>{t('search.button')}</span>
                    )}
                  </button>
                </form>

                {searchError && (
                  <div className="mb-4 p-3 bg-amber-50 border border-amber-100 rounded-xl text-xs font-semibold text-amber-800 flex items-center space-x-2">
                    <span>⚠️</span>
                    <span>{searchError}</span>
                  </div>
                )}

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-[300px] overflow-y-auto pr-1">
                  {searchResults.map((product) => (
                    <div
                      key={product.code}
                      onClick={() => analyzeBarcode(product.code, product.name, product.image)}
                      className="flex items-center space-x-3 p-3 bg-white hover:bg-teal-50/20 border border-slate-100 hover:border-teal-200 rounded-2xl cursor-pointer transition shadow-sm hover:shadow group"
                    >
                      <div className="w-12 h-12 bg-slate-50 border border-slate-150 rounded-xl overflow-hidden flex items-center justify-center shrink-0">
                        {product.image ? (
                          <img src={product.image} alt={product.name} className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
                        ) : (
                          <span className="text-xl">🥫</span>
                        )}
                      </div>
                      <div className="min-w-0 flex-grow">
                        <h4 className="text-xs font-bold text-slate-800 truncate group-hover:text-brand-teal transition-colors">{product.name}</h4>
                        <p className="text-[10px] text-slate-500 font-medium truncate">{product.brand}</p>
                        <p className="text-[9px] text-slate-400 font-mono mt-0.5 truncate">{product.code}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'history' && (
              <div className="flex flex-col h-full">
                <h2 className="text-2xl font-bold text-slate-800 mb-2">📜 Scan History</h2>
                <p className="text-sm text-slate-500 mb-6">Your recently analyzed food labels.</p>

                {!authToken ? (
                  <div className="flex flex-col items-center justify-center py-10 text-center animate-fadeIn">
                    <span className="text-4xl mb-4">🔒</span>
                    <h3 className="text-base font-extrabold text-slate-800 mb-2">Login to save your scan history</h3>
                    <p className="text-xs text-slate-500 max-w-sm mb-6">Create a free account or login to automatically sync and save all your scanned products.</p>
                    <button
                      onClick={() => {
                        setAuthTab('login');
                        setShowAuthModal(true);
                      }}
                      className="px-6 py-2.5 bg-brand-teal hover:bg-teal-700 text-white font-bold rounded-xl text-xs transition shadow-sm hover:shadow focus:outline-none"
                    >
                      Login Now
                    </button>
                  </div>
                ) : loadingHistory ? (
                  <div className="flex justify-center py-12">
                    <svg className="animate-spin h-8 w-8 text-brand-teal" fill="none" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                  </div>
                ) : userHistory.length === 0 ? (
                  <div className="text-center py-12 text-slate-400">
                    <span className="text-4xl block mb-2">📜</span>
                    <p className="text-xs font-semibold">No scans recorded yet. Go scan or search a product!</p>
                  </div>
                ) : (
                  <div className="flex flex-col space-y-3 max-h-[350px] overflow-y-auto pr-1 animate-fadeIn">
                    {userHistory.map((h) => (
                      <div
                        key={h.id}
                        onClick={() => {
                          setParsedIngredients(h.ingredients_data || []);
                          setOverallScore(h.overall_score);
                          setScoreLabel(h.score_label || 'Good');
                          setProductGuess(h.product_name);
                          setPreviewUrl(null);
                          setAlternative(null);
                          if (h.ingredients_data && h.ingredients_data.length > 0) {
                            fetchAlternative(h.product_name, h.ingredients_data);
                          }
                        }}
                        className="flex items-center justify-between p-4 bg-white hover:bg-teal-50/10 border border-slate-100 hover:border-teal-200 rounded-2xl cursor-pointer transition shadow-sm hover:shadow group"
                      >
                        <div className="min-w-0 flex-1 pr-3">
                          <h4 className="font-bold text-slate-800 text-sm truncate group-hover:text-brand-teal transition-colors">
                            {h.product_name}
                          </h4>
                          <p className="text-[10px] text-slate-400 mt-0.5">
                            Scanned: {h.scanned_at ? new Date(h.scanned_at).toLocaleDateString() : 'N/A'}
                          </p>
                        </div>
                        <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase shrink-0 ${
                          h.overall_score >= 80 ? 'bg-emerald-50 text-emerald-700' : h.overall_score >= 60 ? 'bg-yellow-50 text-yellow-700' : h.overall_score >= 40 ? 'bg-amber-50 text-amber-700' : 'bg-orange-50/70 text-orange-700'
                        }`}>
                          {h.overall_score}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'favorites' && (
              <div className="flex flex-col h-full">
                <h2 className="text-2xl font-bold text-slate-800 mb-2">❤️ Favorites</h2>
                <p className="text-sm text-slate-500 mb-6">Your collection of healthy or saved food items.</p>

                {!authToken ? (
                  <div className="flex flex-col items-center justify-center py-10 text-center animate-fadeIn">
                    <span className="text-4xl mb-4">🔒</span>
                    <h3 className="text-base font-extrabold text-slate-800 mb-2">Login to save your favorites</h3>
                    <p className="text-xs text-slate-500 max-w-sm mb-6">Create a free account or login to automatically sync and save all your favorite products.</p>
                    <button
                      onClick={() => {
                        setAuthTab('login');
                        setShowAuthModal(true);
                      }}
                      className="px-6 py-2.5 bg-brand-teal hover:bg-teal-700 text-white font-bold rounded-xl text-xs transition shadow-sm hover:shadow focus:outline-none"
                    >
                      Login Now
                    </button>
                  </div>
                ) : loadingFavorites ? (
                  <div className="flex justify-center py-12">
                    <svg className="animate-spin h-8 w-8 text-brand-teal" fill="none" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                  </div>
                ) : userFavorites.length === 0 ? (
                  <div className="text-center py-12 text-slate-400">
                    <span className="text-4xl block mb-2">❤️</span>
                    <p className="text-xs font-semibold">No favorites saved yet. Heart any product scan result to add it here!</p>
                  </div>
                ) : (
                  <div className="flex flex-col space-y-3 max-h-[350px] overflow-y-auto pr-1 animate-fadeIn">
                    {userFavorites.map((f) => (
                      <div
                        key={f.id}
                        className="flex items-center justify-between p-4 bg-white hover:bg-teal-50/10 border border-slate-100 hover:border-teal-200 rounded-2xl transition shadow-sm hover:shadow group"
                      >
                        <div
                          onClick={() => {
                            setParsedIngredients(f.ingredients_data || []);
                            setOverallScore(f.overall_score);
                            setScoreLabel(f.overall_score >= 80 ? 'Excellent' : f.overall_score >= 60 ? 'Good' : f.overall_score >= 40 ? 'Moderate' : 'Poor');
                            setProductGuess(f.product_name);
                            setPreviewUrl(null);
                            setAlternative(null);
                            if (f.ingredients_data && f.ingredients_data.length > 0) {
                              fetchAlternative(f.product_name, f.ingredients_data);
                            }
                          }}
                          className="min-w-0 flex-1 pr-3 cursor-pointer"
                        >
                          <h4 className="font-bold text-slate-800 text-sm truncate group-hover:text-brand-teal transition-colors">
                            {f.product_name}
                          </h4>
                          <p className="text-[10px] text-slate-400 mt-0.5">
                            Saved: {f.added_at ? new Date(f.added_at).toLocaleDateString() : 'N/A'}
                          </p>
                        </div>
                        
                        <div className="flex items-center space-x-3 shrink-0">
                          <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase ${
                            f.overall_score >= 80 ? 'bg-emerald-50 text-emerald-700' : f.overall_score >= 60 ? 'bg-yellow-50 text-yellow-700' : f.overall_score >= 40 ? 'bg-amber-50 text-amber-700' : 'bg-orange-50/70 text-orange-700'
                          }`}>
                            {f.overall_score}
                          </span>
                          <button
                            onClick={() => deleteFavorite(f.id)}
                            className="p-1.5 text-rose-500 hover:text-rose-700 hover:bg-rose-50 rounded-lg transition"
                            title="Remove from favorites"
                          >
                            🗑️
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Gemini Extracted Output Panel */}
          <div className="lg:col-span-5 flex flex-col h-full">
            {analysis.rawText ? (
              <div className="bg-white/95 rounded-3xl border border-teal-100 shadow-xl shadow-teal-900/5 p-6 flex flex-col h-full animate-fadeIn">
                <div className="flex items-center justify-between pb-4 border-b border-teal-50 mb-4">
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 rounded-lg bg-teal-50 text-brand-teal flex items-center justify-center font-bold">
                      💡
                    </div>
                    <h3 className="font-bold text-slate-800 text-lg">{t('extracted.title')}</h3>
                  </div>
                  <button
                    onClick={copyToClipboard}
                    className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 transition text-slate-500 hover:text-slate-800 flex items-center space-x-1 text-xs font-semibold focus:outline-none"
                  >
                    {copied ? (
                      <>
                        <svg className="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                        </svg>
                        <span className="text-emerald-600">{t('extracted.copied')}</span>
                      </>
                    ) : (
                      <>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                        </svg>
                        <span>{t('extracted.copy')}</span>
                      </>
                    )}
                  </button>
                </div>
                
                <div className="flex-grow max-h-[300px] overflow-y-auto bg-slate-50 border border-slate-100 p-4 rounded-2xl text-slate-700 font-mono text-sm leading-relaxed whitespace-pre-wrap">
                  {analysis.rawText}
                </div>
                
                <p className="text-xs text-slate-400 mt-4 italic text-center">
                  {t('extracted.footer')}
                </p>
              </div>
            ) : (
              /* Idle state features details */
              <div className="space-y-6">
                <div className="bg-white/60 backdrop-blur-md rounded-2xl border border-teal-100/40 p-6 shadow-md shadow-teal-900/5">
                  <div className="flex items-center space-x-4 mb-3">
                    <div className="w-10 h-10 rounded-lg bg-teal-50 text-brand-teal flex items-center justify-center font-semibold">
                      🤖
                    </div>
                    <h3 className="font-bold text-slate-800 text-lg">{t('features.ai.title')}</h3>
                  </div>
                  <p className="text-sm text-slate-600 leading-relaxed">{t('features.ai.desc')}</p>
                </div>

                <div className="bg-white/60 backdrop-blur-md rounded-2xl border border-teal-100/40 p-6 shadow-md shadow-teal-900/5">
                  <div className="flex items-center space-x-4 mb-3">
                    <div className="w-10 h-10 rounded-lg bg-teal-50 text-brand-teal flex items-center justify-center font-semibold">
                      👁️
                    </div>
                    <h3 className="font-bold text-slate-800 text-lg">{t('features.blur.title')}</h3>
                  </div>
                  <p className="text-sm text-slate-600 leading-relaxed">{t('features.blur.desc')}</p>
                </div>
              </div>
            )}
        </div>
      </div>

      {/* Compare by Category Leaderboard Card */}
      <div className="mt-12 bg-white/70 backdrop-blur-md rounded-3xl border border-teal-100/50 shadow-xl shadow-teal-900/5 p-8 flex flex-col animate-fadeIn">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div>
            <h3 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
              <span>📊</span> {t('compare.title')}
            </h3>
            <p className="text-sm text-slate-500">{t('compare.subtitle')}</p>
          </div>
          
          <div className="relative shrink-0">
            <select
              id="category-compare-select"
              value={selectedCategory}
              onChange={handleCategoryChange}
              className="w-48 px-4 py-2 bg-white border border-teal-200 focus:border-brand-teal rounded-xl text-xs font-bold text-slate-700 outline-none transition shadow-sm cursor-pointer appearance-none pr-8"
            >
              <option value="">{t('compare.select')}</option>
              <option value="biscuits">{t('compare.biscuits')}</option>
              <option value="ketchup">{t('compare.ketchup')}</option>
              <option value="juice">{t('compare.juice')}</option>
              <option value="chips">{t('compare.chips')}</option>
              <option value="chocolates">{t('compare.chocolates')}</option>
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-500">
              <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/>
              </svg>
            </div>
          </div>
        </div>

        {loadingLeaderboard && (
          <div className="flex flex-col items-center justify-center py-12">
            <svg className="animate-spin h-8 w-8 text-brand-teal mb-3" fill="none" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
            </svg>
            <p className="text-xs font-semibold text-slate-500">{t('alternative.loading')}</p>
          </div>
        )}

        {leaderboardError && (
          <div className="p-4 bg-amber-50 border border-amber-100 rounded-2xl text-xs font-semibold text-amber-800 flex items-center space-x-2">
            <span>⚠️</span>
            <span>{leaderboardError}</span>
          </div>
        )}

        {!loadingLeaderboard && leaderboard.length > 0 && (
          <div className="overflow-x-auto border border-slate-150 rounded-2xl shadow-sm bg-white animate-fadeIn">
            <table className="min-w-full divide-y divide-slate-150 text-sm">
              <thead className="bg-slate-50">
                <tr>
                  <th scope="col" className="px-4 py-3 text-center font-bold text-slate-500 uppercase tracking-wider text-xs w-16">
                    {t('compare.rank')}
                  </th>
                  <th scope="col" className="px-4 py-3 text-left font-bold text-slate-500 uppercase tracking-wider text-xs">
                    {t('compare.product')}
                  </th>
                  <th scope="col" className="px-4 py-3 text-center font-bold text-slate-500 uppercase tracking-wider text-xs w-28">
                    {t('compare.score')}
                  </th>
                  <th scope="col" className="px-4 py-3 text-center font-bold text-slate-500 uppercase tracking-wider text-xs w-32">
                    {/* Action */}
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-150 bg-white">
                {leaderboard.map((item) => {
                  const isGold = item.rank === 1;
                  const isSilver = item.rank === 2;
                  const isBronze = item.rank === 3;
                  
                  let rankBadge = (
                    <span className="w-6 h-6 rounded-full bg-slate-100 text-slate-600 flex items-center justify-center font-bold text-xs mx-auto">
                      {item.rank}
                    </span>
                  );
                  if (isGold) {
                    rankBadge = (
                      <span className="w-6 h-6 rounded-full bg-yellow-100 border border-yellow-350 text-yellow-800 flex items-center justify-center font-bold text-xs mx-auto shadow-sm" title="Gold Medal">
                        🥇
                      </span>
                    );
                  } else if (isSilver) {
                    rankBadge = (
                      <span className="w-6 h-6 rounded-full bg-slate-200 border border-slate-350 text-slate-800 flex items-center justify-center font-bold text-xs mx-auto shadow-sm" title="Silver Medal">
                        🥈
                      </span>
                    );
                  } else if (isBronze) {
                    rankBadge = (
                      <span className="w-6 h-6 rounded-full bg-amber-100 border border-amber-300 text-amber-800 flex items-center justify-center font-bold text-xs mx-auto shadow-sm" title="Bronze Medal">
                        🥉
                      </span>
                    );
                  }

                  return (
                    <tr key={item.code} className="hover:bg-slate-50/50 transition-colors">
                      <td className="px-4 py-3 text-center">{rankBadge}</td>
                      <td className="px-4 py-3">
                        <div className="flex items-center space-x-3">
                          <div className="w-10 h-10 bg-slate-50 border border-slate-100 rounded-lg overflow-hidden flex items-center justify-center shrink-0">
                            {item.image ? (
                              <img src={item.image} alt={item.name} className="w-full h-full object-cover" />
                            ) : (
                              <span className="text-lg">🥫</span>
                            )}
                          </div>
                          <div className="min-w-0">
                            <p className="font-bold text-slate-800 truncate text-xs sm:text-sm">{item.name}</p>
                            <p className="text-[10px] text-slate-500 font-medium truncate">{item.brand}</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-black uppercase tracking-wider ${
                          item.score >= 80 ? 'bg-emerald-50 text-emerald-700' : item.score >= 60 ? 'bg-yellow-50 text-yellow-700' : item.score >= 40 ? 'bg-amber-50 text-amber-700' : 'bg-orange-50/70 text-orange-700'
                        }`}>
                          {item.score} ({item.label})
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <button
                          type="button"
                          onClick={() => analyzeBarcode(item.code, item.name, item.image)}
                          className="px-3 py-1 bg-white hover:bg-teal-50 border border-teal-200 hover:border-brand-teal text-brand-teal font-bold rounded-lg text-xs transition shadow-sm focus:outline-none"
                        >
                          {t('compare.viewDetails')}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Parsed Ingredients Safety Assessment */}
      {(parsing || parsedIngredients.length > 0) && (
          <div className="mt-12 bg-white/80 backdrop-blur-md rounded-3xl border border-teal-100/50 shadow-xl p-8 animate-fadeIn">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-teal to-brand-green flex items-center justify-center text-white text-xl shadow-md">
                🛡️
              </div>
              <div>
                <h3 className="text-2xl font-bold text-slate-800">{t('safety.title')}</h3>
                <p className="text-sm text-slate-500">{t('safety.subtitle')}</p>
              </div>
            </div>

            {parsing ? (
              <div className="flex flex-col items-center justify-center py-12">
                <svg className="animate-spin h-10 w-10 text-brand-teal mb-4" fill="none" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <p className="text-slate-600 font-semibold">{t('safety.loading')}</p>
                <p className="text-xs text-slate-400 mt-1">{t('safety.loadingSubtitle')}</p>
              </div>
            ) : (
              <div>
                {/* Product Guess Title */}
                {productGuess && (
                  <div className="flex flex-col items-center justify-center text-center mb-6 relative">
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-1">{t('safety.detectedCategory')}</span>
                    <div className="flex items-center justify-center space-x-2">
                      <span className="text-xl sm:text-2xl font-extrabold text-slate-800 bg-gradient-to-r from-teal-700 to-emerald-700 bg-clip-text text-transparent">
                        {productGuess}
                      </span>
                      
                      {/* Favorite Button */}
                      <button
                        onClick={addFavorite}
                        className={`p-1.5 rounded-full border transition-all duration-300 focus:outline-none ${
                          userFavorites.some((f) => f.product_name === productGuess)
                            ? 'bg-rose-50 border-rose-200 text-rose-500 scale-105 shadow-sm'
                            : 'bg-slate-50 border-slate-200 hover:border-rose-300 text-slate-400 hover:text-rose-500'
                        }`}
                        title={userFavorites.some((f) => f.product_name === productGuess) ? "Favorited" : "Add to favorites"}
                      >
                        {userFavorites.some((f) => f.product_name === productGuess) ? '❤️' : '🖤'}
                      </button>
                    </div>
                  </div>
                )}

                {/* Score + Chart Row */}
                {overallScore !== null && (
                  <div className="flex flex-col sm:flex-row items-center sm:items-start justify-center gap-6 mb-8">

                    {/* Circular Score Badge */}
                    <div className="flex flex-col items-center justify-center p-4 bg-teal-50/10 border border-teal-100/30 rounded-2xl min-w-[160px] shadow-sm">
                      <div className="relative w-28 h-28 flex items-center justify-center">
                        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                          <circle cx="50" cy="50" r="38" className="stroke-slate-100 fill-none" strokeWidth="8" />
                          <circle
                            cx="50" cy="50" r="38"
                            className={`fill-none transition-all duration-1000 ${
                              overallScore >= 80 ? 'stroke-emerald-500' : overallScore >= 60 ? 'stroke-yellow-500' : overallScore >= 40 ? 'stroke-amber-500' : 'stroke-orange-500'
                            }`}
                            strokeWidth="8"
                            strokeDasharray="238.76"
                            strokeDashoffset={238.76 - (238.76 * overallScore) / 100}
                            strokeLinecap="round"
                          />
                        </svg>
                        <div className="absolute flex flex-col items-center">
                          <span className={`text-2xl font-black ${
                            overallScore >= 80 ? 'text-emerald-600' : overallScore >= 60 ? 'text-yellow-600' : overallScore >= 40 ? 'text-amber-600' : 'text-orange-600'
                          }`}>{overallScore}</span>
                          <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Score</span>
                        </div>
                      </div>
                      <div className={`mt-3 px-3 py-0.5 rounded-full text-xs font-black uppercase tracking-wider ${
                        overallScore >= 80 ? 'bg-emerald-50 text-emerald-700' : overallScore >= 60 ? 'bg-yellow-50 text-yellow-700' : overallScore >= 40 ? 'bg-amber-50 text-amber-700' : 'bg-orange-50/70 text-orange-700'
                      }`}>
                        {scoreLabel}
                      </div>
                    </div>

                    {/* Safety Breakdown Bar Chart */}
                    {parsedIngredients.length > 0 && (() => {
                      const counts = {
                        safe:     parsedIngredients.filter(i => i.safety_status === 'safe').length,
                        moderate: parsedIngredients.filter(i => i.safety_status === 'moderate').length,
                        unsafe:   parsedIngredients.filter(i => i.safety_status === 'unsafe').length,
                        unknown:  parsedIngredients.filter(i => !['safe','moderate','unsafe'].includes(i.safety_status)).length,
                      };
                      const chartData = [
                        { name: 'Safe',     value: counts.safe,     color: '#10b981' },
                        { name: 'Moderate', value: counts.moderate, color: '#f59e0b' },
                        { name: 'Unsafe',   value: counts.unsafe,   color: '#ef4444' },
                        { name: 'Unknown',  value: counts.unknown,  color: '#94a3b8' },
                      ].filter(d => d.value > 0);
                      return (
                        <div className="flex-1 min-w-0 bg-white/60 border border-slate-100 rounded-2xl p-4 shadow-sm">
                          <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-3">Ingredient Breakdown</p>
                          <ResponsiveContainer width="100%" height={120}>
                            <BarChart data={chartData} margin={{ top: 0, right: 4, left: -28, bottom: 0 }} barSize={28}>
                              <XAxis dataKey="name" tick={{ fontSize: 10, fontWeight: 600, fill: '#64748b' }} axisLine={false} tickLine={false} />
                              <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} allowDecimals={false} axisLine={false} tickLine={false} />
                              <Tooltip
                                contentStyle={{ borderRadius: '10px', border: '1px solid #e2e8f0', fontSize: 12 }}
                                cursor={{ fill: 'rgba(0,0,0,0.04)' }}
                              />
                              <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                                {chartData.map((entry, index) => (
                                  <Cell key={index} fill={entry.color} />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      );
                    })()}
                  </div>
                )}

                {/* ── Allergen Summary Banner ── */}
                {parsedIngredients.length > 0 && (() => {
                  const allergens = [...new Set(
                    parsedIngredients
                      .map(i => i.allergen)
                      .filter(a => a && a.toLowerCase() !== 'none')
                      .map(a => a.charAt(0).toUpperCase() + a.slice(1))
                  )];
                  if (allergens.length === 0) return null;
                  return (
                    <div className="mb-5 flex items-center space-x-3 px-4 py-3 bg-amber-50 border border-amber-200 rounded-xl shadow-sm animate-fadeIn">
                      <span className="text-xl shrink-0">⚠️</span>
                      <div>
                        <span className="text-sm font-bold text-amber-800">Contains: </span>
                        <span className="text-sm font-semibold text-amber-700">{allergens.join(', ')}</span>
                        <p className="text-xs text-amber-600 mt-0.5">This product may not be suitable for people with these allergies.</p>
                      </div>
                    </div>
                  );
                })()}

                {/* Safety Assessment Table */}
                {parsedIngredients.length > 0 && (
                  <div className="overflow-x-auto border border-slate-150 rounded-2xl shadow-sm">
                    <table className="min-w-full divide-y divide-slate-150 text-sm">
                      <thead className="bg-slate-50">
                        <tr>
                          <th scope="col" className="px-4 py-3 text-left font-bold text-slate-500 uppercase tracking-wider text-xs">{t('safety.col.name')}</th>
                          <th scope="col" className="px-4 py-3 text-center font-bold text-slate-500 uppercase tracking-wider text-xs">{t('safety.col.status')}</th>
                          <th scope="col" className="px-4 py-3 text-left font-bold text-slate-500 uppercase tracking-wider text-xs">{t('safety.col.whatIsThis')}</th>
                          <th scope="col" className="px-4 py-3 text-left font-bold text-slate-500 uppercase tracking-wider text-xs">{t('safety.col.reason')}</th>
                          <th scope="col" className="px-4 py-3 text-center font-bold text-slate-500 uppercase tracking-wider text-xs">{t('safety.col.allergen')}</th>
                          <th scope="col" className="px-4 py-3 text-center font-bold text-slate-500 uppercase tracking-wider text-xs">{t('safety.col.frequency')}</th>
                          <th scope="col" className="px-4 py-3 text-center font-bold text-slate-500 uppercase tracking-wider text-xs">{t('safety.col.source')}</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-150 bg-white">
                        {parsedIngredients.map((item, idx) => {
                          const isUnsafe = item.safety_status === 'unsafe';
                          const isSafe = item.safety_status === 'safe';
                          const isModerate = item.safety_status === 'moderate';
                          
                          let rowBg = 'bg-white hover:bg-slate-50/50';
                          let statusIcon = '⚠️';
                          let reasonStyle = 'text-slate-500 font-normal';
                          
                          if (isUnsafe) {
                            rowBg = 'bg-red-50/70 hover:bg-red-50/90 border-y border-red-100';
                            statusIcon = '❌';
                            reasonStyle = 'text-red-700 font-bold';
                          } else if (isSafe) {
                            rowBg = 'bg-emerald-50/15 hover:bg-emerald-50/30';
                            statusIcon = '✓';
                          } else if (isModerate) {
                            rowBg = 'bg-amber-50/20 hover:bg-amber-50/35';
                            statusIcon = '⚠️';
                          }

                          return (
                            <tr key={idx} className={`${rowBg} transition-colors`}>
                              <td className="px-4 py-3.5 font-bold text-slate-800 whitespace-nowrap">{item.ingredient}</td>
                              <td className="px-4 py-3.5 text-center text-lg">{statusIcon}</td>
                              <td className="px-4 py-3.5 text-slate-600 leading-relaxed font-normal min-w-[200px]">{item.simple_explanation}</td>
                              <td className={`px-4 py-3.5 leading-relaxed min-w-[150px] ${reasonStyle}`}>{item.reason || '-'}</td>
                              <td className="px-4 py-3.5 text-center whitespace-nowrap">
                                {item.allergen !== 'none' ? (
                                  <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${
                                    isUnsafe ? 'bg-red-100 text-red-800' : 'bg-amber-100 text-amber-800'
                                  }`}>
                                    {item.allergen}
                                  </span>
                                ) : (
                                  <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-500">{t('safety.col.noAllergen')}</span>
                                )}
                              </td>
                              <td className="px-4 py-3.5 text-center whitespace-nowrap font-medium text-slate-600">{item.safe_frequency}</td>
                              <td className="px-4 py-3.5 text-center whitespace-nowrap text-slate-500 font-medium">{item.source}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* ── Sources Citation ── */}
                {parsedIngredients.length > 0 && (() => {
                  const usedSources = [...new Set(parsedIngredients.map(i => i.source).filter(Boolean))];
                  const SOURCE_META: Record<string, { icon: string; url: string }> = {
                    'FSSAI':           { icon: '🇮🇳', url: 'https://www.fssai.gov.in' },
                    'WHO':             { icon: '🌍', url: 'https://www.who.int' },
                    'ICMR':            { icon: '🔬', url: 'https://www.icmr.gov.in' },
                    'FDA':             { icon: '🇺🇸', url: 'https://www.fda.gov' },
                    'EFSA':            { icon: '🇪🇺', url: 'https://www.efsa.europa.eu' },
                    'Open Food Facts': { icon: '🥫', url: 'https://world.openfoodfacts.org' },
                    'Gemini AI':       { icon: '🤖', url: 'https://ai.google.dev' },
                    'System Fallback': { icon: '⚙️', url: '#' },
                  };
                  return (
                    <div className="mt-6 pt-5 border-t border-slate-100">
                      <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-3">Sources</p>
                      <div className="flex flex-wrap gap-2 mb-2">
                        {usedSources.map((src) => {
                          const meta = SOURCE_META[src] || { icon: '📄', url: '#' };
                          return (
                            <a
                              key={src}
                              href={meta.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center space-x-1.5 px-3 py-1 bg-slate-50 hover:bg-teal-50 border border-slate-200 hover:border-teal-200 rounded-full text-xs font-semibold text-slate-600 hover:text-brand-teal transition-colors"
                            >
                              <span>{meta.icon}</span>
                              <span>{src}</span>
                            </a>
                          );
                        })}
                      </div>
                      <p className="text-[11px] text-slate-400 italic">Safety data based on official food safety guidelines and open data sources. This app is for informational purposes only and does not constitute medical advice.</p>
                    </div>
                  );
                })()}
              </div>
            )}

            {/* ── Alternative Suggestion Card ── */}
            {!parsing && (loadingAlternative || alternative) && (
              <div className="mt-8 animate-fadeIn">
                {loadingAlternative ? (
                  <div className="flex items-center space-x-3 p-5 bg-teal-50/60 border border-teal-100 rounded-2xl">
                    <svg className="animate-spin h-5 w-5 text-brand-teal shrink-0" fill="none" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <p className="text-sm font-semibold text-teal-700">{t('alternative.loading')}</p>
                  </div>
                ) : alternative ? (
                  <div className="relative overflow-hidden rounded-2xl border border-emerald-200 bg-gradient-to-br from-emerald-50 to-teal-50/60 shadow-md shadow-emerald-900/5 p-6">
                    {/* Decorative background blob */}
                    <div className="absolute -top-6 -right-6 w-32 h-32 rounded-full bg-emerald-100/50 blur-2xl pointer-events-none" />

                    <div className="flex items-start space-x-4">
                      <div className="w-11 h-11 rounded-xl bg-emerald-100 text-emerald-700 flex items-center justify-center text-xl shrink-0 shadow-sm">
                        💡
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-[10px] font-bold uppercase tracking-widest text-emerald-600 mb-0.5">{t('alternative.label')}</p>
                        <h4 className="text-lg font-extrabold text-slate-800 leading-tight mb-3">
                          {t('alternative.heading')}&nbsp;
                          <span className="bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent">
                            {alternative.alternative_name}
                          </span>
                        </h4>
                        {alternative.reasons.length > 0 && (
                          <ul className="space-y-1.5">
                            {alternative.reasons.map((reason, i) => (
                              <li key={i} className="flex items-start space-x-2 text-sm text-slate-600">
                                <span className="mt-0.5 w-4 h-4 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center text-[10px] font-black shrink-0">
                                  {i + 1}
                                </span>
                                <span>{reason}</span>
                              </li>
                            ))}
                          </ul>
                        )}
                        <p className="mt-4 text-[10px] text-slate-400 italic">{t('alternative.disclaimer')}</p>
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-slate-900 text-slate-400 py-6 mt-12 border-t border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-xs">
          <p className="font-medium">{t('footer.rights', { year: new Date().getFullYear() })}</p>
          <p className="mt-1 opacity-70">{t('footer.tagline')}</p>
        </div>
      </footer>

      {/* ── Phase 9: Authentication Modal ── */}
      {showAuthModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-fadeIn">
          <div className="relative w-full max-w-md bg-white rounded-3xl border border-teal-100 shadow-2xl p-8 flex flex-col animate-slideUp">
            
            {/* Close Button */}
            <button
              onClick={() => {
                setShowAuthModal(false);
                setAuthError(null);
              }}
              className="absolute top-4 right-4 p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-full transition focus:outline-none"
            >
              ✕
            </button>

            {/* Modal Tabs: Login / Signup */}
            <div className="flex border-b border-slate-100 mb-6">
              <button
                onClick={() => {
                  setAuthTab('login');
                  setAuthError(null);
                }}
                className={`flex-1 pb-3 text-sm font-bold transition-all border-b-2 text-center focus:outline-none ${
                  authTab === 'login' 
                    ? 'border-brand-teal text-brand-teal' 
                    : 'border-transparent text-slate-400 hover:text-slate-700'
                }`}
              >
                Log In
              </button>
              <button
                onClick={() => {
                  setAuthTab('signup');
                  setAuthError(null);
                }}
                className={`flex-1 pb-3 text-sm font-bold transition-all border-b-2 text-center focus:outline-none ${
                  authTab === 'signup' 
                    ? 'border-brand-teal text-brand-teal' 
                    : 'border-transparent text-slate-400 hover:text-slate-700'
                }`}
              >
                Sign Up
              </button>
            </div>

            {/* Modal Form */}
            <form onSubmit={authSubmit} className="space-y-4">
              {authTab === 'signup' && (
                <div>
                  <label className="block text-xs font-bold text-slate-500 uppercase mb-1">Full Name</label>
                  <input
                    type="text"
                    value={authName}
                    onChange={(e) => setAuthName(e.target.value)}
                    required
                    placeholder="Enter your name"
                    className="w-full px-4 py-2 bg-slate-50 border border-slate-200 focus:border-brand-teal rounded-xl text-sm outline-none text-slate-800 transition"
                  />
                </div>
              )}

              <div>
                <label className="block text-xs font-bold text-slate-500 uppercase mb-1">Email Address</label>
                <input
                  type="email"
                  value={authEmail}
                  onChange={(e) => setAuthEmail(e.target.value)}
                  required
                  placeholder="name@example.com"
                  className="w-full px-4 py-2 bg-slate-50 border border-slate-200 focus:border-brand-teal rounded-xl text-sm outline-none text-slate-800 transition"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-500 uppercase mb-1">Password</label>
                <input
                  type="password"
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  required
                  placeholder="••••••••"
                  className="w-full px-4 py-2 bg-slate-50 border border-slate-200 focus:border-brand-teal rounded-xl text-sm outline-none text-slate-800 transition"
                />
              </div>

              {authError && (
                <div className="p-3 bg-rose-50 border border-rose-100 rounded-xl text-xs font-semibold text-rose-800 flex items-center space-x-2 animate-fadeIn">
                  <span>⚠️</span>
                  <span>{authError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={authLoading}
                className="w-full py-3 bg-gradient-to-tr from-brand-teal to-brand-green hover:opacity-90 disabled:opacity-60 text-white font-extrabold rounded-xl transition shadow-md hover:shadow-lg focus:outline-none flex items-center justify-center space-x-2 text-sm"
              >
                {authLoading ? (
                  <>
                    <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <span>Processing...</span>
                  </>
                ) : (
                  <span>{authTab === 'login' ? 'Log In' : 'Sign Up'}</span>
                )}
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
