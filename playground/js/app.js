/**
 * SweatStack PyScript Framework
 *
 * Shared utilities for PyScript apps with SweatStack OAuth.
 * Apps define CONFIG and call App.init() in their HTML.
 */

const $ = (id) => document.getElementById(id);

// Config helper - only clientId is app-specific
const SweatStack = {
    config(clientId) {
        return {
            clientId,
            redirectUri: location.origin + location.pathname,
            authUrl: 'https://app.sweatstack.no/oauth/authorize',
            tokenUrl: 'https://app.sweatstack.no/api/v1/oauth/token',
            apiBase: 'https://app.sweatstack.no/api/v1',
            scope: 'openid profile data:read',
            pyScriptVersion: '2025.3.1'
        };
    }
};

// ----------------------------------------------------------------------------
// PKCE (OAuth security)
// ----------------------------------------------------------------------------

const PKCE = {
    randomString(len = 64) {
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~';
        const values = crypto.getRandomValues(new Uint8Array(len));
        return Array.from(values, v => chars[v % chars.length]).join('');
    },

    async challenge(verifier) {
        const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier));
        return btoa(String.fromCharCode(...new Uint8Array(hash)))
            .replace(/\+/g, '-')
            .replace(/\//g, '_')
            .replace(/=+$/, '');
    }
};

// ----------------------------------------------------------------------------
// Authentication
// ----------------------------------------------------------------------------

const Auth = {
    async start() {
        const verifier = PKCE.randomString();
        localStorage.setItem('pkce_verifier', verifier);

        const params = new URLSearchParams({
            client_id: CONFIG.clientId,
            redirect_uri: CONFIG.redirectUri,
            response_type: 'code',
            scope: CONFIG.scope || 'openid profile',
            code_challenge: await PKCE.challenge(verifier),
            code_challenge_method: 'S256',
            prompt: 'none'
        });

        location.href = `${CONFIG.authUrl}?${params}`;
    },

    async handleCallback() {
        const code = new URLSearchParams(location.search).get('code');
        const verifier = localStorage.getItem('pkce_verifier');

        if (!code || !verifier) return false;

        const res = await fetch(CONFIG.tokenUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({
                grant_type: 'authorization_code',
                code,
                redirect_uri: CONFIG.redirectUri,
                client_id: CONFIG.clientId,
                code_verifier: verifier
            })
        });

        if (!res.ok) return false;

        const data = await res.json();
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('token_expiry', Date.now() + data.expires_in * 1000);
        localStorage.removeItem('pkce_verifier');
        history.replaceState({}, '', CONFIG.redirectUri);

        return true;
    },

    getToken() {
        const token = localStorage.getItem('access_token');
        const expiry = localStorage.getItem('token_expiry');

        if (token && expiry && Date.now() < parseInt(expiry)) {
            return token;
        }

        localStorage.removeItem('access_token');
        localStorage.removeItem('token_expiry');
        return null;
    },

    logout() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('token_expiry');
        $('auth').style.display = 'block';
        $('app').classList.remove('visible');
    }
};

// ----------------------------------------------------------------------------
// PyScript
// ----------------------------------------------------------------------------

const PyScript = {
    load() {
        const version = CONFIG.pyScriptVersion || '2025.3.1';
        const scriptName = window.ANALYSIS_SCRIPT || 'main.py';
        const libPath = window.LIB_PATH || '.';

        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = `https://pyscript.net/releases/${version}/core.css`;
        document.head.appendChild(link);

        const script = document.createElement('script');
        script.type = 'module';
        script.src = `https://pyscript.net/releases/${version}/core.js`;
        document.head.appendChild(script);

        const config = {
            packages: ['micropip'],
            files: {
                ['./' + scriptName]: scriptName,
                [`${libPath}/py/runtime.py`]: 'runtime.py'
            }
        };

        const extraFiles = window.ANALYSIS_FILES || [];
        for (const file of extraFiles) {
            config.files['./' + file] = file;
        }

        const py = document.createElement('script');
        py.type = 'py';
        py.src = `${libPath}/py/worker.py`;
        py.setAttribute('config', JSON.stringify(config));
        py.setAttribute('worker', '');
        document.body.appendChild(py);
    }
};

// ----------------------------------------------------------------------------
// Status Indicator
// ----------------------------------------------------------------------------

const Status = {
    _loadTimer: null,

    set(state, text) {
        for (const el of [$('status'), $('info-status')]) {
            if (!el) continue;
            el.className = 'status ' + state;
            el.querySelector('.label').textContent = text;
        }

        if (state === 'loading') {
            if (!this._loadTimer) this._startLoadTimer();
        } else {
            this._stopLoadTimer();
        }
    },

    _startLoadTimer() {
        const el = $('output-status');
        if (!el) return;

        const messages = [
            [8000,  'Setting up the analysis environment...'],
            [20000, 'Installing packages \u2014 this may take a minute on first visit...'],
            [45000, 'Almost there \u2014 large packages take a moment...'],
        ];

        const start = Date.now();
        this._loadTimer = setInterval(() => {
            const elapsed = Date.now() - start;
            for (let i = messages.length - 1; i >= 0; i--) {
                if (elapsed >= messages[i][0]) {
                    el.textContent = messages[i][1];
                    return;
                }
            }
        }, 1000);
    },

    _stopLoadTimer() {
        if (this._loadTimer) {
            clearInterval(this._loadTimer);
            this._loadTimer = null;
        }
        const el = $('output-status');
        if (el) el.textContent = '';
    }
};

// ----------------------------------------------------------------------------
// Code Modal
// ----------------------------------------------------------------------------

const CodeModal = {
    _cache: {},
    _activeFile: null,

    show() {
        $('code-modal').classList.add('visible');
        document.body.style.overflow = 'hidden';

        const scriptName = window.ANALYSIS_SCRIPT || 'main.py';
        const extraFiles = window.ANALYSIS_FILES || [];
        const files = [scriptName, ...extraFiles];
        const titleEl = $('code-modal').querySelector('.modal-title, .code-tabs');

        if (files.length > 1) {
            titleEl.className = 'code-tabs';
            titleEl.innerHTML = files.map(f =>
                `<button class="code-tab" data-file="${f}">${f}</button>`
            ).join('');
        } else {
            titleEl.className = 'modal-title';
            titleEl.textContent = scriptName;
        }

        this.showFile(this._activeFile && files.includes(this._activeFile)
            ? this._activeFile : scriptName);
    },

    showFile(name) {
        this._activeFile = name;
        const codeEl = $('code-view').querySelector('code');

        // Update active tab
        for (const tab of $('code-modal').querySelectorAll('.code-tab')) {
            tab.classList.toggle('active', tab.dataset.file === name);
        }

        if (this._cache[name]) {
            codeEl.textContent = this._cache[name];
            codeEl.removeAttribute('data-highlighted');
            hljs.highlightElement(codeEl);
            return;
        }

        codeEl.textContent = '';
        fetch('./' + name)
            .then(r => r.text())
            .then(code => {
                this._cache[name] = code;
                if (this._activeFile === name) {
                    codeEl.textContent = code;
                    codeEl.removeAttribute('data-highlighted');
                    hljs.highlightElement(codeEl);
                }
            });
    },

    _copyIcon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',

    copy() {
        const code = this._activeFile && this._cache[this._activeFile];
        if (!code) return;
        navigator.clipboard.writeText(code).then(() => {
            const btn = $('code-modal').querySelector('.modal-copy');
            if (btn) {
                btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>';
                setTimeout(() => { btn.innerHTML = this._copyIcon; }, 1500);
            }
        });
    },

    hide() {
        $('code-modal').classList.remove('visible');
        document.body.style.overflow = '';
    },

    init() {
        $('code-modal').addEventListener('click', (e) => {
            if (e.target === $('code-modal')) {
                this.hide();
                return;
            }
            const tab = e.target.closest('.code-tab');
            if (tab) this.showFile(tab.dataset.file);
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hide();
            }
        });
    }
};

// ----------------------------------------------------------------------------
// Output Panel
// ----------------------------------------------------------------------------

const OutputPanel = {
    init() {
        const show = () => {
            document.querySelector('.output-panel').classList.add('visible');
        };

        new MutationObserver(() => {
            if ($('output').children.length || $('output-status').textContent) {
                show();
            }
        }).observe($('output'), { childList: true });

        new MutationObserver(() => {
            if ($('output-status').textContent) {
                show();
            }
        }).observe($('output-status'), { childList: true, characterData: true, subtree: true });
    }
};

// ----------------------------------------------------------------------------
// Info Modal
// ----------------------------------------------------------------------------

const InfoModal = {
    show() {
        $('info-modal').classList.add('visible');
        document.body.style.overflow = 'hidden';
    },

    hide() {
        $('info-modal').classList.remove('visible');
        document.body.style.overflow = '';
    },

    init() {
        $('info-modal').addEventListener('click', (e) => {
            if (e.target === $('info-modal')) {
                this.hide();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hide();
            }
        });
    }
};

const ExperimentalModal = {
    show() {
        $('experimental-modal').classList.add('visible');
        document.body.style.overflow = 'hidden';
    },

    hide() {
        $('experimental-modal').classList.remove('visible');
        document.body.style.overflow = '';
    },

    init() {
        $('experimental-modal').addEventListener('click', (e) => {
            if (e.target === $('experimental-modal')) {
                this.hide();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hide();
            }
        });
    }
};

// ----------------------------------------------------------------------------
// Python Bridge
// ----------------------------------------------------------------------------

window.setStatus = (state, text) => {
    Status.set(state, text);

    // Re-enable run button when analysis completes
    const btn = $('run-btn');
    if (btn && state === 'ready') {
        btn.disabled = false;
        if (btn.dataset.label) {
            btn.textContent = btn.dataset.label;
        }
    }
};

const Analysis = {
    ready: false,
    pending: null,

    run(params) {
        if (this.ready) {
            window.runAnalysis(params);
        } else {
            this.pending = params;
        }
    }
};

window.onPythonReady = () => {
    Analysis.ready = true;
    Status.set('ready', 'ready');

    // Enable run button if present
    const btn = $('run-btn');
    if (btn) btn.disabled = false;

    // Run pending analysis
    if (Analysis.pending !== null) {
        window.runAnalysis(Analysis.pending);
        Analysis.pending = null;
    }
};

// ----------------------------------------------------------------------------
// App Initialization
// ----------------------------------------------------------------------------

const App = {
    async init() {
        // Handle OAuth callback
        if (location.search.includes('code=')) {
            await Auth.handleCallback();
        }

        if (!Auth.getToken()) return;

        // Show app UI
        $('auth').style.display = 'none';
        $('app').classList.add('visible');

        // Initialize shared components
        OutputPanel.init();
        CodeModal.init();
        InfoModal.init();
        ExperimentalModal.init();

        // Call app-specific init if defined
        if (window.onInit) {
            await window.onInit();
        }

        // Load PyScript
        Status.set('loading', 'loading');
        PyScript.load();
    }
};
