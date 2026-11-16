/* Theme lab (SMUDGY_THEME_LAB=1), not part of normal builds.
   Applies the stored choice on load, then
   builds a floating picker. Choices persist in localStorage. */
(function () {
  const KEY = "smudgy-theme-lab";

  const OPTIONS = {
    code: {
      label: "Code colours",
      values: {
        smudgy: "smudgy (custom, 0 mismatches)",
        solarized: "Solarized (0 mismatches)",
        ayu: "Ayu light / mirage (0)",
        catppuccin: "Catppuccin latte / mocha (1, current)",
        gruvbox: "Gruvbox (2)",
        one: "One light / dark (5)",
        github: "GitHub (10)",
        "tango-nord": "Tango / Nord, old (8)",
      },
    },
    pagebg: {
      label: "Page bg",
      values: {
        graphite: "white / graphite (current)",
        classic: "white / #191921 (old)",
        paper: "warm paper",
        slate: "cool slate",
        catppuccin: "catppuccin base",
      },
    },
    navbg: {
      label: "Nav bar bg",
      values: {
        current: "current (#f6f7fa / #1e1e28)",
        page: "same as page",
        tint: "accent tint",
        deeper: "deeper than page",
        catppuccin: "catppuccin mantle",
      },
    },
    codebg: {
      label: "Code bg",
      values: {
        page: "same as page, border only (current)",
        neutral: "#f6f7fa / #22222d (old)",
        style: "from code style",
        tint: "accent tint",
        deeper: "deeper than page",
        raised: "raised (lighter in dark)",
      },
    },
    emphasis: {
      label: "Token style",
      values: {
        style: "as style defines",
        plain: "plain, no bold/italic (current)",
        comments: "italic comments only",
      },
    },
    accent: {
      label: "Accent",
      values: { blue: "blue (current)", indigo: "indigo", teal: "teal", amber: "amber" },
    },
    caption: {
      label: "Nav captions",
      values: { plain: "plain text (current)", ink: "ink block", accent: "accent block", subtle: "subtle tint" },
    },
    body: {
      label: "Body font",
      values: {
        plex: "IBM Plex Sans (current)",
        inter: "Inter",
        source: "Source Sans 3",
        atkinson: "Atkinson Hyperlegible Next",
        system: "System UI",
      },
    },
    heading: {
      label: "Heading font",
      values: { body: "same as body", serif: "Source Serif 4" },
    },
    mono: {
      label: "Code font",
      values: {
        inconsolata: "Inconsolata",
        plexmono: "IBM Plex Mono (current)",
        jetbrains: "JetBrains Mono",
        sourcecode: "Source Code Pro",
        atkinsonmono: "Atkinson Hyperlegible Mono",
      },
    },
    size: {
      label: "Body size",
      values: { "100%": "100% (16px)", "106.25%": "106% (17px)" },
    },
    codesize: {
      label: "Code size",
      values: { "0.85em": "0.85em (current)", "0.9em": "0.90em", "0.95em": "0.95em", "1em": "1.00em", "1.05em": "1.05em" },
    },
    figures: {
      label: "Dark figures",
      values: { none: "as is", panel: "rounded panel", dim: "dimmed" },
    },
  };

  const DEFAULTS = {
    code: "catppuccin", codebg: "page", pagebg: "graphite", navbg: "current", emphasis: "plain", accent: "blue",
    caption: "plain", body: "plex", heading: "body", mono: "plexmono",
    size: "100%", codesize: "0.85em", figures: "none",
  };

  // [light, light-hover, dark, dark-hover]
  const ACCENTS = {
    blue: ["#3352cc", "#2440b0", "#8ba4f0", "#adc0f7"],
    indigo: ["#4f46e5", "#3f37c9", "#a5b4fc", "#c7d2fe"],
    teal: ["#0f766e", "#0b5f58", "#5ccfc0", "#8ee3d8"],
    amber: ["#b45309", "#92400e", "#f5b546", "#f8cc7a"],
  };
  // background choices: [light, dark]; keys missing here = custom.css values
  const mix = (pct) => `color-mix(in srgb, var(--smudgy-accent) ${pct}%, var(--smudgy-bg))`;
  const PAGE_BG = {
    paper: ["#fbfaf7", "#1c1b19"],
    slate: ["#f8fafc", "#0f1520"],
    classic: ["#ffffff", "#191921"],
    catppuccin: ["#eff1f5", "#1e1e2e"],
  };
  const NAV_BG = {
    page: ["var(--smudgy-bg)", "var(--smudgy-bg)"],
    tint: [mix(5), mix(7)],
    deeper: ["color-mix(in srgb, #000 4%, var(--smudgy-bg))", "color-mix(in srgb, #000 30%, var(--smudgy-bg))"],
    catppuccin: ["#e6e9ef", "#181825"],
  };
  const CODE_BG = {
    neutral: ["#f6f7fa", "#22222d"],
    tint: [mix(4), mix(6)],
    deeper: ["color-mix(in srgb, #000 4%, var(--smudgy-bg))", "color-mix(in srgb, #000 30%, var(--smudgy-bg))"],
    raised: ["color-mix(in srgb, #000 2.5%, var(--smudgy-bg))", "color-mix(in srgb, #fff 6%, var(--smudgy-bg))"],
  };
  const BODY_FONTS = {
    plex: "'IBM Plex Sans'",
    inter: "'Inter'",
    source: "'Source Sans 3'",
    atkinson: "'Atkinson Hyperlegible Next'",
    system: "system-ui",
  };
  const MONO_FONTS = {
    inconsolata: "'Inconsolata'",
    plexmono: "'IBM Plex Mono'",
    jetbrains: "'JetBrains Mono'",
    sourcecode: "'Source Code Pro'",
    atkinsonmono: "'Atkinson Hyperlegible Mono'",
  };

  let state = Object.assign({}, DEFAULTS);
  try { Object.assign(state, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}

  const html = document.documentElement;
  const dyn = document.createElement("style");
  dyn.id = "lab-dyn";
  document.head.appendChild(dyn);

  // Rules for light / dark mode, incl. furo's "auto" via prefers-color-scheme.
  function perMode(light, dark) {
    const B = "html[data-lab] body";
    return `${B}[data-theme="light"]{${light}}
${B}[data-theme="dark"]{${dark}}
@media (prefers-color-scheme: light){${B}[data-theme="auto"]{${light}}}
@media (prefers-color-scheme: dark){${B}[data-theme="auto"]{${dark}}}`;
  }

  function apply() {
    html.setAttribute("data-lab", "");
    html.setAttribute("data-lab-code", state.code);
    const a = ACCENTS[state.accent];
    let css = perMode(
      `--smudgy-accent:${a[0]};--smudgy-accent-hover:${a[1]};`,
      `--smudgy-accent:${a[2]};--smudgy-accent-hover:${a[3]};`
    );
    const B = "html[data-lab] body[data-theme]";
    const mono = `${MONO_FONTS[state.mono]}, ui-monospace, SFMono-Regular, Menlo, monospace`;
    const sans = `${BODY_FONTS[state.body]}, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
    css += `
${B}{--smudgy-font-body:${sans};--smudgy-font-mono:${mono};
  --font-size--normal:${state.size};
  --smudgy-code-size:${state.codesize};
  --smudgy-inline-code-size:calc(${state.codesize} - 0.05em);}`;
    if (state.heading === "serif") {
      css += `${B} h1, ${B} h2, ${B} h3, ${B} h4 {font-family:'Source Serif 4', Georgia, serif;font-weight:600;}`;
    }
    if (state.codebg === "style") {
      css += `${B}{--smudgy-code-bg:var(--lab-style-bg);--smudgy-code-fg:var(--lab-style-fg);}`;
    }
    const bg = (name, table, key) => {
      const v = table[key];
      if (v) css += perMode(`${name}:${v[0]};`, `${name}:${v[1]};`);
    };
    bg("--smudgy-bg", PAGE_BG, state.pagebg);
    bg("--smudgy-surface", NAV_BG, state.navbg);
    bg("--smudgy-code-bg", CODE_BG, state.codebg);
    if (state.emphasis !== "style") {
      css += `${B} .highlight span{font-weight:normal!important;font-style:normal!important;}`;
    }
    if (state.emphasis === "comments") {
      css += `${B} .highlight :is(.c,.c1,.cm,.ch,.cs,.cpf,.sd){font-style:italic!important;}`;
    }
    const cap = {
      ink: [
        "--smudgy-caption-bg:#1f2330;--smudgy-caption-fg:#fff;",
        "--smudgy-caption-bg:#2e3045;--smudgy-caption-fg:#fff;",
      ],
      accent: [
        "--smudgy-caption-bg:var(--smudgy-accent);--smudgy-caption-fg:#fff;",
        "--smudgy-caption-bg:var(--smudgy-accent);--smudgy-caption-fg:#191921;",
      ],
      subtle: [
        "--smudgy-caption-bg:var(--smudgy-accent-soft);--smudgy-caption-fg:var(--smudgy-accent);",
        "--smudgy-caption-bg:var(--smudgy-accent-soft);--smudgy-caption-fg:var(--smudgy-accent);",
      ],
      plain: ["", ""],
    }[state.caption];
    if (cap[0]) css += perMode(cap[0], cap[1]);
    const img = 'article img:not(.inline-title-icon)';
    if (state.figures === "panel") {
      css += `html[data-lab] body[data-theme="dark"] ${img}{border-radius:8px;box-shadow:0 0 0 1px var(--smudgy-border);}
@media (prefers-color-scheme: dark){html[data-lab] body[data-theme="auto"] ${img}{border-radius:8px;box-shadow:0 0 0 1px var(--smudgy-border);}}`;
    } else if (state.figures === "dim") {
      css += `html[data-lab] body[data-theme="dark"] ${img}{filter:brightness(.85) contrast(1.05);border-radius:4px;}
@media (prefers-color-scheme: dark){html[data-lab] body[data-theme="auto"] ${img}{filter:brightness(.85) contrast(1.05);border-radius:4px;}}`;
    }
    dyn.textContent = css;
    localStorage.setItem(KEY, JSON.stringify(state));
  }

  apply();

  document.addEventListener("DOMContentLoaded", () => {
    const panel = document.createElement("div");
    panel.id = "lab-panel";
    if (localStorage.getItem(KEY + "-collapsed") === "1") panel.classList.add("collapsed");
    let rows = "";
    for (const [k, o] of Object.entries(OPTIONS)) {
      const opts = Object.entries(o.values)
        .map(([v, t]) => `<option value="${v}"${state[k] === v ? " selected" : ""}>${t}</option>`)
        .join("");
      rows += `<label>${o.label}<select data-key="${k}">${opts}</select></label>`;
    }
    panel.innerHTML = `<div class="lab-head"><span>Theme lab</span><span class="lab-toggle">▾</span></div>
<div class="lab-body">${rows}
<div class="lab-btns"><button data-act="reset">Reset</button><button data-act="copy">Copy config</button></div>
<textarea readonly></textarea></div>`;
    document.body.appendChild(panel);

    panel.querySelector(".lab-head").addEventListener("click", () => {
      panel.classList.toggle("collapsed");
      localStorage.setItem(KEY + "-collapsed", panel.classList.contains("collapsed") ? "1" : "0");
    });
    panel.querySelectorAll("select").forEach((s) =>
      s.addEventListener("change", () => { state[s.dataset.key] = s.value; apply(); })
    );
    panel.querySelector('[data-act="reset"]').addEventListener("click", () => {
      state = Object.assign({}, DEFAULTS);
      panel.querySelectorAll("select").forEach((s) => (s.value = state[s.dataset.key]));
      apply();
    });
    panel.querySelector('[data-act="copy"]').addEventListener("click", () => {
      const txt = JSON.stringify(state, null, 1);
      const ta = panel.querySelector("textarea");
      ta.value = txt;
      ta.style.display = "block";
      ta.select();
      if (navigator.clipboard) navigator.clipboard.writeText(txt).catch(() => {});
    });
  });
})();
