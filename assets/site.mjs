export function searchPages(pages, query) {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  return terms.length ? pages.filter(p => terms.every(t => `${p.title} ${p.text}`.toLowerCase().includes(t))).slice(0, 8) : [];
}

export function confusionCounts(rows, model, threshold) {
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('Invalid threshold');
  const counts = { tn: 0, fp: 0, fn: 0, tp: 0 };
  for (const row of rows) {
    const p = row[model];
    if (!Number.isFinite(p) || p < 0 || p > 1 || ![0, 1].includes(row.malignant)) throw new Error('Invalid prediction');
    counts[row.malignant ? (p >= threshold ? 'tp' : 'fn') : (p >= threshold ? 'fp' : 'tn')]++;
  }
  return counts;
}

function initialize() {
  const announce = text => { document.querySelector('#interaction-status').textContent = text; };
  const search = document.querySelector('#study-search');
  const results = document.querySelector('#search-results');
  const pages = JSON.parse(document.querySelector('#search-data').textContent);
  search.addEventListener('input', () => {
    const matches = searchPages(pages, search.value);
    results.replaceChildren();
    for (const page of matches) {
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = new URL(page.path, document.querySelector('.brand').href).href;
      link.textContent = page.title;
      item.append(link);
      results.append(item);
    }
    document.querySelector('#search-status').textContent = search.value.trim() ? `${matches.length} matching pages shown (maximum 8).` : '';
  });
  document.querySelector('#clear-search').addEventListener('click', () => {
    search.value = '';
    search.dispatchEvent(new Event('input'));
    search.focus();
  });
  const theme = document.querySelector('#theme-toggle');
  const setTheme = value => {
    document.documentElement.dataset.theme = value;
    theme.setAttribute('aria-pressed', String(value === 'night'));
    theme.textContent = value === 'night' ? 'Day palette' : 'Night palette';
  };
  try { setTheme(localStorage.getItem('reading-theme') === 'night' ? 'night' : 'day'); } catch { setTheme('day'); }
  theme.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'night' ? 'day' : 'night';
    setTheme(next);
    try { localStorage.setItem('reading-theme', next); } catch { announce('Palette changed for this page.'); }
  });
  for (const block of document.querySelectorAll('main pre')) {
    const code = block.querySelector('code');
    if (!code) continue;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-button';
    button.textContent = 'Copy code';
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = 'Copied';
        announce('Code copied to clipboard.');
        setTimeout(() => { button.textContent = 'Copy code'; }, 1600);
      } catch {
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        announce('Clipboard unavailable. The code is selected for manual copying.');
      }
    });
    block.prepend(button);
  }
  const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
  for (const picture of document.querySelectorAll('picture.protocol-motion')) {
    const image = picture.querySelector('img');
    const motion = image.getAttribute('src');
    const still = image.dataset.still;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'protocol-play';
    let timer;
    const stop = () => {
      clearTimeout(timer);
      image.src = still;
      picture.dataset.playing = 'false';
      button.textContent = motionPreference.matches ? 'Reduced motion: static diagram' : 'Play workflow';
      button.disabled = motionPreference.matches;
    };
    stop();
    button.addEventListener('click', () => {
      if (picture.dataset.playing === 'true') { stop(); return; }
      picture.dataset.playing = 'true';
      image.src = `${motion}?replay=${Date.now()}`;
      button.textContent = 'Pause workflow';
      timer = setTimeout(stop, 4600);
    });
    motionPreference.addEventListener('change', stop);
    picture.insertAdjacentElement('afterend', button);
  }
  const dialog = document.querySelector('#figure-dialog');
  const expanded = dialog.querySelector('img');
  const zoom = document.querySelector('#figure-zoom');
  let lastFigureButton;
  for (const image of document.querySelectorAll('main img')) {
    if (!image.src.includes('/assets/figures/')) continue;
    const controls = document.createElement('div');
    controls.className = 'figure-controls';
    const inspect = document.createElement('button');
    inspect.type = 'button';
    inspect.textContent = 'Inspect figure';
    inspect.setAttribute('aria-label', `Inspect figure: ${image.alt}`);
    const download = document.createElement('a');
    download.href = image.src;
    download.download = image.src.split('/').pop();
    download.className = 'button secondary';
    download.textContent = `Download ${new URL(image.src).pathname.split('.').pop().toUpperCase()}`;
    inspect.addEventListener('click', () => {
      lastFigureButton = inspect;
      expanded.src = image.currentSrc || image.src;
      expanded.alt = image.alt;
      document.querySelector('#figure-caption').textContent = image.alt;
      zoom.value = '100';
      expanded.style.width = '100%';
      document.querySelector('#zoom-value').textContent = '100%';
      dialog.showModal();
    });
    controls.append(inspect, download);
    image.parentElement.insertAdjacentElement('afterend', controls);
  }
  zoom.addEventListener('input', () => {
    expanded.style.width = `${zoom.value}%`;
    document.querySelector('#zoom-value').textContent = `${zoom.value}%`;
  });
  document.querySelector('#close-figure').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => lastFigureButton?.focus());
  if (document.querySelector('#threshold-explorer')) {
    const data = JSON.parse(document.querySelector('#prediction-data').textContent);
    const model = document.querySelector('#model-choice');
    const threshold = document.querySelector('#decision-threshold');
    const update = () => {
      const counts = confusionCounts(data.rows, model.value, Number(threshold.value));
      document.querySelector('#threshold-value').textContent = Number(threshold.value).toFixed(2);
      for (const [key, count] of Object.entries(counts)) {
        document.querySelector(`#count-${key}`).textContent = String(count);
        document.querySelector(`#bar-${key}`).style.width = `${100 * count / data.rows.length}%`;
      }
      document.querySelector('#decision-summary').textContent = `${data.rows.length} held-out records; ${counts.fn} false negatives and ${counts.fp} false positives at threshold ${Number(threshold.value).toFixed(2)}.`;
    };
    model.addEventListener('change', update);
    threshold.addEventListener('input', update);
    document.querySelector('#reset-threshold').addEventListener('click', () => {
      threshold.value = '.50'; model.value = data.selected; update();
    });
    update();
  }
  const progress = document.querySelector('#reading-progress');
  let pending = false;
  const updateProgress = () => {
    const distance = document.documentElement.scrollHeight - innerHeight;
    progress.value = distance > 0 ? Math.min(1, Math.max(0, scrollY / distance)) : 1;
    pending = false;
  };
  addEventListener('scroll', () => {
    if (!pending) { pending = true; requestAnimationFrame(updateProgress); }
  }, { passive: true });
  addEventListener('resize', updateProgress);
  updateProgress();
}

if (typeof document !== 'undefined') initialize();
