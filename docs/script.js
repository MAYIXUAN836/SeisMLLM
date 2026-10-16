const copyButton = document.getElementById('copy-citation');
const citation = document.getElementById('bibtex');
const copyStatus = document.getElementById('copy-status');

copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyButton.querySelector('span').textContent = 'Copied';
    copyStatus.textContent = 'BibTeX copied to clipboard.';
    window.setTimeout(() => {
      copyButton.querySelector('span').textContent = 'Copy BibTeX';
    }, 2500);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citation);
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Citation selected. Press Ctrl+C or Command+C to copy.';
  }
});

// Reveal each research block once as it enters the viewport.
if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches && 'IntersectionObserver' in window) {
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.14 });

  document.querySelectorAll('#overview .reading-width, #method .section-heading, .method-step, .dataset-layout, #results .section-heading, .result, .result-table-block, #resources .container, .citation-section').forEach((block) => {
    if (block.getBoundingClientRect().top >= window.innerHeight) {
      block.classList.add('reveal');
      revealObserver.observe(block);
    }
  });
}
