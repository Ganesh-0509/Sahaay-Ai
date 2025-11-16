document.addEventListener('DOMContentLoaded', function () {
  async function createPost(text) {
    const res = await fetch('/api/community/post', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    if (res.ok) return res.json();
    if (res.status === 401) {
      alert('Please log in to post.');
      return null;
    }
    const err = await res.text();
    alert('Error: ' + err);
    return null;
  }

  // In-page composer has been removed; modal is used to create posts.

  // Modal handlers
  document.addEventListener('click', async (e) => {
    if (e.target.matches('#open-post-modal')) { const m = document.getElementById('post-modal'); if (m) m.style.display = 'flex'; return; }
    if (e.target.matches('#post-modal-close')) { const m = document.getElementById('post-modal'); if (m) m.style.display = 'none'; return; }
    if (e.target.matches('#modal-post-submit')) {
      const txt = document.getElementById('modal-post-text').value.trim();
      if (!txt) return alert('Please write something.');
      const created = await createPost(txt);
      if (created) {
        const list = document.getElementById('posts-list');
        const el = document.createElement('div');
        el.className = 'card mb-4';
        el.innerHTML = `<div class="meta"><strong>Anonymous</strong> • just now</div><div class="text mt-2">${escapeHtml(txt)}</div>`;
        list.insertBefore(el, list.firstChild);
        document.getElementById('modal-post-text').value = '';
        const m = document.getElementById('post-modal'); if (m) m.style.display = 'none';
      }
      return;
    }
  });

  function escapeHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
});
