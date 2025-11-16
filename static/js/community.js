async function apiGet(path) {
  try {
    const resp = await fetch(path, {credentials: 'same-origin'});
    const j = await resp.json().catch(() => ({ok: false, error: 'invalid json'}));
    return j;
  } catch (e) {
    return {ok: false, error: String(e)};
  }
}

async function apiPost(path, body) {
  try {
    const resp = await fetch(path, {
      method: 'POST',
      credentials: 'same-origin',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const j = await resp.json().catch(() => ({ok: false, error: 'invalid json'}));
    return j;
  } catch (e) {
    return {ok: false, error: String(e)};
  }
}

function renderPost(post, highlight=false) {
  const el = document.createElement('div');
  el.className = 'community-post';
  if (highlight) el.style.border = '1px solid #007bff';
  el.innerHTML = `
    <div class="meta"><strong>${post.author_display}</strong> • ${post.created_at || ''}</div>
    <div class="text">${escapeHtml(post.text)}</div>
    <div class="actions">
      <button class="react" data-id="${post.id}" data-reaction="heart">❤️ <span class="count">${post.reactions_count.heart || 0}</span></button>
      <button class="react" data-id="${post.id}" data-reaction="thumbs_up">👍 <span class="count">${post.reactions_count.thumbs_up || 0}</span></button>
    </div>
  `;
  return el;
}

function renderPoll(poll) {
  const el = document.createElement('div');
  el.className = 'community-poll';
  let optionsHtml = '';
  for (const [optId, opt] of Object.entries(poll.options || {})) {
    optionsHtml += `<div class="opt"><button class="vote" data-poll="${poll.id}" data-opt="${optId}">${escapeHtml(opt.text)} <span class="count">(${opt.votes || 0})</span></button></div>`;
  }
  el.innerHTML = `
    <div class="meta"><strong>${poll.author_display}</strong> • ${poll.created_at || ''}</div>
    <div class="question">${escapeHtml(poll.question)}</div>
    <div class="options">${optionsHtml}</div>
  `;
  return el;
}

function escapeHtml(s) {
  if (!s) return '';
  return s.replace(/[&<>'"]/g, function(c) { return ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;','\'':'&#39;'}[c]); });
}

async function loadFeed() {
  const res = await apiGet('/api/community/feed');
  if (!res || !res.ok) {
    const postsEl = document.getElementById('posts');
    const pollsEl = document.getElementById('polls');
    postsEl.innerHTML = '<div class="muted">Please log in to view community content.</div>';
    pollsEl.innerHTML = '';
    return;
  }
  const postsEl = document.getElementById('posts');
  const pollsEl = document.getElementById('polls');
  const myPostsEl = document.getElementById('my-posts');
  const myPollsEl = document.getElementById('my-polls');
  postsEl.innerHTML = '';
  pollsEl.innerHTML = '';
  myPostsEl.innerHTML = '<h4>My Posts</h4>';
  myPollsEl.innerHTML = '<h4>My Polls</h4>';

  // get current user id from a meta tag if provided, else server will mark posts with author_uid
  const currentUserId = window.CURRENT_USER_ID || null;

  if (!res.posts || res.posts.length === 0) {
    postsEl.innerHTML = '<div class="muted">No posts yet — be the first to post.</div>';
  } else {
    res.posts.forEach(p => {
      postsEl.appendChild(renderPost(p));
      if (p.author_uid && currentUserId && p.author_uid === currentUserId) {
        myPostsEl.appendChild(renderPost(p, true));
      }
    });
  }

  if (!res.polls || res.polls.length === 0) {
    pollsEl.innerHTML = '<div class="muted">No polls yet — create one.</div>';
  } else {
    res.polls.forEach(p => {
      pollsEl.appendChild(renderPoll(p));
      if (p.author_uid && currentUserId && p.author_uid === currentUserId) {
        myPollsEl.appendChild(renderPoll(p, true));
      }
    });
  }
}

document.addEventListener('click', async (e) => {
  if (e.target.matches('#post-submit') || e.target.closest && e.target.closest('#post-submit')) {
    const text = document.getElementById('post-text').value;
    await apiPost('/api/community/post', {text});
    document.getElementById('post-text').value = '';
    loadFeed();
  }
  if (e.target.matches('#poll-submit') || e.target.closest && e.target.closest('#poll-submit')) {
    const q = document.getElementById('poll-question').value;
    const opts = Array.from(document.querySelectorAll('.poll-opt')).map(i => i.value).filter(Boolean);
    await apiPost('/api/community/poll', {question: q, options: opts});
    document.getElementById('poll-question').value = '';
    document.querySelectorAll('.poll-opt').forEach((n,i) => { if (i>1) n.remove(); else n.value = ''; });
    loadFeed();
  }
  if (e.target.matches('#open-post-modal')) {
    const m = document.getElementById('post-modal'); if (m) m.style.display = 'flex'; return;
  }
  if (e.target.matches('#post-modal-close')) {
    const m = document.getElementById('post-modal'); if (m) m.style.display = 'none'; return;
  }
  if (e.target.matches('#modal-post-submit')) {
    const text = document.getElementById('modal-post-text').value;
    if (!text || !text.trim()) return alert('Please enter content');
    await apiPost('/api/community/post', {text});
    document.getElementById('modal-post-text').value = '';
    const m = document.getElementById('post-modal'); if (m) m.style.display = 'none';
    if (window.loadFeed) window.loadFeed();
    return;
  }
  const reactBtn = e.target.closest ? e.target.closest('.react') : null;
  if (reactBtn) {
    const postId = reactBtn.getAttribute('data-id');
    const reaction = reactBtn.getAttribute('data-reaction');
    await apiPost('/api/community/post/react', {post_id: postId, reaction});
    loadFeed();
    return;
  }
  const voteBtn = e.target.closest ? e.target.closest('.vote') : null;
  if (voteBtn) {
    const pollId = voteBtn.getAttribute('data-poll');
    const optId = voteBtn.getAttribute('data-opt');
    await apiPost('/api/community/poll/vote', {poll_id: pollId, option_id: optId});
    loadFeed();
    return;
  }
});

// initial load
loadFeed();

// Welcome modal: show on first visit
function showWelcomeIfFirstVisit() {
  try {
    const key = 'community_welcome_shown_v1';
    if (!localStorage.getItem(key)) {
      const modal = document.getElementById('community-welcome');
      if (modal) modal.style.display = 'flex';
      // wire up buttons
      const postBtn = document.getElementById('welcome-post');
      const pollBtn = document.getElementById('welcome-poll');
      const closeBtn = document.getElementById('welcome-close');
      if (postBtn) postBtn.addEventListener('click', () => { document.getElementById('community-welcome').style.display='none'; document.getElementById('posts-col').scrollIntoView(); localStorage.setItem(key, '1'); });
      if (pollBtn) pollBtn.addEventListener('click', () => { document.getElementById('community-welcome').style.display='none'; document.getElementById('polls-col').scrollIntoView(); localStorage.setItem(key, '1'); });
      if (closeBtn) closeBtn.addEventListener('click', () => { document.getElementById('community-welcome').style.display='none'; localStorage.setItem(key, '1'); });
    }
  } catch (e) {
    // ignore
  }
}

showWelcomeIfFirstVisit();
