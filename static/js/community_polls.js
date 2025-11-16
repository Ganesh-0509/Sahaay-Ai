// run after DOM ready; expose a function to decorate polls so it can be re-run
function decoratePollCards() {
  const submit = document.getElementById('poll-submit');

  // When creating a poll
  submit?.addEventListener('click', async () => {
    const q = document.getElementById('poll-question').value.trim();
  // Composer uses inputs with class 'poll-opt' in the main composer
  const opts = Array.from(document.querySelectorAll('.poll-opt, .poll-option')).map(i => i.value.trim()).filter(Boolean);
    if (!q) return alert('Please enter a question.');
    if (opts.length < 2) return alert('Please provide at least 2 options.');

    const res = await fetch('/api/community/poll', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, options: opts })
    });
    if (res.ok) {
      alert('Poll created.');
      location.reload();
    } else if (res.status === 401) {
      alert('Please log in to create polls.');
    } else {
      alert('Error creating poll');
    }
  });

  // Make listed polls selectable and send vote
  document.querySelectorAll('#polls-list .card').forEach(card => {
    const pollId = card.getAttribute('data-poll-id');
    const options = card.querySelectorAll('.options li');
    // decorate options to include radio and a vote button
    options.forEach((li, idx) => {
      const text = li.textContent || '';
      // Clear and re-render with radio + label
      const optText = li.getAttribute('data-option-text') || text.split('—')[0].trim();
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = `poll_${pollId}`;
  radio.value = li.getAttribute('data-option-id') || (`opt${idx}`);
      const label = document.createElement('label');
      label.appendChild(radio);
      label.appendChild(document.createTextNode(' ' + optText));
      // append votes span
      const count = li.querySelector('.count')?.textContent || '0';
      const span = document.createElement('span');
      span.className = 'count';
      span.textContent = ' — ' + count + ' votes';

      li.innerHTML = '';
      li.appendChild(label);
      li.appendChild(span);
    });

    // Add vote button
    const voteBtn = document.createElement('button');
    voteBtn.textContent = 'Vote';
    voteBtn.className = 'btn-primary';
    voteBtn.style.marginTop = '8px';
    voteBtn.addEventListener('click', async () => {
      const selected = card.querySelector(`input[name="poll_${pollId}"]:checked`);
      if (!selected) return alert('Please select an option to vote.');
  const optionId = selected.value;
      const res = await fetch('/api/community/poll/vote', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ poll_id: pollId, option_id: optionId })
      });
      if (res.ok) {
        const data = await res.json();
        alert('Vote recorded');
        location.reload();
      } else if (res.status === 401) {
        alert('Please log in to vote.');
      } else {
        const txt = await res.text();
        alert('Vote error: ' + txt);
      }
    });
    card.appendChild(voteBtn);
  });
}

document.addEventListener('DOMContentLoaded', function () {
  decoratePollCards();
});

// Modal handlers for poll creation
document.addEventListener('click', async (e) => {
  if (e.target.matches('#open-poll-modal')) {
    const m = document.getElementById('poll-modal'); if (m) m.style.display = 'flex';
    return;
  }
  if (e.target.matches('#poll-modal-close')) {
    const m = document.getElementById('poll-modal'); if (m) m.style.display = 'none';
    return;
  }
  if (e.target.matches('#modal-add-option')) {
    const container = document.getElementById('modal-poll-options');
    const input = document.createElement('input');
    input.className = 'modal-poll-option w-full p-2 mb-2';
    input.placeholder = 'Option';
    container.appendChild(input);
    return;
  }
  if (e.target.matches('#modal-poll-submit')) {
    const q = document.getElementById('modal-poll-question').value.trim();
    const opts = Array.from(document.querySelectorAll('.modal-poll-option')).map(i => i.value.trim()).filter(Boolean);
    if (!q) return alert('Please enter a question.');
    if (opts.length < 2) return alert('Please provide at least 2 options.');
    const res = await fetch('/api/community/poll', {
      method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ question: q, options: opts })
    });
    if (res.ok) {
      alert('Poll created.');
      const m = document.getElementById('poll-modal'); if (m) m.style.display = 'none';
      // clear inputs
      document.getElementById('modal-poll-question').value = '';
      document.querySelectorAll('.modal-poll-option').forEach((n,i) => { if (i>1) n.remove(); else n.value = ''; });
      // reload feed if available
      if (window.loadFeed) window.loadFeed();
    } else {
      const txt = await res.text(); alert('Error creating poll: ' + txt);
    }
    return;
  }
});
