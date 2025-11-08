// Sidebar toggle
const menuBtn = document.getElementById('menuBtn');
const sidebar = document.getElementById('sidebar');
const mainContent = document.getElementById('mainContent');
if(menuBtn){
    menuBtn.addEventListener('click', () => {
        sidebar.classList.toggle('-ml-64');
        sidebar.classList.toggle('sidebar-open');
        if (window.innerWidth >= 768) mainContent.classList.toggle('main-shrink');
    });
}

window.addEventListener('resize', () => {
    if (window.innerWidth < 768) {
        sidebar.classList.add('-ml-64');
        sidebar.classList.remove('sidebar-open');
        mainContent.classList.remove('main-shrink');
    }
});

// Profile dropdown
const profileBtn = document.getElementById('profileBtn');
const profileDropdown = document.getElementById('profileDropdown');
if(profileBtn){
    profileBtn.addEventListener('click', () => { profileDropdown.classList.toggle('hidden'); });
    document.addEventListener('click', e => {
        if (!profileBtn.contains(e.target) && !profileDropdown.contains(e.target))
            profileDropdown.classList.add('hidden');
    });
}

// Dark Mode
const darkModeBtn = document.getElementById('darkModeBtn');
if(darkModeBtn){
    if (localStorage.getItem('darkMode') === 'enabled') {
        document.documentElement.classList.add('dark');
        darkModeBtn.textContent = "☀️";
    } else { darkModeBtn.textContent = "🌙"; }

    darkModeBtn.addEventListener('click', () => {
        document.documentElement.classList.toggle('dark');
        darkModeBtn.textContent = document.documentElement.classList.contains('dark') ? "☀️" : "🌙";
        localStorage.setItem('darkMode', document.documentElement.classList.contains('dark') ? 'enabled' : 'disabled');
    });
}

// Language selection modal and dynamic translation loading
function setLanguage(lang) {
  localStorage.setItem('userLang', lang);
  document.getElementById('langModal').style.display = 'none';
  location.reload();
}

function getLanguage() {
  return localStorage.getItem('userLang') || 'en';
}

function detectAndSetLanguage() {
  const lang = getLanguage();
  fetch(`/translations/${lang}.json`)
    .then(res => res.json())
    .then(dict => {
      for (const key in dict) {
        const el = document.querySelector(`[data-i18n='${key}']`);
        if (el) el.textContent = dict[key];
      }
    });
}

document.addEventListener('DOMContentLoaded', () => {
  if (!localStorage.getItem('userLang')) {
    document.getElementById('langModal').style.display = 'flex';
  } else {
    document.getElementById('langModal').style.display = 'none';
  }
  detectAndSetLanguage();
});

function changeLanguage(lang) {
  localStorage.setItem('userLang', lang);
  fetch('/set_language/' + lang)
    .then(() => location.reload());
}

// Lazy load images
function lazyLoadImages() {
  const images = document.querySelectorAll('img[loading="lazy"]');
  images.forEach(img => {
    if ('IntersectionObserver' in window) {
      let observer = new IntersectionObserver((entries, obs) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.src = entry.target.dataset.src;
            obs.unobserve(entry.target);
          }
        });
      });
      observer.observe(img);
    } else {
      img.src = img.dataset.src;
    }
  });
}
document.addEventListener('DOMContentLoaded', lazyLoadImages);
