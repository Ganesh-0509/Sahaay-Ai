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
