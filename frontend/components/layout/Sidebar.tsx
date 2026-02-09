'use client';

import { usePathname, useRouter } from 'next/navigation';
import { useState } from 'react';
import { useAuthStore } from '@/store/authStore';

interface NavLink {
    href: string;
    label: string;
    icon: string;
}

const navLinks: NavLink[] = [
    { href: '/dashboard', label: 'Overview', icon: '⭐' },
    { href: '/mood', label: 'Mood Journal', icon: '📔' },
    { href: '/tools', label: 'Coping Tools', icon: '🧘' },
    { href: '/chat', label: 'AI Support Chat', icon: '💬' },
    { href: '/analytics', label: 'Analytics', icon: '📊' },
    { href: '/settings', label: 'Settings', icon: '⚙️' },
];

export default function Sidebar() {
    const pathname = usePathname();
    const router = useRouter();
    const logout = useAuthStore((state) => state.logout);
    const user = useAuthStore((state) => state.user);
    const [isMobileOpen, setIsMobileOpen] = useState(false);

    const handleLogout = async () => {
        await logout();
        router.push('/login');
    };

    const handleNavClick = (href: string) => {
        router.push(href);
        setIsMobileOpen(false);
    };

    return (
        <>
            {/* Mobile Menu Button - STRICTLY hidden on desktop (sm:hidden) */}
            <button
                id="menuBtn"
                onClick={() => setIsMobileOpen(!isMobileOpen)}
                className="sm:hidden fixed top-4 left-4 z-50 p-2 rounded-lg bg-theme-panel backdrop-blur-lg text-theme-text-main hover:scale-110 active:scale-95 transition-transform shadow-lg"
                aria-label="Toggle menu"
            >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
                </svg>
            </button>

            {/* Sidebar - Fixed width, handled via translate */}
            {/* Logic: 
                - Mobile Default: -translate-x-full (Hidden)
                - Mobile Open: translate-x-0 (Visible)
                - Desktop (sm): translate-x-0 (Always Visible) 
            */}
            <aside
                id="sidebar"
                className={`fixed inset-y-0 left-0 w-64 bg-theme-panel backdrop-blur-xl border-r border-theme-primary/20 shadow-2xl p-6 flex flex-col transition-transform duration-300 ease-in-out z-40 
                ${isMobileOpen ? 'translate-x-0' : '-translate-x-full sm:translate-x-0'}`}
            >
                {/* Logo & Title */}
                <div className="flex items-center mb-8">
                    <div className="h-10 w-10 mr-3 rounded-full bg-gradient-to-br from-theme-primary to-theme-primary-light flex items-center justify-center text-xl shadow-primary-glow">
                        🧘‍♀️
                    </div>
                    <h2 className="text-2xl font-bold text-white">Sahaay AI</h2>
                </div>

                {/* User Info (Optional - matches Flask design which usually has user info or icon) */}
                {user && (
                    <div className="mb-6 p-3 rounded-xl bg-theme-hover/20 border border-theme-primary/10">
                        <p className="text-xs text-theme-text-subtle">Welcome back,</p>
                        <p className="font-bold text-theme-text-main truncate text-sm">{user.username || user.name || 'Friend'}</p>
                    </div>
                )}

                {/* Navigation Links */}
                <nav className="space-y-2 flex-1 relative overflow-y-auto max-h-[calc(100vh-250px)] no-scrollbar">
                    {navLinks.map((link) => {
                        const isActive = pathname === link.href;
                        return (
                            <button
                                key={link.href}
                                onClick={() => handleNavClick(link.href)}
                                className={`sidebar-link w-full text-left group ${isActive ? 'sidebar-active' : 'text-theme-text-subtle hover:text-theme-accent'}`}
                            >
                                <span className={`text-xl transition-transform group-hover:scale-110 ${isActive ? 'scale-110' : ''}`}>{link.icon}</span>
                                <span className="font-medium">{link.label}</span>
                            </button>
                        );
                    })}
                </nav>

                {/* Logout Button */}
                <div className="mt-auto pt-4 border-t border-theme-primary/20">
                    <button
                        onClick={handleLogout}
                        className="sidebar-link w-full text-left text-theme-text-subtle hover:bg-theme-danger/20 hover:text-white group"
                    >
                        <span className="text-xl group-hover:-translate-x-1 transition-transform">🚪</span>
                        <span>Logout</span>
                    </button>
                </div>
            </aside>

            {/* Mobile Overlay - Only visible on mobile when menu is open */}
            {isMobileOpen && (
                <div
                    onClick={() => setIsMobileOpen(false)}
                    className="md:hidden fixed inset-0 bg-black/50 z-30 backdrop-blur-sm"
                />
            )}
        </>
    );
}
