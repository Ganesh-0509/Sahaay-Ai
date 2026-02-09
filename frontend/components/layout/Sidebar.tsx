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
    { href: '/dashboard', label: 'Overview', icon: '🏠' },
    { href: '/mood', label: 'Mood Journal', icon: '📔' },
    { href: '/tools', label: 'Coping Tools', icon: '🧘' },
    { href: '/community', label: 'Community support', icon: '💬' },
    { href: '/analytics', label: 'Analytics', icon: '📊' },
    { href: '/settings', label: 'Settings', icon: '⚙' },
];

export default function Sidebar() {
    const pathname = usePathname();
    const router = useRouter();
    const logout = useAuthStore((state) => state.logout);
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
                className="md:hidden p-2 rounded-lg bg-theme-panel backdrop-blur-lg text-white mb-4 fixed top-4 left-4 z-50 hover:scale-110 active:scale-95 transition-transform"
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
                className={`fixed inset-y-0 left-0 w-64 bg-theme-panel backdrop-blur-xl border-r border-theme-primary/20 shadow-2xl p-6 flex flex-col transition-transform duration-500 ease-in-out z-40 
                ${isMobileOpen ? 'translate-x-0' : '-translate-x-full sm:translate-x-0'}`}
            >
                {/* Logo & Title */}
                <div className="flex items-center mb-8">
                    <img
                        src="/static/logo.png"
                        alt="Sahaay-AI Logo"
                        className="h-12 w-12 mr-3 rounded-full"
                        loading="lazy"
                        onError={(event) => {
                            (event.currentTarget as HTMLImageElement).src = '/static/logo.svg';
                        }}
                    />
                    <h2 className="text-3xl font-bold text-white">Sahaay AI</h2>
                </div>

                {/* Navigation Links */}
                <nav className="space-y-2 flex-1">
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
                <div className="mt-auto pt-4">
                    <button
                        onClick={handleLogout}
                        className="sidebar-link w-full text-left text-theme-text-subtle hover:bg-red-500/20 hover:text-white group"
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
