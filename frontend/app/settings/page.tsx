'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import Layout from '@/components/layout/Layout';

export default function SettingsPage() {
    const [activeTab, setActiveTab] = useState('account');
    const [language, setLanguage] = useState('en');
    const [notifications, setNotifications] = useState(true);
    const [theme, setTheme] = useState('dark');
    const [saveStatus, setSaveStatus] = useState('');

    const user = useAuthStore((state) => state.user);
    const authLoading = useAuthStore((state) => state.isLoading);
    const checkAuth = useAuthStore((state) => state.checkAuth);
    const router = useRouter();

    useEffect(() => {
        checkAuth();
    }, [checkAuth]);

    useEffect(() => {
        if (authLoading) return;
        if (!user) {
            router.push('/login');
            return;
        }
    }, [user, authLoading]);

    const handleSaveSettings = async () => {
        setSaveStatus('Saving...');
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        setSaveStatus('Saved ✓');
        setTimeout(() => setSaveStatus(''), 2000);
    };

    if (authLoading || !user) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="loading-shimmer h-8 w-48 rounded-lg"></div>
            </div>
        );
    }

    const tabs = [
        { id: 'account', label: 'Account', icon: '👤' },
        { id: 'preferences', label: 'Preferences', icon: '⚙️' },
        { id: 'privacy', label: 'Privacy', icon: '🔒' },
        { id: 'about', label: 'About', icon: 'ℹ️' },
    ];

    return (
        <Layout>
            {/* Header */}
            <div className="mb-8 animate-fade-in-up">
                <h1 className="text-4xl font-bold text-theme-text-main mb-2">
                    Settings ⚙️
                </h1>
                <p className="text-theme-text-subtle text-lg">
                    Customize your Sahaay AI experience
                </p>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 mb-8 overflow-x-auto">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-6 py-3 rounded-xl font-semibold transition-all whitespace-nowrap ${activeTab === tab.id
                                ? 'bg-gradient-to-r from-theme-primary to-theme-primary-light text-white shadow-primary-glow'
                                : 'bg-theme-hover/30 text-theme-text-subtle hover:bg-theme-hover/50'
                            }`}
                    >
                        <span className="mr-2">{tab.icon}</span>
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Account Tab */}
            {activeTab === 'account' && (
                <div className="space-y-6 animate-fade-in-up">
                    <div className="glass-card">
                        <h2 className="text-2xl font-bold text-theme-text-main mb-6">Account Information</h2>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-theme-text-subtle mb-2">
                                    Username
                                </label>
                                <div className="input bg-theme-hover/30 cursor-not-allowed">
                                    {user.username || user.name}
                                </div>
                                <p className="text-xs text-theme-text-muted mt-1">Username cannot be changed</p>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-theme-text-subtle mb-2">
                                    Email Address
                                </label>
                                <input
                                    type="email"
                                    placeholder="your@email.com"
                                    className="input"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-theme-text-subtle mb-2">
                                    Password
                                </label>
                                <button className="btn btn-outline">
                                    Change Password
                                </button>
                            </div>
                        </div>
                    </div>

                    <div className="glass-card border-2 border-theme-danger/30">
                        <h3 className="text-xl font-bold text-theme-danger mb-4">Danger Zone</h3>
                        <div className="space-y-3">
                            <button className="btn btn-outline text-theme-danger border-theme-danger/50 hover:bg-theme-danger/10 w-full md:w-auto">
                                Export My Data
                            </button>
                            <button className="btn btn-outline text-theme-danger border-theme-danger/50 hover:bg-theme-danger/10 w-full md:w-auto ml-0 md:ml-3">
                                Delete Account
                            </button>
                        </div>
                        <p className="text-sm text-theme-text-muted mt-4">
                            These actions are permanent and cannot be undone.
                        </p>
                    </div>
                </div>
            )}

            {/* Preferences Tab */}
            {activeTab === 'preferences' && (
                <div className="space-y-6 animate-fade-in-up">
                    <div className="glass-card">
                        <h2 className="text-2xl font-bold text-theme-text-main mb-6">App Preferences</h2>
                        <div className="space-y-6">
                            <div>
                                <label className="block text-sm font-medium text-theme-text-subtle mb-3">
                                    Language
                                </label>
                                <select
                                    value={language}
                                    onChange={(e) => setLanguage(e.target.value)}
                                    className="input cursor-pointer"
                                >
                                    <option value="en">English</option>
                                    <option value="hi">हिन्दी (Hindi)</option>
                                    <option value="ta">தமிழ் (Tamil)</option>
                                    <option value="te">తెలుగు (Telugu)</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-theme-text-subtle mb-3">
                                    Theme
                                </label>
                                <div className="grid grid-cols-3 gap-3">
                                    {['light', 'dark', 'auto'].map((t) => (
                                        <button
                                            key={t}
                                            onClick={() => setTheme(t)}
                                            className={`p-4 rounded-xl border-2 transition-all capitalize ${theme === t
                                                    ? 'border-theme-primary bg-theme-primary/20'
                                                    : 'border-theme-hover/30 bg-theme-hover/10 hover:border-theme-hover/50'
                                                }`}
                                        >
                                            {t === 'light' && '☀️'} {t === 'dark' && '🌙'} {t === 'auto' && '🔄'}
                                            <div className="text-sm mt-2">{t}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="flex items-center justify-between p-4 rounded-xl bg-theme-hover/20">
                                <div>
                                    <div className="font-semibold text-theme-text-main">Daily Reminders</div>
                                    <div className="text-sm text-theme-text-subtle">Get reminded to check in daily</div>
                                </div>
                                <button
                                    onClick={() => setNotifications(!notifications)}
                                    className={`relative w-14 h-8 rounded-full transition-colors ${notifications ? 'bg-theme-primary' : 'bg-theme-hover/50'
                                        }`}
                                >
                                    <div className={`absolute top-1 left-1 w-6 h-6 rounded-full bg-white transition-transform ${notifications ? 'translate-x-6' : 'translate-x-0'
                                        }`}></div>
                                </button>
                            </div>
                        </div>

                        <div className="mt-6 flex items-center gap-3">
                            <button onClick={handleSaveSettings} className="btn btn-primary">
                                Save Preferences
                            </button>
                            {saveStatus && (
                                <span className="text-theme-success font-semibold">{saveStatus}</span>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Privacy Tab */}
            {activeTab === 'privacy' && (
                <div className="space-y-6 animate-fade-in-up">
                    <div className="glass-card">
                        <h2 className="text-2xl font-bold text-theme-text-main mb-6">Privacy & Security</h2>
                        <div className="space-y-6">
                            <div className="p-4 rounded-xl bg-theme-hover/20 border border-theme-primary/20">
                                <div className="flex items-start gap-3">
                                    <div className="text-3xl">🔒</div>
                                    <div>
                                        <h3 className="font-semibold text-theme-text-main mb-2">Your Data is Private</h3>
                                        <p className="text-sm text-theme-text-subtle">
                                            All your conversations and mood data are encrypted and stored securely.
                                            We never share your personal information with third parties.
                                        </p>
                                    </div>
                                </div>
                            </div>

                            <div className="p-4 rounded-xl bg-theme-hover/20 border border-theme-primary/20">
                                <div className="flex items-start gap-3">
                                    <div className="text-3xl">🤖</div>
                                    <div>
                                        <h3 className="font-semibold text-theme-text-main mb-2">AI Conversations</h3>
                                        <p className="text-sm text-theme-text-subtle">
                                            Your chats with the AI are used only to provide support and improve your experience.
                                            They are not used for marketing or sold to third parties.
                                        </p>
                                    </div>
                                </div>
                            </div>

                            <div>
                                <h3 className="font-semibold text-theme-text-main mb-3">Data Controls</h3>
                                <div className="space-y-3">
                                    <button className="btn btn-outline w-full md:w-auto">
                                        Download My Data
                                    </button>
                                    <button className="btn btn-outline w-full md:w-auto ml-0 md:ml-3">
                                        Clear Chat History
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* About Tab */}
            {activeTab === 'about' && (
                <div className="space-y-6 animate-fade-in-up">
                    <div className="glass-card text-center">
                        <div className="text-7xl mb-6">🧘‍♀️</div>
                        <h2 className="text-3xl font-bold text-theme-text-main mb-3">Sahaay AI</h2>
                        <p className="text-theme-text-subtle mb-6">
                            Your personal mental wellness companion
                        </p>
                        <div className="inline-block px-6 py-2 rounded-full bg-theme-primary/20 text-theme-primary font-semibold">
                            Version 1.0.0
                        </div>
                    </div>

                    <div className="glass-card">
                        <h3 className="text-xl font-bold text-theme-text-main mb-4">About This App</h3>
                        <p className="text-theme-text-subtle mb-4">
                            Sahaay AI is designed to support your mental wellness journey through AI-powered conversations,
                            mood tracking, and evidence-based coping tools.
                        </p>
                        <p className="text-theme-text-subtle">
                            Remember: This app is not a substitute for professional mental health care.
                            If you're experiencing a mental health crisis, please seek help from a qualified professional.
                        </p>
                    </div>

                    <div className="glass-card">
                        <h3 className="text-xl font-bold text-theme-text-main mb-4">Resources</h3>
                        <div className="space-y-2">
                            <a href="#" className="block text-theme-primary hover:underline">Privacy Policy</a>
                            <a href="#" className="block text-theme-primary hover:underline">Terms of Service</a>
                            <a href="#" className="block text-theme-primary hover:underline">Help & Support</a>
                            <a href="#" className="block text-theme-primary hover:underline">Give Feedback</a>
                        </div>
                    </div>
                </div>
            )}
        </Layout>
    );
}
