'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';

export default function SettingsPage() {
    const [username, setUsername] = useState('');
    const [language, setLanguage] = useState('en');
    const [dailyReminder, setDailyReminder] = useState(false);
    const [pushToken, setPushToken] = useState('');
    const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);

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
        fetchSettings();
    }, [user, authLoading]);

    const showMessage = (text: string, type: 'success' | 'error' = 'success') => {
        setMessage({ text, type });
        setTimeout(() => setMessage(null), 3000);
    };

    const fetchSettings = async () => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/get_settings`, {
                credentials: 'include',
            });
            const data = await response.json();
            setUsername(data.username || '');
            setLanguage(data.language || 'en');
            setDailyReminder(Boolean(data.daily_reminder));
            setPushToken(data.push_token || '');
        } catch {
            showMessage('Failed to load settings.', 'error');
        }
    };

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/update_settings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username,
                    language,
                    dailyReminder,
                    pushToken,
                }),
                credentials: 'include',
            });
            const result = await response.json();
            if (result.ok) {
                showMessage('Settings updated!');
                setTimeout(() => window.location.reload(), 1000);
            } else {
                showMessage(result.message || 'Update failed.', 'error');
            }
        } catch {
            showMessage('Update failed.', 'error');
        }
    };

    const handleDelete = async () => {
        if (!confirm('Are you sure? This cannot be undone.')) return;
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/delete_account`, {
                method: 'POST',
                credentials: 'include',
            });
            const result = await response.json();
            if (result.ok) {
                alert('Account deleted.');
                window.location.href = '/login';
            } else {
                showMessage(result.message || 'Failed to delete account.', 'error');
            }
        } catch {
            showMessage('Failed to delete account.', 'error');
        }
    };

    if (authLoading || !user) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="loading-shimmer h-8 w-48 rounded-lg"></div>
            </div>
        );
    }

    return (
        <>
            <header className="flex justify-between items-center mb-8">
                <div className="flex items-center gap-4">
                    <h1 className="text-2xl font-bold text-theme-text-main">Settings</h1>
                </div>
            </header>

            <section className="space-y-6 max-w-2xl mx-auto">
                {message && (
                    <div
                        className={`p-4 rounded-xl text-center font-medium transition-all duration-300 ${
                            message.type === 'success'
                                ? 'bg-green-100 text-green-700'
                                : 'bg-red-100 text-red-700'
                        }`}
                    >
                        {message.text}
                    </div>
                )}

                <div className="card">
                    <h3 className="text-xl font-semibold mb-6 text-theme-text-main">Account Preferences</h3>
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div>
                            <label htmlFor="username" className="block text-sm font-medium text-theme-text-subtle">
                                Username
                            </label>
                            <input
                                type="text"
                                id="username"
                                name="username"
                                required
                                value={username}
                                onChange={(event) => setUsername(event.target.value)}
                                className="mt-1 w-full p-3 rounded-md border border-theme-primary/30 bg-theme-panel text-theme-text-main"
                            />
                        </div>

                        <div>
                            <label htmlFor="language" className="block text-sm font-medium text-theme-text-subtle">
                                Change Language
                            </label>
                            <select
                                id="language"
                                name="language"
                                value={language}
                                onChange={(event) => setLanguage(event.target.value)}
                                className="mt-1 w-full p-3 rounded-md border border-theme-primary/30 bg-theme-panel text-theme-text-main"
                            >
                                <option value="en">English</option>
                                <option value="hi">हिन्दी</option>
                                <option value="ta">தமிழ்</option>
                                <option value="te">తెలుగు</option>
                                <option value="ml">മലയാളം</option>
                                <option value="kn">ಕನ್ನಡ</option>
                            </select>
                        </div>

                        <div>
                            <label htmlFor="pushbullet-token" className="block text-sm font-medium text-theme-text-subtle">
                                Pushbullet Token
                            </label>
                            <input
                                type="password"
                                id="pushbullet-token"
                                name="pushbullet-token"
                                value={pushToken}
                                onChange={(event) => setPushToken(event.target.value)}
                                className="mt-1 w-full p-3 rounded-md border border-theme-primary/30 bg-theme-panel text-theme-text-main"
                                placeholder="Optional: To receive crisis alerts"
                            />
                        </div>

                        <div className="flex items-center justify-between">
                            <span className="flex-grow flex flex-col">
                                <span className="text-sm font-medium text-theme-text-main">Daily Reminder</span>
                                <span className="text-xs text-theme-text-subtle">Enable daily check-in reminders (requires Pushbullet).</span>
                            </span>
                            <label className="relative inline-flex items-center cursor-pointer">
                                <input
                                    type="checkbox"
                                    id="daily-reminder"
                                    className="sr-only peer"
                                    checked={dailyReminder}
                                    onChange={(event) => setDailyReminder(event.target.checked)}
                                />
                                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-teal-600"></div>
                            </label>
                        </div>

                        <div>
                            <button
                                type="submit"
                                className="w-full flex justify-center py-2 px-4 border rounded-lg shadow-sm font-medium text-white bg-teal-600 hover:bg-teal-700"
                            >
                                Save Changes
                            </button>
                        </div>
                    </form>
                </div>

                <div className="card">
                    <h3 className="text-xl font-semibold mb-6 text-red-600">Delete Account</h3>
                    <p className="mt-2 mb-4 text-sm text-theme-text-subtle">This action cannot be undone.</p>
                    <button
                        id="delete-account-button"
                        onClick={handleDelete}
                        className="w-full flex justify-center py-2 px-4 border rounded-lg shadow-sm font-medium text-white bg-red-600 hover:bg-red-700"
                    >
                        Delete Account
                    </button>
                </div>
            </section>

            <section className="max-w-4xl mx-auto mt-10 grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="glass-card">
                    <h3 className="text-xl font-bold text-theme-text-main mb-4">About Sahaay AI</h3>
                    <p className="text-theme-text-subtle">
                        Sahaay AI is designed to support your mental wellness journey through AI-powered conversations,
                        mood tracking, and evidence-based coping tools.
                    </p>
                    <p className="text-theme-text-subtle mt-3">
                        Remember: This app is not a substitute for professional mental health care.
                        If you are experiencing a mental health crisis, please seek help from a qualified professional.
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
            </section>
        </>
    );
}
