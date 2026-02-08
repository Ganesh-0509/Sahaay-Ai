'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import { dashboardAPI } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import Link from 'next/link';

interface MoodEntry {
    date: string;
    mood: string;
}

interface HomeData {
    streak: number;
    recent: MoodEntry[];
    mood: string;
    quote: string;
}

export default function DashboardPage() {
    const [data, setData] = useState<HomeData | null>(null);

    const user = useAuthStore((state) => state.user);
    const isLoading = useAuthStore((state) => state.isLoading);
    const logout = useAuthStore((state) => state.logout);
    const checkAuth = useAuthStore((state) => state.checkAuth);
    const router = useRouter();

    useEffect(() => {
        checkAuth();
    }, [checkAuth]);

    useEffect(() => {
        // Don't redirect while still loading auth state
        if (isLoading) return;

        if (!user) {
            router.push('/login');
            return;
        }

        fetchDashboardData();
    }, [user, isLoading, router]);

    const fetchDashboardData = async () => {
        try {
            const response = await dashboardAPI.getHomeData();
            setData(response.data);
        } catch (error) {
            console.error('Failed to fetch dashboard data:', error);
        }
    };

    const handleLogout = async () => {
        await logout();
        router.push('/login');
    };

    if (isLoading || !user) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-background">
                <p className="text-text-secondary">Loading...</p>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background">
            {/* Header */}
            <header className="bg-surface border-b border-border">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
                    <h1 className="text-2xl font-semibold text-text-primary">Sahaay AI</h1>
                    <div className="flex items-center gap-4">
                        <span className="text-text-secondary">{user?.name || user?.email || 'User'}</span>
                        <Button variant="outline" onClick={handleLogout}>
                            Logout
                        </Button>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Welcome Section */}
                <div className="mb-8">
                    <h2 className="text-3xl font-semibold text-text-primary mb-2">
                        Welcome back, {user?.name?.split(' ')[0] || user?.email?.split('@')[0] || 'there'}
                    </h2>
                    <p className="text-text-secondary">
                        {data?.quote || 'How are you feeling today?'}
                    </p>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    {/* Current Mood */}
                    <div className="card">
                        <h3 className="text-sm font-medium text-text-secondary mb-2">
                            Current Mood
                        </h3>
                        <p className="text-2xl font-semibold text-text-primary">
                            {data?.mood || 'No data yet'}
                        </p>
                    </div>

                    {/* Streak */}
                    <div className="card">
                        <h3 className="text-sm font-medium text-text-secondary mb-2">
                            Check-in Streak
                        </h3>
                        <p className="text-2xl font-semibold text-text-primary">
                            {data?.streak || 0} days
                        </p>
                    </div>

                    {/* Quick Actions */}
                    <div className="card">
                        <h3 className="text-sm font-medium text-text-secondary mb-2">
                            Quick Actions
                        </h3>
                        <Link href="/chat">
                            <Button variant="primary" className="w-full">
                                Start Chat
                            </Button>
                        </Link>
                    </div>
                </div>

                {/* Recent Moods */}
                <div className="card">
                    <h3 className="text-lg font-semibold text-text-primary mb-4">
                        Recent Check-ins
                    </h3>
                    {data && data.recent && data.recent.length > 0 ? (
                        <div className="space-y-3">
                            {data.recent.map((entry, index) => (
                                <div
                                    key={index}
                                    className="flex justify-between items-center py-2 border-b border-border last:border-0"
                                >
                                    <span className="text-text-secondary text-sm">{entry.date}</span>
                                    <span className="text-text-primary font-medium">{entry.mood}</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-text-secondary text-center py-8">
                            No check-ins yet. Start by chatting with Sahaay AI.
                        </p>
                    )}
                </div>

                {/* Navigation Links */}
                <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
                    <Link href="/chat" className="card hover:border-primary cursor-pointer text-center">
                        <p className="font-medium text-text-primary">Chat</p>
                    </Link>
                    <Link href="/mood" className="card hover:border-primary cursor-pointer text-center">
                        <p className="font-medium text-text-primary">Mood Tracker</p>
                    </Link>
                    <Link href="/settings" className="card hover:border-primary cursor-pointer text-center">
                        <p className="font-medium text-text-primary">Settings</p>
                    </Link>
                    <div className="card opacity-50 cursor-not-allowed text-center">
                        <p className="font-medium text-text-secondary">Community</p>
                    </div>
                </div>
            </main>
        </div>
    );
}
