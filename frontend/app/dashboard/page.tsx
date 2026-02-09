'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import { dashboardAPI } from '@/lib/api';
import Layout from '@/components/layout/Layout';
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
    const checkAuth = useAuthStore((state) => state.checkAuth);
    const router = useRouter();

    useEffect(() => {
        checkAuth();
    }, [checkAuth]);

    useEffect(() => {
        if (isLoading) return;
        if (!user) {
            router.push('/login');
            return;
        }
        fetchDashboardData();
    }, [user, isLoading]);

    const fetchDashboardData = async () => {
        try {
            const response = await dashboardAPI.getHomeData();
            setData(response.data);
        } catch (error) {
            console.error('Failed to fetch dashboard data:', error);
        }
    };

    if (isLoading || !user) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="loading-shimmer h-8 w-48 rounded-lg"></div>
            </div>
        );
    }

    return (
        <Layout>
            {/* Header */}
            <div className="mb-8 animate-fade-in-up">
                <h1 className="text-4xl font-bold text-theme-text-main mb-2">
                    Welcome back, {user.username || user.name}! 👋
                </h1>
                <p className="text-theme-text-subtle text-lg">
                    Here's your mental wellness overview
                </p>
            </div>

            {/* Unique Bento Grid Layout */}
            <div className="grid grid-cols-12 gap-6 mb-8">
                {/* Large Streak Card - Takes 2x space */}
                <div className="col-span-12 md:col-span-5 glass-card animate-fade-in-up bg-gradient-to-br from-theme-primary/20 to-theme-primary-dark/10 border-2 border-theme-primary/30 hover:border-theme-primary/60"
                    style={{ animationDelay: '100ms', minHeight: '280px' }}>
                    <div className="flex flex-col justify-between h-full">
                        <div>
                            <div className="inline-block p-3 rounded-2xl bg-theme-primary/20 mb-4">
                                <span className="text-5xl">🔥</span>
                            </div>
                            <h3 className="text-2xl font-bold text-theme-text-main mb-2">Check-in Streak</h3>
                            <p className="text-theme-text-subtle mb-6">Keep up the great work!</p>
                        </div>
                        <div>
                            <p className="text-7xl font-bold text-theme-primary mb-2">
                                {data?.streak || 0}
                            </p>
                            <p className="text-xl text-theme-text-subtle">days in a row 🌟</p>
                        </div>
                    </div>
                </div>

                {/* Current Mood Card - Vertical */}
                <div className="col-span-12 md:col-span-4 glass-card animate-fade-in-up bg-gradient-to-br from-theme-secondary/20 to-theme-secondary-dark/10 border-2 border-theme-secondary/30 hover:border-theme-secondary/60"
                    style={{ animationDelay: '200ms', minHeight: '280px' }}>
                    <div className="flex flex-col justify-between h-full text-center">
                        <div>
                            <div className="inline-block p-4 rounded-2xl bg-theme-secondary/20 mb-4">
                                <span className="text-5xl">😊</span>
                            </div>
                            <h3 className="text-xl font-bold text-theme-text-main mb-3">Current Mood</h3>
                        </div>
                        <div>
                            <p className="text-3xl font-semibold text-theme-secondary capitalize mb-4">
                                {data?.mood || 'Not recorded'}
                            </p>
                            <Link href="/mood">
                                <button className="btn btn-secondary w-full">
                                    📔 Update Mood
                                </button>
                            </Link>
                        </div>
                    </div>
                </div>

                {/* Quote Card - Small but impactful */}
                <div className="col-span-12 md:col-span-3 glass-card animate-fade-in-up bg-gradient-to-br from-theme-accent/10 to-theme-accent/5 border-2 border-theme-accent/30 hover:border-theme-accent/60"
                    style={{ animationDelay: '300ms', minHeight: '280px' }}>
                    <div className="flex flex-col justify-between h-full">
                        <div>
                            <div className="inline-block p-3 rounded-2xl bg-theme-accent/20 mb-4">
                                <span className="text-4xl">✨</span>
                            </div>
                            <h3 className="text-lg font-bold text-theme-text-main mb-3">Daily Inspiration</h3>
                        </div>
                        <div>
                            <p className="text-theme-text-subtle italic text-sm leading-relaxed">
                                "{data?.quote || 'Take care of your mind today'}"
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Quick Actions - Unique Card Shapes */}
            <div className="glass-card mb-8 animate-fade-in-up" style={{ animationDelay: '400ms' }}>
                <h2 className="text-2xl font-bold text-theme-text-main mb-6">Quick Actions</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <Link href="/chat">
                        <div className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-theme-primary to-theme-primary-dark p-6 hover:scale-105 transition-transform cursor-pointer border-2 border-theme-primary/50 hover:border-theme-primary">
                            <div className="text-4xl mb-3">💬</div>
                            <h3 className="text-lg font-bold text-white mb-1">Chat with AI</h3>
                            <p className="text-sm text-white/80">Get instant support</p>
                            <div className="absolute top-0 right-0 w-20 h-20 bg-white/10 rounded-full -mr-10 -mt-10"></div>
                        </div>
                    </Link>

                    <Link href="/mood">
                        <div className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-theme-secondary to-theme-secondary-dark p-6 hover:scale-105 transition-transform cursor-pointer border-2 border-theme-secondary/50 hover:border-theme-secondary">
                            <div className="text-4xl mb-3">📔</div>
                            <h3 className="text-lg font-bold text-white mb-1">Log Mood</h3>
                            <p className="text-sm text-white/80">Track your emotions</p>
                            <div className="absolute bottom-0 left-0 w-16 h-16 bg-white/10 rounded-full -ml-8 -mb-8"></div>
                        </div>
                    </Link>

                    <Link href="/tools">
                        <div className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-theme-success/80 to-theme-success p-6 hover:scale-105 transition-transform cursor-pointer border-2 border-theme-success/50 hover:border-theme-success">
                            <div className="text-4xl mb-3">🧘</div>
                            <h3 className="text-lg font-bold text-white mb-1">Coping Tools</h3>
                            <p className="text-sm text-white/80">Find calm & peace</p>
                            <div className="absolute top-0 left-0 w-24 h-24 bg-white/10 rounded-full -ml-12 -mt-12"></div>
                        </div>
                    </Link>

                    <Link href="/analytics">
                        <div className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-theme-accent/80 to-theme-warning p-6 hover:scale-105 transition-transform cursor-pointer border-2 border-theme-accent/50 hover:border-theme-accent">
                            <div className="text-4xl mb-3">📊</div>
                            <h3 className="text-lg font-bold text-white mb-1">View Analytics</h3>
                            <p className="text-sm text-white/80">Track your progress</p>
                            <div className="absolute bottom-0 right-0 w-20 h-20 bg-white/10 rounded-full -mr-10 -mb-10"></div>
                        </div>
                    </Link>
                </div>
            </div>

            {/* Recent Activity - Timeline Style */}
            {data?.recent && data.recent.length > 0 && (
                <div className="glass-card animate-fade-in-up" style={{ animationDelay: '500ms' }}>
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-2xl font-bold text-theme-text-main">Recent Check-ins</h2>
                        <div className="text-sm text-theme-text-subtle">Last 7 days</div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {data.recent.slice(0, 3).map((entry, index) => (
                            <div
                                key={index}
                                className="relative p-5 rounded-2xl bg-theme-hover/30 hover:bg-theme-hover/50 transition-all border-l-4"
                                style={{
                                    borderLeftColor:
                                        entry.mood === 'happy' ? '#7BC67E' :
                                            entry.mood === 'sad' ? '#F28B82' :
                                                entry.mood === 'anxious' ? '#F9C74F' :
                                                    entry.mood === 'calm' ? '#5BA3A3' : '#B0C4DE'
                                }}
                            >
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <p className="font-bold text-lg text-theme-text-main capitalize mb-1">
                                            {entry.mood}
                                        </p>
                                        <p className="text-sm text-theme-text-subtle">{entry.date}</p>
                                    </div>
                                    <div className="text-4xl">
                                        {entry.mood === 'happy' ? '😊' :
                                            entry.mood === 'sad' ? '😢' :
                                                entry.mood === 'anxious' ? '😰' :
                                                    entry.mood === 'calm' ? '😌' : '😐'}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </Layout>
    );
}
