'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import Layout from '@/components/layout/Layout';

export default function AnalyticsPage() {
    const [stats, setStats] = useState({
        totalCheckins: 0,
        moodDistribution: {} as Record<string, number>,
        weeklyAverage: 0,
        streak: 0,
    });

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
        fetchAnalytics();
    }, [user, authLoading]);

    const fetchAnalytics = async () => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/mood_data`, {
                credentials: 'include',
            });
            const data = await response.json();

            if (data.entries && data.entries.length > 0) {
                const moodCounts: Record<string, number> = {};
                data.entries.forEach((entry: any) => {
                    moodCounts[entry.mood] = (moodCounts[entry.mood] || 0) + 1;
                });

                setStats({
                    totalCheckins: data.entries.length,
                    moodDistribution: moodCounts,
                    weeklyAverage: Math.round(data.entries.length / 4), // Estimate
                    streak: 3, // TODO: Calculate actual streak
                });
            }
        } catch (error) {
            console.error('Failed to fetch analytics:', error);
        }
    };

    if (authLoading || !user) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="loading-shimmer h-8 w-48 rounded-lg"></div>
            </div>
        );
    }

    const moodColors: Record<string, string> = {
        happy: '#7BC67E',
        sad: '#6BA4B8',
        anxious: '#F9C74F',
        calm: '#5BA3A3',
        angry: '#F28B82',
        excited: '#FF9B85',
        mixed: '#B0C4DE',
        neutral: '#7A96B0',
    };

    const totalMoods = Object.values(stats.moodDistribution).reduce((a, b) => a + b, 0);

    return (
        <Layout>
            {/* Header */}
            <div className="mb-8 animate-fade-in-up">
                <h1 className="text-4xl font-bold text-theme-text-main mb-2">
                    Analytics 📊
                </h1>
                <p className="text-theme-text-subtle text-lg">
                    Insights into your emotional wellbeing journey
                </p>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div className="glass-card animate-fade-in-up bg-gradient-to-br from-theme-primary/20 to-theme-primary/10 border-2 border-theme-primary/30" style={{ animationDelay: '100ms' }}>
                    <div className="text-center">
                        <div className="text-4xl mb-3">📈</div>
                        <div className="text-sm text-theme-text-subtle mb-2">Total Check-ins</div>
                        <div className="text-4xl font-bold text-theme-primary">{stats.totalCheckins}</div>
                    </div>
                </div>

                <div className="glass-card animate-fade-in-up bg-gradient-to-br from-theme-secondary/20 to-theme-secondary/10 border-2 border-theme-secondary/30" style={{ animationDelay: '200ms' }}>
                    <div className="text-center">
                        <div className="text-4xl mb-3">📅</div>
                        <div className="text-sm text-theme-text-subtle mb-2">Weekly Average</div>
                        <div className="text-4xl font-bold text-theme-secondary">{stats.weeklyAverage}</div>
                        <div className="text-xs text-theme-text-subtle mt-1">checks/week</div>
                    </div>
                </div>

                <div className="glass-card animate-fade-in-up bg-gradient-to-br from-theme-success/20 to-theme-success/10 border-2 border-theme-success/30" style={{ animationDelay: '300ms' }}>
                    <div className="text-center">
                        <div className="text-4xl mb-3">🔥</div>
                        <div className="text-sm text-theme-text-subtle mb-2">Current Streak</div>
                        <div className="text-4xl font-bold text-theme-success">{stats.streak}</div>
                        <div className="text-xs text-theme-text-subtle mt-1">days</div>
                    </div>
                </div>

                <div className="glass-card animate-fade-in-up bg-gradient-to-br from-theme-accent/20 to-theme-accent/10 border-2 border-theme-accent/30" style={{ animationDelay: '400ms' }}>
                    <div className="text-center">
                        <div className="text-4xl mb-3">⭐</div>
                        <div className="text-sm text-theme-text-subtle mb-2">Best Streak</div>
                        <div className="text-4xl font-bold text-theme-accent">5</div>
                        <div className="text-xs text-theme-text-subtle mt-1">days</div>
                    </div>
                </div>
            </div>

            {/* Mood Distribution */}
            <div className="glass-card mb-8 animate-fade-in-up" style={{ animationDelay: '500ms' }}>
                <h2 className="text-2xl font-bold text-theme-text-main mb-6">Mood Distribution</h2>
                <div className="space-y-4">
                    {Object.entries(stats.moodDistribution).sort((a, b) => b[1] - a[1]).map(([mood, count]) => {
                        const percentage = totalMoods > 0 ? Math.round((count / totalMoods) * 100) : 0;
                        return (
                            <div key={mood}>
                                <div className="flex items-center justify-between mb-2">
                                    <span className="font-semibold text-theme-text-main capitalize">{mood}</span>
                                    <span className="text-theme-text-subtle">{count} times ({percentage}%)</span>
                                </div>
                                <div className="w-full h-4 bg-theme-hover/30 rounded-full overflow-hidden">
                                    <div
                                        className="h-full rounded-full transition-all duration-500"
                                        style={{
                                            width: `${percentage}%`,
                                            backgroundColor: moodColors[mood.toLowerCase()] || '#B0C4DE'
                                        }}
                                    ></div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Weekly Patterns */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="glass-card animate-fade-in-up" style={{ animationDelay: '600ms' }}>
                    <h3 className="text-xl font-bold text theme-text-main mb-4">Most Common Moods</h3>
                    <div className="space-y-3">
                        {Object.entries(stats.moodDistribution)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 3)
                            .map(([mood, count], index) => (
                                <div key={mood} className="flex items-center gap-3 p-3 rounded-xl bg-theme-hover/30">
                                    <div className="text-3xl font-bold text-theme-primary">#{index + 1}</div>
                                    <div className="flex-1">
                                        <div className="font-semibold text-theme-text-main capitalize">{mood}</div>
                                        <div className="text-sm text-theme-text-subtle">{count} occurrences</div>
                                    </div>
                                </div>
                            ))}
                    </div>
                </div>

                <div className="glass-card animate-fade-in-up" style={{ animationDelay: '700ms' }}>
                    <h3 className="text-xl font-bold text-theme-text-main mb-4">Insights & Tips</h3>
                    <div className="space-y-3">
                        <div className="p-4 rounded-xl bg-theme-primary/20 border border-theme-primary/30">
                            <div className="text-2xl mb-2">💡</div>
                            <p className="text-sm text-theme-text-main">
                                You're doing great with consistent check-ins! Keep tracking your emotions.
                            </p>
                        </div>
                        <div className="p-4 rounded-xl bg-theme-secondary/20 border border-theme-secondary/30">
                            <div className="text-2xl mb-2">🌟</div>
                            <p className="text-sm text-theme-text-main">
                                Your emotional awareness is growing. Notice patterns and celebrate progress!
                            </p>
                        </div>
                        <div className="p-4 rounded-xl bg-theme-success/20 border border-theme-success/30">
                            <div className="text-2xl mb-2">🎯</div>
                            <p className="text-sm text-theme-text-main">
                                Try the coping tools when you notice challenging moods.
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Placeholder for Charts */}
            <div className="glass-card animate-fade-in-up" style={{ animationDelay: '800ms' }}>
                <h2 className="text-2xl font-bold text-theme-text-main mb-6">Mood Trends Over Time</h2>
                <div className="text-center py-12">
                    <div className="text-6xl mb-4">📈</div>
                    <p className="text-theme-text-subtle mb-4">
                        Advanced charts and trend analysis coming soon!
                    </p>
                    <p className="text-sm text-theme-text-muted">
                        We're working on beautiful visualizations to help you understand your emotional patterns better.
                    </p>
                </div>
            </div>
        </Layout>
    );
}
