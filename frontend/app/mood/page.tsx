'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';


interface MoodEntry {
    date: string;
    mood: string;
    summary?: string;
}

export default function MoodJournalPage() {
    const [entries, setEntries] = useState<MoodEntry[]>([]);
    const [todayMood, setTodayMood] = useState('—');
    const [isLoading, setIsLoading] = useState(true);
    const [streak, setStreak] = useState(0);
    const [mostCommonMood, setMostCommonMood] = useState('N/A');

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
        fetchMoodData();
    }, [user, authLoading]);

    const fetchMoodData = async () => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/mood_data`, {
                credentials: 'include',
            });
            const data = await response.json();

            if (data.entries && data.entries.length > 0) {
                setEntries(data.entries);

                // Get today's mood
                const today = new Date().toISOString().split('T')[0];
                const todayEntry = data.entries.find((e: any) => e.date === today);
                if (todayEntry) {
                    setTodayMood(todayEntry.mood);
                }

                // Calculate streak (consecutive days with entries)
                const dates = data.entries.map((e: any) => e.date).sort().reverse();
                let currentStreak = 0;
                const todayDate = new Date();

                for (let i = 0; i < dates.length; i++) {
                    const entryDate = new Date(dates[i]);
                    const daysDiff = Math.floor((todayDate.getTime() - entryDate.getTime()) / (1000 * 60 * 60 * 24));

                    if (daysDiff === i) {
                        currentStreak++;
                    } else {
                        break;
                    }
                }
                setStreak(currentStreak);

                // Calculate most common mood
                const moodCounts: Record<string, number> = {};
                data.entries.forEach((e: any) => {
                    const mood = e.mood || 'neutral';
                    moodCounts[mood] = (moodCounts[mood] || 0) + 1;
                });
                const sortedMoods = Object.entries(moodCounts).sort((a, b) => b[1] - a[1]);
                if (sortedMoods.length > 0) {
                    setMostCommonMood(sortedMoods[0][0]);
                }
            }
            setIsLoading(false);
        } catch (error) {
            console.error('Failed to fetch mood data:', error);
            setIsLoading(false);
        }
    };

    if (authLoading || !user) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="loading-shimmer h-8 w-48 rounded-lg"></div>
            </div>
        );
    }

    const moodEmojis: Record<string, string> = {
        happy: '😊',
        sad: '😢',
        anxious: '😰',
        calm: '😌',
        angry: '😡',
        excited: '🤩',
        mixed: '😐',
        neutral: '😶',
    };

    const moodColors: Record<string, string> = {
        happy: 'from-theme-success/80 to-theme-success',
        sad: 'from-blue-400 to-blue-600',
        anxious: 'from-theme-warning/80 to-theme-warning',
        calm: 'from-theme-primary/80 to-theme-primary',
        angry: 'from-theme-danger/80 to-theme-danger',
        excited: 'from-pink-400 to-pink-600',
        mixed: 'from-gray-400 to-gray-600',
        neutral: 'from-gray-400 to-gray-500',
    };

    return (
        <>
            {/* Header */}
            <div className="mb-8 animate-fade-in-up">
                <h1 className="text-4xl font-bold text-theme-text-main mb-2">
                    Mood Journal 📔
                </h1>
                <p className="text-theme-text-subtle text-lg">
                    Track and reflect on your emotional journey
                </p>
            </div>

            {/* Today's Mood Card - Large Feature */}
            <div className="glass-card mb-8 animate-fade-in-up" style={{ animationDelay: '100ms' }}>
                <div className="text-center py-8">
                    <div className="text-sm text-theme-text-subtle mb-2">Today's Mood</div>
                    <div className={`inline-block px-8 py-6 rounded-3xl bg-gradient-to-br ${moodColors[todayMood.toLowerCase()] || 'from-theme-hover to-theme-hover/50'} shadow-card-hover`}>
                        <div className="text-7xl mb-4">{moodEmojis[todayMood.toLowerCase()] || '😶'}</div>
                        <div className="text-3xl font-bold text-white capitalize">{todayMood}</div>
                    </div>
                    <p className="text-theme-text-subtle mt-6 max-w-md mx-auto">
                        Your emotions are valid. Take time to understand how you feel.
                    </p>
                </div>
            </div>

            {/* Mood Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="glass-card animate-fade-in-up bg-gradient-to-br from-theme-primary/10 to-theme-primary/5 border-2 border-theme-primary/20" style={{ animationDelay: '200ms' }}>
                    <div className="text-center">
                        <div className="text-4xl mb-3">📊</div>
                        <div className="text-sm text-theme-text-subtle mb-2">Total Entries</div>
                        <div className="text-4xl font-bold text-theme-primary">{entries.length}</div>
                    </div>
                </div>

                <div className="glass-card animate-fade-in-up bg-gradient-to-br from-theme-secondary/10 to-theme-secondary/5 border-2 border-theme-secondary/20" style={{ animationDelay: '300ms' }}>
                    <div className="text-center">
                        <div className="text-4xl mb-3">🔥</div>
                        <div className="text-sm text-theme-text-subtle mb-2">Check-in Streak</div>
                        <div className="text-4xl font-bold text-theme-secondary">{streak}</div>
                        <div className="text-xs text-theme-text-subtle mt-1">days</div>
                    </div>
                </div>

                <div className="glass-card animate-fade-in-up bg-gradient-to-br from-theme-success/10 to-theme-success/5 border-2 border-theme-success/20" style={{ animationDelay: '400ms' }}>
                    <div className="text-center">
                        <div className="text-4xl mb-3">😊</div>
                        <div className="text-sm text-theme-text-subtle mb-2">Most Common</div>
                        <div className="text-2xl font-bold text-theme-success capitalize">
                            {mostCommonMood}
                        </div>
                    </div>
                </div>
            </div>

            {/* Recent Entries */}
            <div className="glass-card mb-8 animate-fade-in-up" style={{ animationDelay: '500ms' }}>
                <h2 className="text-2xl font-bold text-theme-text-main mb-6">Recent Entries</h2>

                {isLoading ? (
                    <div className="space-y-3">
                        {[1, 2, 3].map((i) => (
                            <div key={i} className="loading-shimmer h-20 rounded-xl"></div>
                        ))}
                    </div>
                ) : entries.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {entries.slice(0, 9).map((entry, index) => (
                            <div
                                key={index}
                                className="p-5 rounded-2xl bg-theme-hover/30 hover:bg-theme-hover/50 transition-all border-l-4 cursor-pointer hover:scale-105"
                                style={{
                                    borderLeftColor:
                                        entry.mood === 'happy' ? '#7BC67E' :
                                            entry.mood === 'sad' ? '#6BA4B8' :
                                                entry.mood === 'anxious' ? '#F9C74F' :
                                                    entry.mood === 'calm' ? '#5BA3A3' :
                                                        entry.mood === 'angry' ? '#F28B82' : '#B0C4DE'
                                }}
                            >
                                <div className="flex items-start justify-between mb-3">
                                    <div className="flex-1">
                                        <p className="font-bold text-lg text-theme-text-main capitalize">
                                            {entry.mood}
                                        </p>
                                        <p className="text-sm text-theme-text-subtle">{entry.date}</p>
                                    </div>
                                    <div className="text-3xl">
                                        {moodEmojis[entry.mood.toLowerCase()] || '😶'}
                                    </div>
                                </div>
                                {entry.summary && (
                                    <p className="text-xs text-theme-text-muted line-clamp-2">
                                        {entry.summary}
                                    </p>
                                )}
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="text-center py-12">
                        <div className="text-6xl mb-4">📔</div>
                        <p className="text-theme-text-subtle">No entries yet. Start tracking your mood in the chat!</p>
                    </div>
                )}
            </div>

            {/* Mood Tips */}
            <div className="glass-card animate-fade-in-up" style={{ animationDelay: '600ms' }}>
                <h2 className="text-2xl font-bold text-theme-text-main mb-6">Mood Tracking Tips</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 rounded-xl bg-theme-hover/20 border border-theme-primary/20">
                        <div className="text-2xl mb-2">💭</div>
                        <h3 className="font-semibold text-theme-text-main mb-2">Be Honest</h3>
                        <p className="text-sm text-theme-text-subtle">
                            There's no right or wrong mood. Be honest with yourself about how you feel.
                        </p>
                    </div>
                    <div className="p-4 rounded-xl bg-theme-hover/20 border border-theme-primary/20">
                        <div className="text-2xl mb-2">⏰</div>
                        <h3 className="font-semibold text-theme-text-main mb-2">Stay Consistent</h3>
                        <p className="text-sm text-theme-text-subtle">
                            Track your mood daily to identify patterns and triggers over time.
                        </p>
                    </div>
                    <div className="p-4 rounded-xl bg-theme-hover/20 border border-theme-primary/20">
                        <div className="text-2xl mb-2">📝</div>
                        <h3 className="font-semibold text-theme-text-main mb-2">Add Context</h3>
                        <p className="text-sm text-theme-text-subtle">
                            Note what influenced your mood - events, people, or situations.
                        </p>
                    </div>
                    <div className="p-4 rounded-xl bg-theme-hover/20 border border-theme-primary/20">
                        <div className="text-2xl mb-2">🌱</div>
                        <h3 className="font-semibold text-theme-text-main mb-2">Celebrate Progress</h3>
                        <p className="text-sm text-theme-text-subtle">
                            Notice positive changes and celebrate small wins in your journey.
                        </p>
                    </div>
                </div>
            </div>
        </>
    );
}
