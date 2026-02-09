'use client';

import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';

interface ToolEntry {
    tip_id: string;
    mood?: string;
    description?: string;
}

export default function CopingToolsPage() {
    const [period, setPeriod] = useState('last10');
    const [tools, setTools] = useState<ToolEntry[]>([]);
    const [currentMoodFilter, setCurrentMoodFilter] = useState('all');

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
        fetchTools(period);
    }, [user, authLoading, period]);

    const fetchTools = async (nextPeriod: string) => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/tools_data?period=${nextPeriod}`, {
                credentials: 'include',
            });
            const data = await response.json();
            setTools(Array.isArray(data.tools) ? data.tools : []);
        } catch (error) {
            console.error('Failed to load tools:', error);
            setTools([]);
        }
    };

    const handleHelpful = async (tipId: string) => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/mark_helpful`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tip_id: tipId }),
                credentials: 'include',
            });
            const result = await response.json();
            if (result.ok) {
                alert('Marked as helpful! It will now appear on your Home page.');
            } else {
                alert(result.error || 'Error marking helpful');
            }
        } catch (error) {
            console.error(error);
            alert('Failed to mark helpful');
        }
    };

    const filteredTools = useMemo(() => {
        if (currentMoodFilter === 'all') return tools;
        return tools.filter((tool) => (tool.mood || '').toLowerCase() === currentMoodFilter);
    }, [tools, currentMoodFilter]);

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
                    <h1 className="text-2xl font-bold text-theme-text-main">Coping Tools</h1>
                </div>
                <div className="flex items-center gap-4">
                    <div className="relative inline-block text-left">
                        <select
                            id="filterSelect"
                            value={period}
                            onChange={(event) => setPeriod(event.target.value)}
                            className="bg-theme-panel border border-theme-primary/30 rounded-lg shadow-sm px-4 py-2 text-sm font-medium text-theme-text-main"
                        >
                            <option value="last10">Last 10 Entries</option>
                            <option value="all">All Time</option>
                        </select>
                    </div>
                </div>
            </header>

            <div className="mb-6 flex flex-wrap gap-3">
                <button className="filter-btn px-3 py-1 rounded-lg bg-teal-200" onClick={() => setCurrentMoodFilter('all')}>All</button>
                <button className="filter-btn px-3 py-1 rounded-lg bg-blue-200" onClick={() => setCurrentMoodFilter('sad')}>Sad ☹️</button>
                <button className="filter-btn px-3 py-1 rounded-lg bg-green-200" onClick={() => setCurrentMoodFilter('calm')}>Calm 😌</button>
                <button className="filter-btn px-3 py-1 rounded-lg bg-pink-200" onClick={() => setCurrentMoodFilter('happy')}>Happy 😄</button>
                <button className="filter-btn px-3 py-1 rounded-lg bg-purple-200" onClick={() => setCurrentMoodFilter('anxious')}>Anxious 😰</button>
            </div>

            <section id="toolsList" className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                {filteredTools.length > 0 ? (
                    filteredTools.map((tool) => {
                        const mood = (tool.mood || 'neutral').toLowerCase();
                        const moodColors: Record<string, string> = {
                            sad: 'bg-blue-100',
                            calm: 'bg-green-100',
                            happy: 'bg-pink-100',
                            anxious: 'bg-purple-100',
                        };
                        const color = moodColors[mood] || 'bg-gray-100';
                        return (
                            <div
                                key={tool.tip_id}
                                className={`card p-6 rounded-2xl shadow-lg transition-transform hover:scale-105 ${color}`}
                                data-mood-card={mood}
                            >
                                <h3 className="font-semibold text-lg mb-1 capitalize">Mood: {mood}</h3>
                                <p className="mt-2 text-sm italic text-theme-text-subtle">💡 {tool.description || 'No description.'}</p>
                                <div className="mt-3 flex gap-2">
                                    <button
                                        className="helpful-btn px-3 py-1 rounded bg-teal-200 text-sm"
                                        onClick={() => handleHelpful(tool.tip_id)}
                                    >
                                        ✅ Mark Helpful
                                    </button>
                                </div>
                            </div>
                        );
                    })
                ) : (
                    <div className="card p-6 col-span-full">No tools available for this period.</div>
                )}
            </section>
        </>
    );
}
