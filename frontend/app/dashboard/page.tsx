'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import { dashboardAPI } from '@/lib/api';

interface MoodEntry {
    date: string;
    mood: string;
}

interface HelpfulTip {
    date?: string;
    mood?: string;
    tip: string;
}

interface HomeData {
    streak: number;
    recent: MoodEntry[];
    mood: string;
    quote: string;
    helpful: HelpfulTip[];
}

export default function DashboardPage() {
    const [data, setData] = useState<HomeData | null>(null);
    const [period, setPeriod] = useState('last10');

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
        fetchDashboardData('last10');
    }, [user, isLoading]);

    const fetchDashboardData = async (nextPeriod: string) => {
        try {
            const response = await dashboardAPI.getHomeData(nextPeriod);
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
    const handlePeriodChange = (nextPeriod: string) => {
        setPeriod(nextPeriod);
        fetchDashboardData(nextPeriod);
    };

    return (
        <>
            <header className="flex justify-between items-center mb-8">
                <div className="flex items-center gap-4">
                    <h1 className="text-2xl font-bold text-theme-text-main">Dashboard</h1>
                </div>
                <div className="flex items-center gap-4">
                    <div className="relative inline-block text-left">
                        <select
                            id="filterSelect"
                            value={period}
                            onChange={(event) => handlePeriodChange(event.target.value)}
                            className="bg-theme-panel border border-theme-primary/30 rounded-lg shadow-sm px-4 py-2 text-sm font-medium text-theme-text-main"
                        >
                            <option value="last10">Last 10 Entries</option>
                            <option value="last7days">Last 7 Days</option>
                            <option value="all">All Time</option>
                        </select>
                    </div>
                    <a
                        href="/chat"
                        className="px-5 py-2 btn-primary rounded-lg shadow-md hover:opacity-90 transition"
                    >
                        Daily Check-in
                    </a>
                </div>
            </header>

            <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-8">
                <div className="card">
                    <h3 className="text-xl font-semibold mb-3 text-theme-text-main">Mood Snapshot</h3>
                    <p className="text-theme-text-subtle">
                        {data?.mood || 'No data yet.'}
                    </p>
                </div>
                <div className="card">
                    <h3 className="text-xl font-semibold mb-3 text-theme-text-main">Streak Tracker</h3>
                    <p className="text-theme-text-subtle">
                        {data?.streak ? `${data.streak} days` : '0 days'}
                    </p>
                </div>
                <div className="card">
                    <h3 className="text-xl font-semibold mb-3 text-theme-text-main">Quote of the Day</h3>
                    <p className="text-theme-text-subtle">
                        {data?.quote || 'Stay positive!'}
                    </p>
                </div>
                <div className="card col-span-1 md:col-span-2">
                    <h3 className="text-xl font-semibold mb-3 text-theme-text-main">Recent Check-ins</h3>
                    <ul className="list-disc pl-5 max-h-64 overflow-y-auto text-theme-text-subtle">
                        {data?.recent && data.recent.length > 0 ? (
                            data.recent.map((entry, index) => (
                                <li key={`${entry.date}-${index}`}>{entry.date} - {entry.mood}</li>
                            ))
                        ) : (
                            <li>No recent check-ins.</li>
                        )}
                    </ul>
                </div>
                <div className="card col-span-1 md:col-span-2">
                    <h3 className="text-xl font-semibold mb-3 text-theme-text-main">Helpful Tips</h3>
                    <ul className="list-disc pl-5 max-h-64 overflow-y-auto text-theme-text-subtle">
                        {data?.helpful && data.helpful.length > 0 ? (
                            data.helpful.map((item, index) => (
                                <li key={`${item.tip}-${index}`}>
                                    {(item.date || '').trim()} {item.mood ? `- ${item.mood}` : ''}: {item.tip}
                                </li>
                            ))
                        ) : (
                            <li>No helpful tips saved yet.</li>
                        )}
                    </ul>
                </div>
            </section>
        </>
    );
}
