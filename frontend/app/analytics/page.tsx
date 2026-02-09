'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';

interface AnalyticsData {
    kpis?: {
        avg_sentiment?: number;
        most_common_mood?: string;
        total_checkins?: number;
    };
    mood_counts?: Record<string, number>;
    trend_by_mood?: Record<string, { date: string; sentiment: number }[]>;
    entries_by_date?: Record<string, any[]>;
    most_frequent_words?: [string, number][];
    word_moods?: Record<string, string>;
    top_tips?: string[];
    insights?: string;
}

const moodColors: Record<string, string> = {
    happy: '#FFD166',
    calm: '#4AACEA',
    neutral: '#8A6CFF',
    anxious: '#FF5C8D',
    sad: '#6B7280',
    stressed: '#F87171',
    lonely: '#A99BDB',
    angry: '#FF7C43',
    excited: '#2DD4BF',
    content: '#34D399',
    normal: '#60D394',
};

const moodDisplayNames: Record<string, string> = {
    normal: 'Normal',
    happy: 'Happy',
    calm: 'Calm',
    neutral: 'Neutral',
    anxious: 'Anxious',
    sad: 'Sad',
    stressed: 'Stressed',
    lonely: 'Lonely',
    angry: 'Angry',
    excited: 'Excited',
    content: 'Content',
};

function mapMoodLabel(rawLabel: string) {
    if (!rawLabel) return '';
    if (rawLabel.toLowerCase().startsWith('mixed:')) return 'Normal';
    const key = rawLabel.toLowerCase().trim();
    return moodDisplayNames[key] || key.charAt(0).toUpperCase() + key.slice(1);
}

function formatEntryTime(timestamp: any) {
    try {
        if (!timestamp) return '—';
        if (typeof timestamp === 'string') {
            const parsed = new Date(timestamp);
            if (!Number.isNaN(parsed.getTime())) {
                return parsed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            }
        }
        if (timestamp.seconds) {
            const parsed = new Date(timestamp.seconds * 1000);
            return parsed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
        return String(timestamp);
    } catch {
        return '—';
    }
}

export default function AnalyticsPage() {
    const [period, setPeriod] = useState('last10');
    const [analytics, setAnalytics] = useState<AnalyticsData>({});
    const [selectedDate, setSelectedDate] = useState<string | null>(null);
    const [toast, setToast] = useState<string>('');
    const modalRef = useRef<HTMLDivElement>(null);
    const doughnutRef = useRef<any>(null);
    const trendRef = useRef<any>(null);
    const doughnutCanvasRef = useRef<HTMLCanvasElement>(null);
    const trendCanvasRef = useRef<HTMLCanvasElement>(null);

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
    }, [user, authLoading, period]);

    const fetchAnalytics = async () => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/analytics_data?period=${encodeURIComponent(period)}`, {
                credentials: 'include',
            });
            const data = await response.json();
            setAnalytics(data || {});
        } catch (error) {
            console.error('Failed to fetch analytics:', error);
        }
    };

    const totalCount = useMemo(() => {
        return Object.values(analytics.mood_counts || {}).reduce((acc, value) => acc + value, 0);
    }, [analytics.mood_counts]);

    useEffect(() => {
        const renderCharts = async () => {
            if (!analytics.mood_counts || !analytics.trend_by_mood) return;
            const Chart = (await import('chart.js/auto')).default;

            const labels = Object.keys(analytics.mood_counts);
            const values = Object.values(analytics.mood_counts);

            if (doughnutCanvasRef.current) {
                if (doughnutRef.current) doughnutRef.current.destroy();
                doughnutRef.current = new Chart(doughnutCanvasRef.current, {
                    type: 'doughnut',
                    data: {
                        labels,
                        datasets: [
                            {
                                data: values,
                                backgroundColor: labels.map((label) => {
                                    let base = label;
                                    if (base.startsWith('mixed:')) {
                                        const rest = base.split(':')[1] || '';
                                        base = rest.split('/')[0].split('(')[0];
                                    }
                                    base = base.toLowerCase();
                                    return moodColors[base] || '#8A6CFF';
                                }),
                                borderColor: '#F0EBFF',
                                borderWidth: 2,
                                hoverOffset: 15,
                            },
                        ],
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        cutout: '65%',
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    label: (ctx) => {
                                        const val = ctx.raw as number;
                                        const perc = totalCount > 0 ? ((val / totalCount) * 100).toFixed(1) : '0';
                                        const displayLabel = mapMoodLabel(ctx.label || '');
                                        return `${displayLabel}: ${val} (${perc}%)`;
                                    },
                                },
                                backgroundColor: 'rgba(0,0,0,0.8)',
                                padding: 12,
                            },
                            legend: {
                                position: 'bottom',
                                labels: {
                                    color: '#F0EBFF',
                                    padding: 15,
                                    font: { size: 12, weight: 'bold' },
                                },
                            },
                            title: {
                                display: true,
                                text: 'Mood Distribution',
                                color: '#F0EBFF',
                                font: { size: 16, weight: 'bold' },
                                padding: { top: 10, bottom: 20 },
                            },
                        },
                    },
                });
            }

            if (trendCanvasRef.current) {
                if (trendRef.current) trendRef.current.destroy();
                const datesSet = new Set<string>();
                Object.values(analytics.trend_by_mood || {}).forEach((trend) =>
                    trend.forEach((item) => datesSet.add(item.date))
                );
                const sortedDates = Array.from(datesSet).sort();
                const datasets: any[] = [];
                Object.entries(analytics.trend_by_mood || {}).forEach(([mood, trend]) => {
                    const dataMap = new Map(trend.map((item) => [item.date, item.sentiment]));
                    let base = mood;
                    if (base.startsWith('mixed:')) {
                        const rest = base.split(':')[1] || '';
                        base = rest.split('/')[0].split('(')[0];
                    }
                    base = base.toLowerCase();
                    const color = moodColors[base] || '#6B7280';
                    datasets.push({
                        label: mapMoodLabel(mood),
                        data: sortedDates.map((date) => dataMap.get(date)),
                        borderColor: color,
                        backgroundColor: `${color}33`,
                        fill: false,
                        tension: 0.3,
                        pointRadius: 4,
                    });
                });
                trendRef.current = new Chart(trendCanvasRef.current, {
                    type: 'line',
                    data: { labels: sortedDates, datasets },
                    options: {
                        responsive: true,
                        interaction: { mode: 'index', intersect: false },
                        scales: {
                            y: {
                                beginAtZero: false,
                                title: { display: true, text: 'Average Sentiment', color: '#F0EBFF' },
                                ticks: { color: '#F0EBFF' },
                                grid: { color: 'rgba(240, 235, 255, 0.1)' },
                            },
                            x: {
                                title: { display: true, text: 'Date', color: '#F0EBFF' },
                                ticks: { color: '#F0EBFF' },
                                grid: { color: 'rgba(240, 235, 255, 0.1)' },
                            },
                        },
                        plugins: {
                            legend: { labels: { color: '#F0EBFF' } },
                        },
                        onClick: (evt: any) => {
                            const points = trendRef.current.getElementsAtEventForMode(evt, 'nearest', { intersect: true }, true);
                            if (points.length) {
                                const date = trendRef.current.data.labels[points[0].index] as string;
                                setSelectedDate(date);
                            }
                        },
                    },
                });
            }
        };

        renderCharts();
    }, [analytics, totalCount]);

    const handleCopy = async (tip: string) => {
        try {
            await navigator.clipboard.writeText(tip);
            setToast('Copied tip!');
            setTimeout(() => setToast(''), 2000);
        } catch {
            setToast('Copy failed');
            setTimeout(() => setToast(''), 2000);
        }
    };

    const entriesForDate = selectedDate ? analytics.entries_by_date?.[selectedDate] || [] : [];

    if (authLoading || !user) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="loading-shimmer h-8 w-48 rounded-lg"></div>
            </div>
        );
    }

    return (
        <>
            <header className="flex justify-between items-center mb-8 animated-item">
                <div className="flex items-center gap-4">
                    <h1 className="text-3xl font-bold text-theme-text-main">Analytics</h1>
                </div>

                <div className="flex items-center gap-4">
                    <select
                        id="filterSelect"
                        value={period}
                        onChange={(event) => setPeriod(event.target.value)}
                        className="bg-theme-panel border border-theme-primary/30 rounded-xl shadow-lg px-4 py-2 text-sm font-medium text-theme-text-main"
                    >
                        <option value="last10">Last 10 Entries</option>
                        <option value="30days">Last 30 Days</option>
                        <option value="all">All Time</option>
                    </select>
                </div>
            </header>

            <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="card animated-item text-center" style={{ animationDelay: '0.1s' }}>
                    <h2 className="text-lg font-semibold mb-2 text-theme-text-main">Avg Sentiment</h2>
                    <p className="text-3xl font-bold text-theme-accent">{analytics.kpis?.avg_sentiment ?? '--'}</p>
                </div>
                <div className="card animated-item text-center" style={{ animationDelay: '0.2s' }}>
                    <h2 className="text-lg font-semibold mb-2 text-theme-text-main">Most Frequent Mood</h2>
                    <p className="text-3xl font-bold text-theme-primary">{mapMoodLabel(analytics.kpis?.most_common_mood || '--')}</p>
                </div>
                <div className="card animated-item text-center" style={{ animationDelay: '0.3s' }}>
                    <h2 className="text-lg font-semibold mb-2 text-theme-text-main">Total Check-ins</h2>
                    <p className="text-3xl font-bold text-theme-secondary">{analytics.kpis?.total_checkins ?? 0}</p>
                </div>
            </section>

            <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                <div className="card animated-item" style={{ animationDelay: '0.4s' }}>
                    <h2 className="text-lg font-semibold mb-3 text-theme-text-main">Mood Distribution</h2>
                    <canvas ref={doughnutCanvasRef}></canvas>
                </div>
                <div className="card animated-item" style={{ animationDelay: '0.5s' }}>
                    <h2 className="text-lg font-semibold mb-3 text-theme-text-main">Mood Trend</h2>
                    <canvas ref={trendCanvasRef}></canvas>
                </div>
            </section>

            <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="card animated-item" style={{ animationDelay: '0.6s' }}>
                    <h2 className="text-lg font-semibold mb-3 text-theme-text-main">Top Words</h2>
                    <div className="flex flex-wrap gap-2 justify-center py-4">
                        {analytics.most_frequent_words && analytics.most_frequent_words.length > 0 ? (
                            analytics.most_frequent_words.map(([word, count]) => (
                                <span
                                    key={word}
                                    className="p-1 px-2 rounded-full text-white font-semibold shadow-lg"
                                    style={{
                                        fontSize: `${12 + Math.log(count) * 4}px`,
                                        backgroundColor: moodColors[(analytics.word_moods && analytics.word_moods[word]) || 'neutral'] || '#8A6CFF',
                                    }}
                                >
                                    {word}
                                </span>
                            ))
                        ) : (
                            <p className="text-sm italic text-theme-text-subtle">More check-ins needed.</p>
                        )}
                    </div>
                </div>

                <div className="card animated-item" style={{ animationDelay: '0.7s' }}>
                    <h2 className="text-lg font-semibold mb-3 text-theme-text-main">Top 3 Tips</h2>
                    <ul className="list-disc list-inside text-theme-text-subtle">
                        {analytics.top_tips && analytics.top_tips.length > 0 ? (
                            analytics.top_tips.map((tip) => (
                                <li
                                    key={tip}
                                    className="p-3 my-2 hover:bg-theme-primary/20 rounded-xl cursor-pointer transition-all duration-200 border border-theme-primary/20"
                                    onClick={() => handleCopy(tip)}
                                    title="Click to copy"
                                >
                                    {tip}
                                </li>
                            ))
                        ) : (
                            <li>No helpful tips found.</li>
                        )}
                    </ul>
                </div>
            </section>

            <section className="card animated-item mb-8" style={{ animationDelay: '0.8s' }}>
                <h2 className="text-lg font-semibold mb-3 text-theme-text-main">Insights</h2>
                <p className="text-theme-text-main italic">{analytics.insights || 'Check in more often to get insights.'}</p>
            </section>

            {selectedDate && (
                <div
                    ref={modalRef}
                    className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
                    onClick={() => setSelectedDate(null)}
                >
                    <div
                        className="bg-theme-panel backdrop-blur-xl rounded-2xl shadow-2xl p-6 w-full max-w-2xl max-h-full overflow-y-auto border border-theme-primary/30"
                        onClick={(event) => event.stopPropagation()}
                    >
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-2xl font-bold text-theme-text-main">Entries for {selectedDate}</h3>
                            <button
                                onClick={() => setSelectedDate(null)}
                                className="text-theme-text-subtle hover:text-theme-text-main text-3xl transition-colors"
                            >
                                &times;
                            </button>
                        </div>
                        <div className="space-y-4">
                            {entriesForDate.length > 0 ? (
                                entriesForDate.map((entry, index) => (
                                    <div
                                        key={`${selectedDate}-${index}`}
                                        className="bg-theme-panel/50 backdrop-blur-sm p-4 rounded-xl shadow-lg border border-theme-primary/20"
                                    >
                                        <div className="flex items-center space-x-2 mb-2">
                                            <span className="font-bold text-lg capitalize text-theme-text-main">
                                                {entry.mood_label || entry.mood_dominant || 'Neutral'}
                                            </span>
                                            <span className="text-theme-text-subtle text-sm">{formatEntryTime(entry.timestamp)}</span>
                                            <span
                                                className="text-sm rounded-full px-2 py-0.5 text-white shadow-md"
                                                style={{
                                                    backgroundColor:
                                                        moodColors[(entry.mood_dominant || entry.mood_label || 'neutral').toLowerCase()] || '#8A6CFF',
                                                }}
                                            >
                                                Sentiment: {entry.avg_sentiment ?? entry.sentiment ?? 0}
                                            </span>
                                        </div>
                                        <p className="text-theme-text-main">{entry.last_text || entry.text || 'No text provided.'}</p>
                                    </div>
                                ))
                            ) : (
                                <p className="text-center text-theme-text-subtle">No entries found for this date.</p>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {toast && (
                <div className="fixed bottom-5 right-5 z-[100] px-4 py-2 rounded-lg shadow-xl text-white bg-green-500">
                    {toast}
                </div>
            )}
        </>
    );
}
