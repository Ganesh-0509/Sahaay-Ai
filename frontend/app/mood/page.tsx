'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';

interface MoodEntry {
    date: string;
    mood: string;
    summary?: string;
}

interface WeeklyEntry {
    date: string;
    mood?: string;
    summary?: string;
    avg_sentiment?: number;
}

interface DailyJournalResponse {
    ok?: boolean;
    entry?: {
        summary?: string;
    };
}

const moodMap: Record<string, { emoji: string; badge: string }> = {
    happy: { emoji: '😊', badge: 'bg-yellow-200 text-yellow-800' },
    sad: { emoji: '☹', badge: 'bg-blue-100 text-blue-800' },
    angry: { emoji: '😡', badge: 'bg-red-100 text-red-800' },
    anxious: { emoji: '😰', badge: 'bg-orange-100 text-orange-800' },
    normal: { emoji: '😐', badge: 'bg-green-100 text-green-800' },
    neutral: { emoji: '😶', badge: 'bg-gray-100 text-gray-800' },
    excited: { emoji: '🤩', badge: 'bg-pink-100 text-pink-800' },
};

function cleanSummaryGlobal(raw: string) {
    if (!raw) return '';
    let text = raw.toString();
    try {
        const parsed = JSON.parse(text);
        if (parsed && typeof parsed === 'object') {
            if (parsed.response) return parsed.response.toString();
            return Object.values(parsed).join(' ');
        }
    } catch {
        text = text.replace(/json\s*/i, '').replace(/<[^>]*>/g, '').trim();
    }
    text = text.replace(/\n/g, '\n').replace(/^"|"$/g, '');
    return text;
}

function toTitleCase(value: string) {
    if (!value) return '';
    return value.charAt(0).toUpperCase() + value.slice(1);
}

export default function MoodJournalPage() {
    const [entries, setEntries] = useState<MoodEntry[]>([]);
    const [dailySummary, setDailySummary] = useState('');
    const [weeklyEntries, setWeeklyEntries] = useState<WeeklyEntry[]>([]);
    const [note, setNote] = useState('');
    const [noteStatus, setNoteStatus] = useState('');
    const [reflection, setReflection] = useState('');
    const [dailyMeta, setDailyMeta] = useState({
        todayMood: '—',
        summaryLength: '—',
        sentimentScore: '—',
        intensityPercent: 0,
    });
    const [weeklyMeta, setWeeklyMeta] = useState({
        mostFreqMood: '—',
        stability: '—',
        streak: '—',
    });

    const barChartRef = useRef<any>(null);
    const pieChartRef = useRef<any>(null);
    const lineChartRef = useRef<any>(null);
    const barCanvasRef = useRef<HTMLCanvasElement>(null);
    const pieCanvasRef = useRef<HTMLCanvasElement>(null);
    const lineCanvasRef = useRef<HTMLCanvasElement>(null);

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
        fetchMoodEntries();
        fetchDailyJournal();
        fetchWeeklySummaries();
        const interval = setInterval(fetchWeeklySummaries, 30000);
        return () => clearInterval(interval);
    }, [user, authLoading]);

    const fetchMoodEntries = async () => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/mood_data`, {
                credentials: 'include',
            });
            const data = await response.json();
            setEntries(Array.isArray(data.entries) ? data.entries : []);
        } catch (error) {
            console.error('Failed to fetch mood data:', error);
        }
    };

    const fetchDailyJournal = async () => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/daily_journal`, {
                credentials: 'include',
            });
            let data: DailyJournalResponse | null = null;
            try {
                data = await response.json();
            } catch {
                data = null;
            }

            if (data?.ok && data.entry?.summary) {
                const summaryRaw = cleanSummaryGlobal(data.entry.summary || '');
                setDailySummary(summaryRaw);
                return;
            }

            const listResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/daily_journal_list?limit=7`, {
                credentials: 'include',
            });
            const listData = await listResponse.json();
            if (listData?.ok && Array.isArray(listData.entries)) {
                const today = new Date();
                const yyyy = today.getFullYear();
                const mm = String(today.getMonth() + 1).padStart(2, '0');
                const dd = String(today.getDate()).padStart(2, '0');
                const todayKey = `${yyyy}-${mm}-${dd}`;
                const found = listData.entries.find((entry: WeeklyEntry) =>
                    (entry.date || '') === todayKey || (entry.date || '').startsWith(todayKey)
                );
                if (found?.summary) {
                    setDailySummary(cleanSummaryGlobal(found.summary));
                    return;
                }
            }

            setDailySummary('No summary for today.');
        } catch (error) {
            console.error('Failed to fetch daily journal:', error);
        }
    };

    const fetchWeeklySummaries = async () => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/daily_journal_list?limit=7`, {
                credentials: 'include',
            });
            const data = await response.json();
            setWeeklyEntries(Array.isArray(data.entries) ? data.entries : []);
        } catch (error) {
            console.error('Failed to fetch weekly summaries:', error);
        }
    };

    const moodCounts = useMemo(() => {
        const counts: Record<string, number> = {};
        weeklyEntries.forEach((entry) => {
            const key = (entry.mood || 'neutral').toLowerCase();
            counts[key] = (counts[key] || 0) + 1;
        });
        return counts;
    }, [weeklyEntries]);

    useEffect(() => {
        if (weeklyEntries.length === 0) {
            setWeeklyMeta({ mostFreqMood: '—', stability: '—', streak: '—' });
            return;
        }

        const entriesByDate = [...weeklyEntries]
            .map((entry) => ({
                ...entry,
                mood: (entry.mood || 'neutral').toLowerCase(),
                summary: cleanSummaryGlobal(entry.summary || ''),
            }))
            .sort((a, b) => (b.date || '').localeCompare(a.date || ''));

        const [mostFreq] = Object.entries(moodCounts).sort((a, b) => b[1] - a[1]);
        const mostFreqLabel = mostFreq ? mostFreq[0] : '—';

        let changes = 0;
        for (let i = 1; i < entriesByDate.length; i += 1) {
            if (entriesByDate[i].mood !== entriesByDate[i - 1].mood) changes += 1;
        }
        const totalDays = entriesByDate.length;
        const stability = totalDays > 1 ? Math.round((1 - changes / (totalDays - 1)) * 100) : 100;

        const streak = totalDays;

        setWeeklyMeta({
            mostFreqMood: `${moodMap[mostFreqLabel]?.emoji || '🙂'} ${toTitleCase(mostFreqLabel)}`,
            stability: `${stability}%`,
            streak: `${streak} day${streak === 1 ? '' : 's'}`,
        });

        const today = new Date();
        const yyyy = today.getFullYear();
        const mm = String(today.getMonth() + 1).padStart(2, '0');
        const dd = String(today.getDate()).padStart(2, '0');
        const todayKey = `${yyyy}-${mm}-${dd}`;
        const todayEntry = entriesByDate.find((entry) =>
            entry.date === todayKey || (entry.date || '').startsWith(todayKey)
        );
        if (todayEntry) {
            const sentenceCount = (todayEntry.summary || '')
                .split(/[\.\!\?]\s+/)
                .filter(Boolean).length;
            const sentimentValue = Number(todayEntry.avg_sentiment ?? 0);
            const intensity = Math.round(((sentimentValue + 1) / 2) * 100);
            setDailyMeta({
                todayMood: toTitleCase(todayEntry.mood || 'neutral'),
                summaryLength: `${sentenceCount || 0} sentence${sentenceCount === 1 ? '' : 's'}`,
                sentimentScore: todayEntry.avg_sentiment !== undefined ? sentimentValue.toFixed(2) : 'N/A',
                intensityPercent: Number.isFinite(intensity) ? intensity : 0,
            });
        }
    }, [weeklyEntries, moodCounts]);

    useEffect(() => {
        const renderCharts = async () => {
            if (weeklyEntries.length === 0) return;
            const Chart = (await import('chart.js/auto')).default;
            const labels = Object.keys(moodCounts);
            const values = labels.map((label) => moodCounts[label]);

            if (barCanvasRef.current) {
                if (barChartRef.current) barChartRef.current.destroy();
                barChartRef.current = new Chart(barCanvasRef.current, {
                    type: 'bar',
                    data: {
                        labels: labels.map((label) => toTitleCase(label)),
                        datasets: [
                            {
                                label: 'Days',
                                data: values,
                                backgroundColor: labels.map((label) => {
                                    if (label.includes('happy')) return 'rgba(99, 102, 241, 0.9)';
                                    if (label.includes('sad')) return 'rgba(59, 130, 246, 0.9)';
                                    if (label.includes('angry')) return 'rgba(239, 68, 68, 0.9)';
                                    if (label.includes('anxious')) return 'rgba(234, 88, 12, 0.9)';
                                    if (label.includes('normal')) return 'rgba(34, 197, 94, 0.9)';
                                    return 'rgba(139, 92, 246, 0.9)';
                                }),
                                borderRadius: 6,
                                barPercentage: 0.6,
                            },
                        ],
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: { duration: 600, easing: 'easeOutQuart' },
                        scales: {
                            x: { ticks: { color: '#cbd5e1' } },
                            y: { beginAtZero: true, ticks: { color: '#cbd5e1', precision: 0 } },
                        },
                        plugins: { legend: { display: false } },
                    },
                });
            }

            if (pieCanvasRef.current) {
                if (pieChartRef.current) pieChartRef.current.destroy();
                pieChartRef.current = new Chart(pieCanvasRef.current, {
                    type: 'pie',
                    data: {
                        labels: labels.map((label) => toTitleCase(label)),
                        datasets: [
                            {
                                data: values,
                                backgroundColor: labels.map((label) => {
                                    if (label.includes('happy')) return 'rgba(99, 102, 241, 0.9)';
                                    if (label.includes('sad')) return 'rgba(59, 130, 246, 0.9)';
                                    if (label.includes('angry')) return 'rgba(239, 68, 68, 0.9)';
                                    if (label.includes('anxious')) return 'rgba(234, 88, 12, 0.9)';
                                    if (label.includes('normal')) return 'rgba(34, 197, 94, 0.9)';
                                    return 'rgba(139, 92, 246, 0.9)';
                                }),
                            },
                        ],
                    },
                    options: { responsive: true, maintainAspectRatio: false, animation: { duration: 600 } },
                });
            }

            if (lineCanvasRef.current) {
                if (lineChartRef.current) lineChartRef.current.destroy();
                const trendLabels = [...weeklyEntries].reverse().map((entry) => entry.date);
                const trendValues = [...weeklyEntries]
                    .reverse()
                    .map((entry) => (entry.avg_sentiment !== undefined ? Number(entry.avg_sentiment) : 0));
                lineChartRef.current = new Chart(lineCanvasRef.current, {
                    type: 'line',
                    data: {
                        labels: trendLabels,
                        datasets: [
                            {
                                label: 'Sentiment',
                                data: trendValues,
                                borderColor: 'rgba(138,108,255,0.95)',
                                backgroundColor: 'rgba(138,108,255,0.12)',
                                tension: 0.3,
                                pointRadius: 4,
                            },
                        ],
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } },
                });
            }
        };

        renderCharts();
    }, [weeklyEntries, moodCounts]);

    const handleSaveNote = async () => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/save_journal_note`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ note }),
                credentials: 'include',
            });
            const data = await response.json();
            if (data.ok) {
                setNoteStatus('Saved');
                setTimeout(() => setNoteStatus(''), 2000);
            } else {
                alert(`Failed to save note: ${data.error || 'unknown'}`);
            }
        } catch (error) {
            console.error('Save note failed', error);
            alert('Save note failed');
        }
    };

    const handlePrintDaily = () => {
        const html = `
            <html><head><title>Daily Summary</title></head><body style="font-family:sans-serif;padding:20px;">
            <h1>Daily Summary</h1>
            <div>${dailySummary || 'No summary for today.'}</div>
            </body></html>`;
        const popup = window.open('', '_blank');
        if (!popup) {
            alert('Popup blocked. Please allow popups to print.');
            return;
        }
        popup.document.write(html);
        popup.document.close();
        popup.focus();
        setTimeout(() => popup.print(), 500);
    };

    const handleWeeklyReflection = async () => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/generate_weekly_reflection`, {
                method: 'POST',
                credentials: 'include',
            });
            const data = await response.json();
            if (data.ok) {
                setReflection(data.reflection || 'No reflection generated.');
            } else {
                alert(`Failed: ${data.error || 'unknown'}`);
            }
        } catch (error) {
            console.error('Reflection generation failed', error);
            alert('Reflection generation failed');
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
                    <h1 className="text-2xl font-bold text-theme-text-main">Mood Journal</h1>
                </div>
            </header>

            <section className="space-y-6">
                <div className="card">
                    <h3 className="text-xl font-semibold mb-3 text-theme-text-main">Recent Entries</h3>
                    <ul className="list-disc pl-5 max-h-64 overflow-y-auto text-theme-text-subtle">
                        {entries.length > 0 ? (
                            entries.map((entry) => (
                                <li key={`${entry.date}-${entry.mood}`}>[{entry.date}] {entry.mood}</li>
                            ))
                        ) : (
                            <li>No entries yet.</li>
                        )}
                    </ul>
                </div>

                <div className="card p-4 flex flex-col md:flex-row items-start md:items-center gap-4">
                    <div className="flex-1">
                        <div className="text-sm text-theme-text-subtle">Today's Mood</div>
                        <div className="text-2xl font-bold text-theme-text-main">{dailyMeta.todayMood}</div>
                    </div>
                    <div className="flex-1">
                        <div className="text-sm text-theme-text-subtle">Summary Length</div>
                        <div className="text-lg font-semibold text-theme-text-main">{dailyMeta.summaryLength}</div>
                    </div>
                    <div className="flex-1">
                        <div className="text-sm text-theme-text-subtle">Sentiment Score</div>
                        <div className="text-lg font-semibold text-theme-text-main">{dailyMeta.sentimentScore}</div>
                    </div>
                    <div className="w-full md:w-auto" style={{ minWidth: '220px' }}>
                        <div className="text-sm text-theme-text-subtle">Mood Intensity</div>
                        <div className="w-full bg-theme-hover/40 rounded-full h-3 mt-1 overflow-hidden">
                            <div
                                className="h-3 bg-gradient-to-r from-theme-primary to-theme-secondary"
                                style={{ width: `${dailyMeta.intensityPercent}%` }}
                            ></div>
                        </div>
                    </div>
                </div>

                <div className="card">
                    <h3 className="text-xl font-semibold mb-3 text-theme-text-main">Daily Summary</h3>
                    <div className="whitespace-pre-wrap text-theme-text-subtle">
                        {dailySummary || 'No summary for today.'}
                    </div>
                    <div className="mt-4">
                        <textarea
                            id="userNote"
                            rows={3}
                            value={note}
                            onChange={(event) => setNote(event.target.value)}
                            placeholder="Write a private note for today..."
                            className="w-full p-3 rounded-md bg-theme-panel text-theme-text-main"
                        />
                        <div className="flex items-center gap-3 mt-2">
                            <button onClick={handleSaveNote} className="btn-primary">
                                {noteStatus || 'Save Note'}
                            </button>
                            <button
                                onClick={handlePrintDaily}
                                className="btn-primary"
                                style={{ background: 'transparent', border: '1px solid rgba(138,108,255,0.25)' }}
                            >
                                Print / Save PDF
                            </button>
                        </div>
                    </div>
                </div>

                <div className="card">
                    <h3 className="text-xl font-semibold mb-3 text-theme-text-main">Last 7 Daily Summaries</h3>
                    <div id="weeklyAnalytics" className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        <div className="p-3 rounded-lg border border-theme-primary/30">
                            <div className="text-sm text-theme-text-subtle">Most Frequent Mood</div>
                            <div className="text-lg font-semibold mt-1 text-theme-text-main">{weeklyMeta.mostFreqMood}</div>
                        </div>
                        <div className="p-3 rounded-lg border border-theme-primary/30">
                            <div className="text-sm text-theme-text-subtle">Weekly Stability</div>
                            <div className="text-lg font-semibold mt-1 text-theme-text-main">{weeklyMeta.stability}</div>
                        </div>
                        <div className="p-3 rounded-lg border border-theme-primary/30">
                            <div className="text-sm text-theme-text-subtle">Mood Streak</div>
                            <div className="text-lg font-semibold mt-1 text-theme-text-main">{weeklyMeta.streak}</div>
                        </div>
                    </div>

                    {reflection && (
                        <div className="mt-3 p-3 rounded-md bg-theme-panel text-theme-text-main">
                            {reflection}
                        </div>
                    )}

                    <div className="overflow-x-auto">
                        {weeklyEntries.length > 0 ? (
                            <table className="w-full text-left border-collapse rounded-md overflow-hidden">
                                <thead>
                                    <tr className="text-sm text-theme-text-subtle bg-theme-panel/60">
                                        <th className="px-4 py-2 border-b border-theme-primary/20">Date</th>
                                        <th className="px-4 py-2 border-b border-theme-primary/20">Mood</th>
                                        <th className="px-4 py-2 border-b border-theme-primary/20">Summary</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {weeklyEntries.map((entry) => {
                                        const key = (entry.mood || 'neutral').toLowerCase();
                                        const summaryText = cleanSummaryGlobal(entry.summary || '');
                                        const shortText = summaryText.length > 140 ? `${summaryText.slice(0, 137)}…` : summaryText;
                                        return (
                                            <tr key={entry.date} className="align-top hover:bg-theme-hover/20">
                                                <td className="px-4 py-3 text-sm text-theme-text-main border-b border-theme-primary/10">{entry.date}</td>
                                                <td className="px-4 py-3 text-sm text-theme-text-main border-b border-theme-primary/10">
                                                    <span className={`inline-flex items-center gap-2 px-2 py-1 rounded-full text-xs font-medium ${moodMap[key]?.badge || 'bg-purple-100 text-purple-800'}`}>
                                                        <span>{moodMap[key]?.emoji || '🙂'}</span>
                                                        <span className="capitalize">{key}</span>
                                                    </span>
                                                </td>
                                                <td className="px-4 py-3 text-sm text-theme-text-subtle border-b border-theme-primary/10">
                                                    {shortText || 'No summary.'}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        ) : (
                            <p className="text-theme-text-subtle">No summaries available.</p>
                        )}
                    </div>

                    <button onClick={handleWeeklyReflection} className="btn-primary mt-3">
                        Generate Weekly Reflection
                    </button>
                </div>

                <div className="card mt-4">
                    <h3 className="text-xl font-semibold mb-3 text-theme-text-main">Week Mood Summary</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4" style={{ height: '180px' }}>
                        <div className="w-full" style={{ height: '180px' }}>
                            <canvas ref={barCanvasRef} height={180}></canvas>
                        </div>
                        <div className="w-full" style={{ height: '180px' }}>
                            <canvas ref={pieCanvasRef} height={180}></canvas>
                        </div>
                        <div className="w-full" style={{ height: '180px' }}>
                            <canvas ref={lineCanvasRef} height={180}></canvas>
                        </div>
                    </div>
                </div>
            </section>
        </>
    );
}
