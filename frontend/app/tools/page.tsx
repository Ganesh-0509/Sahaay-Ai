'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import Layout from '@/components/layout/Layout';

interface RecommendedTool {
    id: string;
    title: string;
    reason: string;
    priority: number;
}

export default function CopingToolsPage() {
    const [activeExercise, setActiveExercise] = useState<string | null>(null);
    const [breathCount, setBreathCount] = useState(0);
    const [isBreathing, setIsBreathing] = useState(false);
    const [recommendedTools, setRecommendedTools] = useState<RecommendedTool[]>([]);
    const [recommendationMessage, setRecommendationMessage] = useState('Loading personalized tools...');
    const [isLoadingRecommendations, setIsLoadingRecommendations] = useState(true);

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
        fetchRecommendedTools();
    }, [user, authLoading]);

    const fetchRecommendedTools = async () => {
        try {
            setIsLoadingRecommendations(true);
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/recommended_tools`, {
                credentials: 'include',
            });
            const data = await response.json();

            if (data.ok && data.tools) {
                setRecommendedTools(data.tools);
                setRecommendationMessage(data.message || 'Personalized tools for you');
            } else {
                // Fallback to default tools
                setRecommendedTools([
                    { id: 'breathing', title: 'Breathing Exercise', reason: 'Helps calm your nervous system', priority: 1 },
                    { id: 'meditation', title: 'Quick Meditation', reason: 'Promotes mindfulness', priority: 2 },
                    { id: 'grounding', title: '5-4-3-2-1 Grounding', reason: 'Anchors you in the present', priority: 3 },
                ]);
                setRecommendationMessage('General wellness tools');
            }
            setIsLoadingRecommendations(false);
        } catch (error) {
            console.error('Failed to fetch recommendations:', error);
            setRecommendationMessage('General wellness tools');
            setIsLoadingRecommendations(false);
        }
    };

    const startBreathingExercise = async () => {
        setIsBreathing(true);
        setBreathCount(0);

        for (let i = 0; i < 5; i++) {
            setBreathCount(i + 1);
            await new Promise(resolve => setTimeout(resolve, 8000)); // 8 seconds per breath
        }

        setIsBreathing(false);
        alert('Great job! You completed 5 breathing cycles. 🌟');
    };

    if (authLoading || !user) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="loading-shimmer h-8 w-48 rounded-lg"></div>
            </div>
        );
    }

    const allTools = [
        {
            id: 'breathing',
            title: 'Breathing Exercise',
            emoji: '🫁',
            color: 'from-theme-primary/80 to-theme-primary',
            description: '4-4-4 breathing to calm your mind',
        },
        {
            id: 'meditation',
            title: 'Quick Meditation',
            emoji: '🧘',
            color: 'from-theme-secondary/80 to-theme-secondary',
            description: '5-minute guided relaxation',
        },
        {
            id: 'grounding',
            title: '5-4-3-2-1 Grounding',
            emoji: '🌍',
            color: 'from-theme-success/80 to-theme-success',
            description: 'Ground yourself in the present moment',
        },
        {
            id: 'journal',
            title: 'Gratitude Journal',
            emoji: '📝',
            color: 'from-theme-accent/80 to-theme-warning',
            description: 'Write down 3 things you are grateful for',
        },
        {
            id: 'music',
            title: 'Calming Sounds',
            emoji: '🎵',
            color: 'from-blue-400 to-blue-600',
            description: 'Soothing nature sounds and music',
        },
        {
            id: 'affirmations',
            title: 'Positive Affirmations',
            emoji: '💪',
            color: 'from-pink-400 to-pink-600',
            description: 'Boost your confidence and mood',
        },
    ];

    const affirmations = [
        "I am worthy of love and respect.",
        "I choose peace over worry.",
        "My feelings are valid.",
        "I am doing the best I can.",
        "This too shall pass.",
        "I am stronger than I think.",
        "I deserve to be happy.",
        "I am enough, just as I am.",
    ];

    return (
        <Layout>
            {/* Header */}
            <div className="mb-8 animate-fade-in-up">
                <h1 className="text-4xl font-bold text-theme-text-main mb-2">
                    Coping Tools 🧘
                </h1>
                <p className="text-theme-text-subtle text-lg">
                    {recommendationMessage}
                </p>
            </div>

            {/* AI Recommended Tools */}
            {isLoadingRecommendations ? (
                <div className="mb-8">
                    <div className="glass-card p-8">
                        <div className="text-center">
                            <div className="loading-shimmer h-6 w-64 mx-auto mb-4 rounded"></div>
                            <div className="loading-shimmer h-4 w-48 mx-auto rounded"></div>
                        </div>
                    </div>
                </div>
            ) : recommendedTools.length > 0 && (
                <div className="mb-8 animate-fade-in-up">
                    <div className="glass-card p-6 bg-gradient-to-br from-theme-accent/10 to-theme-accent/5 border-2 border-theme-accent/30">
                        <h2 className="text-2xl font-bold text-theme-text-main mb-4 flex items-center gap-2">
                            <span>🤖</span>
                            AI Recommended For You
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {recommendedTools.map((recTool) => {
                                const toolInfo = allTools.find(t => t.id === recTool.id);
                                if (!toolInfo) return null;

                                return (
                                    <div
                                        key={recTool.id}
                                        onClick={() => setActiveExercise(recTool.id)}
                                        className="group cursor-pointer p-5 rounded-xl bg-white/5 hover:bg-white/10 border-2 border-theme-accent/40 hover:border-theme-accent transition-all hover:scale-105"
                                    >
                                        <div className="flex items-center gap-3 mb-3">
                                            <div className="text-4xl">{toolInfo.emoji}</div>
                                            <div className="flex-1">
                                                <h3 className="font-bold text-theme-text-main">{recTool.title}</h3>
                                                <div className="text-xs text-theme-accent">Priority #{recTool.priority}</div>
                                            </div>
                                        </div>
                                        <p className="text-sm text-theme-text-subtle italic">
                                            {recTool.reason}
                                        </p>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            )}

            {/* Active Exercise Modal */}
            {activeExercise && (
                <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setActiveExercise(null)}>
                    <div className="glass-card max-w-2xl w-full p-8" onClick={(e) => e.stopPropagation()}>
                        {activeExercise === 'breathing' && (
                            <div className="text-center">
                                <div className="text-6xl mb-6">🫁</div>
                                <h2 className="text-3xl font-bold text-theme-text-main mb-4">Breathing Exercise</h2>
                                <p className="text-theme-text-subtle mb-8">
                                    Follow the 4-4-4 breathing pattern: Inhale for 4 seconds, hold for 4 seconds, exhale for 4 seconds.
                                </p>
                                {isBreathing && (
                                    <div className="mb-6">
                                        <div className="text-5xl font-bold text-theme-primary mb-4">{breathCount}/5</div>
                                        <div className="w-full h-4 bg-theme-hover/30 rounded-full overflow-hidden">
                                            <div className="h-full bg-gradient-to-r from-theme-primary to-theme-primary-light animate-shimmer" style={{ width: `${(breathCount / 5) * 100}%` }}></div>
                                        </div>
                                    </div>
                                )}
                                <button
                                    onClick={startBreathingExercise}
                                    disabled={isBreathing}
                                    className="btn btn-primary"
                                >
                                    {isBreathing ? 'Breathing...' : 'Start Exercise'}
                                </button>
                            </div>
                        )}

                        {activeExercise === 'meditation' && (
                            <div className="text-center">
                                <div className="text-6xl mb-6">🧘</div>
                                <h2 className="text-3xl font-bold text-theme-text-main mb-4">Quick Meditation</h2>
                                <div className="text-left space-y-4 text-theme-text-subtle mb-8">
                                    <p>• Find a comfortable seated position</p>
                                    <p>• Close your eyes and take a deep breath</p>
                                    <p>• Focus on your breath, let thoughts pass</p>
                                    <p>• Continue for 5 minutes</p>
                                    <p>• Slowly open your eyes when ready</p>
                                </div>
                                <button onClick={() => setActiveExercise(null)} className="btn btn-secondary">
                                    I'm Ready to Begin
                                </button>
                            </div>
                        )}

                        {activeExercise === 'grounding' && (
                            <div className="text-center">
                                <div className="text-6xl mb-6">🌍</div>
                                <h2 className="text-3xl font-bold text-theme-text-main mb-4">5-4-3-2-1 Grounding</h2>
                                <div className="text-left space-y-4 text-theme-text-subtle mb-8">
                                    <p><strong className="text-theme-text-main">5 things</strong> you can see around you</p>
                                    <p><strong className="text-theme-text-main">4 things</strong> you can touch</p>
                                    <p><strong className="text-theme-text-main">3 things</strong> you can hear</p>
                                    <p><strong className="text-theme-text-main">2 things</strong> you can smell</p>
                                    <p><strong className="text-theme-text-main">1 thing</strong> you can taste</p>
                                </div>
                                <button onClick={() => setActiveExercise(null)} className="btn btn-primary">
                                    Close
                                </button>
                            </div>
                        )}

                        {activeExercise === 'affirmations' && (
                            <div className="text-center">
                                <div className="text-6xl mb-6">💪</div>
                                <h2 className="text-3xl font-bold text-theme-text-main mb-4">Positive Affirmations</h2>
                                <div className="space-y-3 mb-8">
                                    {affirmations.map((affirmation, index) => (
                                        <div key={index} className="p-4 rounded-xl bg-theme-hover/30 text-theme-text-main">
                                            {affirmation}
                                        </div>
                                    ))}
                                </div>
                                <button onClick={() => setActiveExercise(null)} className="btn btn-secondary">
                                    Close
                                </button>
                            </div>
                        )}

                        {(activeExercise === 'journal' || activeExercise === 'music') && (
                            <div className="text-center">
                                <div className="text-6xl mb-6">{activeExercise === 'journal' ? '📝' : '🎵'}</div>
                                <h2 className="text-3xl font-bold text-theme-text-main mb-4">
                                    {activeExercise === 'journal' ? 'Gratitude Journal' : 'Calming Sounds'}
                                </h2>
                                <p className="text-theme-text-subtle mb-8">
                                    {activeExercise === 'journal'
                                        ? 'This feature is coming soon! You can use the chat to journal your thoughts for now.'
                                        : 'This feature is coming soon! Try listening to calming music on your favorite platform.'}
                                </p>
                                <button onClick={() => setActiveExercise(null)} className="btn btn-outline">
                                    Close
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* All Tools Grid */}
            <h2 className="text-2xl font-bold text-theme-text-main mb-4">All Coping Tools</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                {allTools.map((tool, index) => (
                    <div
                        key={tool.id}
                        onClick={() => setActiveExercise(tool.id)}
                        className="group cursor-pointer animate-fade-in-up"
                        style={{ animationDelay: `${index * 100}ms` }}
                    >
                        <div className={`h-full p-6 rounded-2xl bg-gradient-to-br ${tool.color} hover:scale-105 transition-transform shadow-card hover:shadow-card-hover border-2 border-white/20`}>
                            <div className="text-5xl mb-4">{tool.emoji}</div>
                            <h3 className="text-xl font-bold text-white mb-2">{tool.title}</h3>
                            <p className="text-white/80 text-sm">{tool.description}</p>
                            <div className="mt-4 text-white/60 text-sm group-hover:text-white/90 transition-colors">
                                Click to start →
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Emergency Resources */}
            <div className="glass-card animate-fade-in-up bg-gradient-to-br from-theme-danger/10 to-theme-danger/5 border-2 border-theme-danger/30" style={{ animationDelay: '600ms' }}>
                <div className="flex items-start gap-4">
                    <div className="text-5xl">🆘</div>
                    <div className="flex-1">
                        <h2 className="text-2xl font-bold text-theme-text-main mb-3">Emergency Resources</h2>
                        <p className="text-theme-text-subtle mb-4">
                            If you are in crisis or need immediate support, please reach out to these resources:
                        </p>
                        <div className="space-y-2 text-theme-text-main">
                            <p><strong>National Suicide Prevention Lifeline (US):</strong> 1-800-273-8255</p>
                            <p><strong>Crisis Text Line:</strong> Text HOME to 741741</p>
                            <p><strong>International Association for Suicide Prevention:</strong> <a href="https://www.iasp.info/resources/Crisis_Centres/" target="_blank" rel="noopener noreferrer" className="text-theme-primary hover:underline">Find a helpline</a></p>
                        </div>
                    </div>
                </div>
            </div>
        </Layout>
    );
}
