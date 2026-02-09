'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import { chatAPI } from '@/lib/api';

interface Message {
    text: string;
    sender: 'user' | 'ai';
    mood?: string;
}

export default function ChatPage() {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const [isAnonymous, setIsAnonymous] = useState(false);
    const [crisisDetected, setCrisisDetected] = useState(false);
    const [consentStatus, setConsentStatus] = useState('Consent status unavailable');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const user = useAuthStore((state) => state.user);
    const authLoading = useAuthStore((state) => state.isLoading);
    const checkAuth = useAuthStore((state) => state.checkAuth);
    const logout = useAuthStore((state) => state.logout);
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
    }, [user, authLoading, router]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;
        const userMessage: Message = { text: input, sender: 'user' };
        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await chatAPI.sendMessage(input, 'en');
            const payload = response.data?.data || response.data;
            if (payload?.response) {
                setMessages((prev) => [...prev, { text: payload.response, sender: 'ai', mood: payload.mood }]);
                setCrisisDetected(Boolean(payload.crisis_detected));
            }
        } catch (error) {
            console.error('Failed to send message:', error);
            setMessages((prev) => [...prev, { text: 'Sorry, I encountered an error. Please try again.', sender: 'ai' }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSend();
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
            <div className="flex flex-col min-h-screen">
                <header className="flex justify-between items-center mb-6 bg-theme-panel p-4 rounded-2xl shadow animated-item">
                    <h1 className="text-xl font-semibold text-theme-text-main">Daily Check-in</h1>

                    <div className="flex items-center gap-3">
                        <a
                            href="/dashboard"
                            className="flex items-center gap-2 px-4 py-2 btn-primary rounded-lg shadow-md hover:opacity-90 transition"
                            aria-label="Dashboard"
                        >
                            🏠 Dashboard
                        </a>

                        <div className="relative inline-block text-left">
                            <button
                                onClick={() => setShowDropdown((prev) => !prev)}
                                className="flex items-center justify-center w-10 h-10 bg-theme-panel rounded-full focus:outline-none focus:ring-2 focus:ring-theme-accent transition"
                                aria-label="User menu"
                            >
                                👤
                            </button>

                            {showDropdown && (
                                <div className="origin-top-right absolute right-0 mt-2 w-56 rounded-md shadow-lg bg-theme-panel ring-1 ring-black ring-opacity-5 z-[1000]">
                                    <div className="py-1">
                                        <div className="px-4 py-2 text-sm">
                                            <span>{user.username || user.name || 'User'}</span>
                                        </div>
                                        <hr className="border-theme-primary mx-2" />
                                        <div className="block px-4 py-2 text-sm hover:bg-theme-accent transition">
                                            <div className="flex items-center justify-between">
                                                <span>Anonymous Mode</span>
                                                <label className="flex items-center cursor-pointer">
                                                    <div className="relative">
                                                        <input
                                                            type="checkbox"
                                                            checked={isAnonymous}
                                                            onChange={(event) => setIsAnonymous(event.target.checked)}
                                                            className="sr-only"
                                                        />
                                                        <div className="block bg-theme-hover w-10 h-6 rounded-full"></div>
                                                        <div className={`dot absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-all duration-300 ${isAnonymous ? 'translate-x-4' : ''}`}></div>
                                                    </div>
                                                </label>
                                            </div>
                                        </div>
                                        <button
                                            className="flex items-center gap-2 px-4 py-2 text-sm text-theme-pulse hover:bg-theme-pulse/20 hover:text-white rounded-md transition w-full"
                                            onClick={() => alert('Delete data is not available in this view.')}
                                        >
                                            🗑 Delete Data
                                        </button>
                                        <button
                                            className="flex items-center gap-2 px-4 py-2 text-sm hover:bg-theme-accent rounded-md transition w-full"
                                            onClick={async () => {
                                                await logout();
                                                router.push('/login');
                                            }}
                                        >
                                            🚪 Logout
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </header>

                {crisisDetected && (
                    <div className="bg-red-500 text-white p-4 rounded-xl shadow-md text-sm text-center mb-4">
                        <p>If you are in crisis, please reach out to local emergency services or a crisis hotline.</p>
                        <button className="underline font-bold mt-1">Get Help Resources</button>
                    </div>
                )}

                <div className="bg-theme-panel rounded-2xl shadow p-4 mb-4 animated-item">
                    <h2 className="text-lg font-semibold mb-2 text-theme-text-main">
                        Consent Status <span title="Data is anonymized, kept for 30 days">❓</span>
                    </h2>
                    <p className="text-theme-text-subtle">{consentStatus}</p>
                </div>

                <div className="flex flex-col flex-1 bg-theme-panel rounded-2xl shadow overflow-hidden animated-item">
                    <div id="chatBox" className="flex-1 flex flex-col gap-3 p-4 overflow-y-auto" role="log" aria-live="polite">
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`chat-bubble ${msg.sender === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai'}`}>
                                    {msg.text}
                                    {msg.mood && <div className="mt-2 text-xs opacity-75">Detected mood: {msg.mood}</div>}
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div className="flex justify-start">
                                <div className="chat-bubble-ai p-4">Thinking…</div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    <div className="p-2 border-t border-theme-primary bg-theme-panel">
                        <div className="emoji-row flex gap-2 items-center">
                            {['😀', '😢', '😰', '😌', '❤'].map((emoji) => (
                                <button key={emoji} className="px-2 py-1" onClick={() => setInput((prev) => `${prev}${emoji}`)}>
                                    {emoji}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="flex gap-2 items-center p-3 border-t border-theme-primary bg-theme-panel">
                        <div className="flex items-center gap-2 w-full">
                            <input
                                id="message"
                                type="text"
                                value={input}
                                onChange={(event) => setInput(event.target.value)}
                                onKeyDown={handleKeyPress}
                                placeholder="Type your message..."
                                className="flex-1 p-3 rounded-xl border-theme-primary focus:outline-none focus:ring-2 focus:ring-theme-accent bg-theme-panel text-theme-text-main"
                            />
                            <div className="flex items-center gap-2">
                                <div id="micContainer" className="relative">
                                    <button
                                        id="micBtn"
                                        aria-label="Start voice input"
                                        title="Voice input"
                                        className="w-12 h-12 rounded-full bg-gradient-to-r from-theme-primary to-theme-secondary text-white flex items-center justify-center shadow-lg hover:scale-105 transition focus:outline-none"
                                    >
                                        🎤
                                    </button>
                                </div>

                                <button
                                    id="sendBtn"
                                    onClick={handleSend}
                                    className="flex items-center gap-2 btn-primary px-6 py-3 rounded-xl hover:opacity-90 transition"
                                    disabled={isLoading}
                                >
                                    Send
                                    {isLoading && <span className="ml-2">...</span>}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <footer className="text-center text-xs text-theme-text-subtle mt-4">
                    <p><a href="/privacy" className="underline">Privacy Policy</a></p>
                </footer>
            </div>
        </>
    );
}
