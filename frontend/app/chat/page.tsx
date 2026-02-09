'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import { chatAPI } from '@/lib/api';
import Layout from '@/components/layout/Layout';

interface Message {
    text: string;
    sender: 'user' | 'ai';
    mood?: string;
    timestamp?: string;
}

export default function ChatPage() {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const user = useAuthStore((state) => state.user);
    const authLoading = useAuthStore((state) => state.isLoading);
    const checkAuth = useAuthStore((state) => state.checkAuth);
    const router = useRouter();

    useEffect(() => {
        checkAuth();
    }, [checkAuth]);

    useEffect(() => {
        // Don't redirect while still loading auth state
        if (authLoading) return;

        if (!user) {
            router.push('/login');
            return;
        }

        // Load conversation history
        loadConversation();
    }, [user, authLoading, router]);

    useEffect(() => {
        // Scroll to bottom on new messages
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const loadConversation = async () => {
        try {
            const response = await chatAPI.getConversation();
            console.log('Conversation response:', response.data);

            if (response.data.ok && response.data.messages) {
                const formattedMessages: Message[] = response.data.messages.map((msg: any) => ({
                    text: msg.text,
                    sender: msg.sender,
                    mood: msg.mood,
                    timestamp: msg.timestamp
                }));

                console.log(`Loaded ${formattedMessages.length} messages`);
                setMessages(formattedMessages);
            } else {
                console.log('No messages in conversation history');
            }
        } catch (error) {
            console.error('Failed to load conversation:', error);
        }
    };

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage: Message = { text: input, sender: 'user' };
        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await chatAPI.sendMessage(input);

            if (response.data.response) {
                const aiMessage: Message = {
                    text: response.data.response,
                    sender: 'ai',
                    mood: response.data.mood
                };
                setMessages((prev) => [...prev, aiMessage]);
            }
        } catch (error) {
            console.error('Failed to send message:', error);
            const errorMessage: Message = {
                text: 'Sorry, I encountered an error. Please try again.',
                sender: 'ai'
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
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
        <Layout>
            <div className="flex flex-col h-[calc(100vh-4rem)] max-w-4xl mx-auto">
                {/* Header */}
                <div className="mb-6 animate-fade-in-up">
                    <h1 className="text-4xl font-bold text-theme-text-main mb-2">
                        AI Support Chat 💬
                    </h1>
                    <p className="text-theme-text-subtle">
                        Talk to our AI companion about how you're feeling
                    </p>
                </div>

                {/* Messages Container */}
                <div className="flex-1 glass-card overflow-y-auto mb-4 p-6 space-y-4">
                    {messages.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full text-center">
                            <div className="text-6xl mb-4 animate-float">🤗</div>
                            <h3 className="text-2xl font-semibold text-theme-text-main mb-2">
                                Start a conversation
                            </h3>
                            <p className="text-theme-text-subtle max-w-md">
                                I'm here to listen and support you. Share how you're feeling, and I'll do my best to help.
                            </p>
                        </div>
                    ) : (
                        messages.map((msg, idx) => (
                            <div
                                key={idx}
                                className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div className={`chat-bubble ${msg.sender === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai'}`}>
                                    {msg.text}
                                    {msg.mood && (
                                        <div className="mt-2 text-xs opacity-75">
                                            Detected mood: {msg.mood}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))
                    )}
                    {isLoading && (
                        <div className="flex justify-start">
                            <div className="chat-bubble-ai p-4">
                                <div className="flex gap-2">
                                    <div className="w-2 h-2 bg-theme-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                    <div className="w-2 h-2 bg-theme-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                    <div className="w-2 h-2 bg-theme-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Container */}
                <div className="glass-card p-4">
                    <div className="flex gap-3">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="Type your message here..."
                            className="input flex-1"
                            disabled={isLoading}
                        />
                        <button
                            onClick={handleSend}
                            disabled={!input.trim() || isLoading}
                            className="btn btn-primary px-8"
                        >
                            {isLoading ? '...' : 'Send'}
                        </button>
                    </div>
                </div>
            </div>
        </Layout>
    );
}
