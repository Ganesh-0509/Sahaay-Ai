'use client';

import { useEffect, useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import { chatAPI } from '@/lib/api';
import { ChatBubble } from '@/components/chat/ChatBubble';
import { ChatInput } from '@/components/chat/ChatInput';
import { Button } from '@/components/ui/Button';
import Link from 'next/link';

interface Message {
    text: string;
    sender: 'user' | 'ai';
    mood?: string;
    timestamp?: string;
}

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
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

            if (response.data && response.data.ok && response.data.messages) {
                const formattedMessages: Message[] = response.data.messages.map((msg: any) => ({
                    text: msg.text || msg.message || '',
                    sender: msg.sender || 'ai',
                    mood: msg.mood,
                }));
                setMessages(formattedMessages);
                console.log(`Loaded ${formattedMessages.length} messages`);
            } else {
                console.log('No messages in conversation history');
            }
        } catch (error: any) {
            console.error('Failed to load conversation:', error);
            console.error('Error response:', error.response?.data);
            // Don't show error to user - empty conversation is fine for first time
        }
    };

    const handleSendMessage = async (messageText: string) => {
        // Add user message immediately
        const userMessage: Message = {
            text: messageText,
            sender: 'user',
        };
        setMessages((prev) => [...prev, userMessage]);
        setIsLoading(true);

        try {
            const response = await chatAPI.sendMessage(messageText);
            // Flask returns: {ok: true, data: {response, mood, crisis_detected, ...}}
            const aiMessage: Message = {
                text: response.data.data?.response || 'I am here to support you.',
                sender: 'ai',
                mood: response.data.data?.mood,
            };
            setMessages((prev) => [...prev, aiMessage]);

            // Handle crisis detection
            if (response.data.data?.crisis_detected) {
                alert('Crisis detected. Please reach out to a mental health professional or emergency services.');
            }
        } catch (error: any) {
            console.error('Failed to send message:', error);
            const errorMessage: Message = {
                text: 'Sorry, I am having trouble responding right now. Please try again.',
                sender: 'ai',
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleLogout = async () => {
        await logout();
        router.push('/login');
    };

    if (!user) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-background">
                <p className="text-text-secondary">Loading...</p>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-screen bg-background">
            {/* Header */}
            <header className="bg-surface border-b border-border px-4 py-4 flex justify-between items-center">
                <div className="flex items-center gap-4">
                    <Link href="/dashboard">
                        <Button variant="outline">← Dashboard</Button>
                    </Link>
                    <h1 className="text-xl font-semibold text-text-primary">Chat with Sahaay AI</h1>
                </div>
                <Button variant="outline" onClick={handleLogout}>
                    Logout
                </Button>
            </header>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-6">
                <div className="max-w-3xl mx-auto">
                    {messages.length === 0 ? (
                        <div className="text-center text-text-secondary py-12">
                            <p className="mb-4">👋 Hi, I'm Sahaay AI</p>
                            <p>How are you feeling today?</p>
                        </div>
                    ) : (
                        messages.map((message, index) => (
                            <ChatBubble key={index} message={message} />
                        ))
                    )}
                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Chat Input */}
            <div className="bg-surface border-t border-border px-4 py-4">
                <div className="max-w-3xl mx-auto">
                    <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
                </div>
            </div>
        </div>
    );
}
