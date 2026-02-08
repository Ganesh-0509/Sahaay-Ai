import React from 'react';

interface Message {
    text: string;
    sender: 'user' | 'ai';
    mood?: string;
    timestamp?: string;
}

interface ChatBubbleProps {
    message: Message;
}

export const ChatBubble: React.FC<ChatBubbleProps> = ({ message }) => {
    const isUser = message.sender === 'user';

    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
            <div className={`chat-bubble ${isUser ? 'chat-bubble-user' : 'chat-bubble-ai'}`}>
                <p className="text-text-primary whitespace-pre-wrap">{message.text}</p>
                {message.mood && (
                    <span className="text-xs text-text-muted mt-2 block">
                        Mood: {message.mood}
                    </span>
                )}
            </div>
        </div>
    );
};
