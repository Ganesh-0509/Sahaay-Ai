import React, { useState } from 'react';
import { Button } from '../ui/Button';

interface ChatInputProps {
    onSendMessage: (message: string) => void;
    isLoading: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, isLoading }) => {
    const [message, setMessage] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (message.trim() && !isLoading) {
            onSendMessage(message);
            setMessage('');
        }
    };

    return (
        <form onSubmit={handleSubmit} className="flex gap-2">
            <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Type your message..."
                className="input flex-1"
                disabled={isLoading}
                maxLength={2000}
            />
            <Button type="submit" variant="primary" disabled={isLoading || !message.trim()}>
                {isLoading ? 'Sending...' : 'Send'}
            </Button>
        </form>
    );
};
