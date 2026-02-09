'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import Link from 'next/link';

export default function LoginPage() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const router = useRouter();
    const login = useAuthStore((state) => state.login);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            await login(email, password);
            router.push('/dashboard');
        } catch (err: any) {
            setError(err.response?.data?.message || 'Invalid email or password');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="auth-gradient px-4">
            <div className="fade-in bg-gray-900/80 backdrop-blur-md rounded-2xl shadow-2xl w-full max-w-md p-8 space-y-6 text-white">
                <div className="text-center">
                    <div className="flex justify-center mb-4">
                        <img
                            src="/static/logo.png"
                            alt="Sahaay-AI Logo"
                            className="h-16 w-16 drop-shadow-lg"
                            loading="lazy"
                            onError={(event) => {
                                (event.currentTarget as HTMLImageElement).style.display = 'none';
                            }}
                        />
                    </div>
                    <h1 className="text-3xl font-bold text-cyan-400">Welcome Back</h1>
                    <p className="mt-2 text-gray-300">Log in to continue to Sahaay-AI</p>
                </div>

                {error && (
                    <div className="p-4 rounded-lg bg-red-500/20 text-red-400">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label htmlFor="email" className="block text-sm font-medium text-gray-300">Email Address</label>
                        <input
                            type="email"
                            id="email"
                            name="email"
                            required
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="mt-1 block w-full px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white focus:ring-cyan-400 focus:border-cyan-400"
                        />
                    </div>

                    <div>
                        <label htmlFor="password" className="block text-sm font-medium text-gray-300">Password</label>
                        <input
                            type="password"
                            id="password"
                            name="password"
                            required
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="mt-1 block w-full px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white focus:ring-cyan-400 focus:border-cyan-400"
                        />
                    </div>

                    <div>
                        <button
                            type="submit"
                            className="w-full py-2 px-4 bg-gradient-to-r from-cyan-500 to-teal-500 text-white font-semibold rounded-lg shadow-lg hover:scale-105 transition"
                            disabled={isLoading}
                        >
                            {isLoading ? 'Logging in...' : 'Log In'}
                        </button>
                    </div>
                </form>

                <p className="text-center text-sm text-gray-400">
                    Don’t have an account?{' '}
                    <Link href="/signup" className="text-cyan-400 hover:text-teal-400 font-medium">
                        Sign Up
                    </Link>
                </p>
            </div>
        </div>
    );
}
