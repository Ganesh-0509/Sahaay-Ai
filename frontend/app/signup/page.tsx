'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import Link from 'next/link';

export default function SignupPage() {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        password: '',
        confirmPassword: '',
        consent: false,
    });
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const router = useRouter();
    const signup = useAuthStore((state) => state.signup);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');

        if (formData.password !== formData.confirmPassword) {
            setError('Passwords do not match');
            return;
        }

        if (!formData.consent) {
            setError('You must agree to the privacy policy');
            return;
        }

        setIsLoading(true);

        try {
            await signup(formData.email, formData.name, formData.password, formData.consent);
            router.push('/dashboard');
        } catch (err: any) {
            setError(err.response?.data?.message || 'Signup failed');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="auth-gradient px-4">
            <main className="fade-in bg-gray-900/80 backdrop-blur-md rounded-2xl shadow-2xl w-full max-w-md p-8 space-y-6 text-white">
                <header className="text-center">
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
                    <h1 className="text-3xl font-bold text-cyan-400">Create Your Account</h1>
                    <p className="mt-2 text-gray-300">Join Sahaay AI to start your mental wellness journey.</p>
                </header>

                {error && (
                    <div className="p-4 rounded-lg bg-red-500/20 text-red-400">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label htmlFor="name" className="block text-sm font-medium text-gray-300">Full Name</label>
                        <input
                            type="text"
                            id="name"
                            name="name"
                            required
                            value={formData.name}
                            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                            className="mt-1 block w-full px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white focus:ring-cyan-400 focus:border-cyan-400"
                        />
                    </div>

                    <div>
                        <label htmlFor="email" className="block text-sm font-medium text-gray-300">Email Address</label>
                        <input
                            type="email"
                            id="email"
                            name="email"
                            required
                            value={formData.email}
                            onChange={(e) => setFormData({ ...formData, email: e.target.value })}
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
                            value={formData.password}
                            onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                            className="mt-1 block w-full px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white focus:ring-cyan-400 focus:border-cyan-400"
                        />
                    </div>

                    <div>
                        <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-300">Confirm Password</label>
                        <input
                            type="password"
                            id="confirmPassword"
                            name="confirmPassword"
                            required
                            value={formData.confirmPassword}
                            onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                            className="mt-1 block w-full px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white focus:ring-cyan-400 focus:border-cyan-400"
                        />
                    </div>

                    <div className="flex items-start">
                        <input
                            type="checkbox"
                            id="consent"
                            checked={formData.consent}
                            onChange={(e) => setFormData({ ...formData, consent: e.target.checked })}
                            className="mt-1 h-4 w-4 rounded border-gray-700 bg-gray-800 text-cyan-500 focus:ring-cyan-400"
                            required
                        />
                        <label htmlFor="consent" className="ml-2 text-sm text-gray-300">
                            I agree to the{' '}
                            <Link href="/privacy" target="_blank" className="text-cyan-400 hover:text-teal-400">
                                privacy policy
                            </Link>{' '}
                            and consent to data storage for mental wellness tracking.
                        </label>
                    </div>

                    <div>
                        <button
                            type="submit"
                            className="w-full py-2 px-4 bg-gradient-to-r from-cyan-500 to-teal-500 text-white font-semibold rounded-lg shadow-lg hover:scale-105 transition"
                            disabled={isLoading}
                        >
                            {isLoading ? 'Creating account...' : 'Sign Up'}
                        </button>
                    </div>
                </form>

                <p className="text-center text-sm text-gray-400">
                    Already have an account?{' '}
                    <Link href="/login" className="text-cyan-400 hover:text-teal-400 font-medium">
                        Log In
                    </Link>
                </p>
            </main>
        </div>
    );
}
