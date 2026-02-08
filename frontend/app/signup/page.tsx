'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
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
        <div className="min-h-screen flex items-center justify-center bg-background py-8">
            <div className="card w-full max-w-md">
                <div className="text-center mb-8">
                    <h1 className="text-3xl font-semibold text-text-primary mb-2">
                        Create Account
                    </h1>
                    <p className="text-text-secondary">
                        Join Sahaay AI for mental wellness support
                    </p>
                </div>

                {error && (
                    <div className="mb-4 p-3 rounded bg-danger/10 border border-danger/20 text-danger text-sm">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4">
                    <Input
                        label="Full Name"
                        type="text"
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        placeholder="John Doe"
                        required
                    />

                    <Input
                        label="Email"
                        type="email"
                        value={formData.email}
                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                        placeholder="you@example.com"
                        required
                    />

                    <Input
                        label="Password"
                        type="password"
                        value={formData.password}
                        onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                        placeholder="••••••••"
                        required
                    />

                    <Input
                        label="Confirm Password"
                        type="password"
                        value={formData.confirmPassword}
                        onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                        placeholder="••••••••"
                        required
                    />

                    <div className="flex items-start">
                        <input
                            type="checkbox"
                            id="consent"
                            checked={formData.consent}
                            onChange={(e) => setFormData({ ...formData, consent: e.target.checked })}
                            className="mt-1 mr-2"
                            required
                        />
                        <label htmlFor="consent" className="text-sm text-text-secondary">
                            I agree to the{' '}
                            <Link href="/privacy" className="text-primary hover:underline">
                                privacy policy
                            </Link>
                        </label>
                    </div>

                    <Button
                        type="submit"
                        variant="primary"
                        className="w-full"
                        disabled={isLoading}
                    >
                        {isLoading ? 'Creating account...' : 'Sign Up'}
                    </Button>
                </form>

                <p className="mt-6 text-center text-sm text-text-secondary">
                    Already have an account?{' '}
                    <Link href="/login" className="text-primary hover:underline">
                        Sign in
                    </Link>
                </p>
            </div>
        </div>
    );
}
