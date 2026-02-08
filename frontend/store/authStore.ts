import { create } from 'zustand';
import { authAPI } from '@/lib/api';

interface User {
    id: string;
    user_id?: string;
    username: string;
    name?: string;
    email?: string;
    is_anonymous?: boolean;
}

interface AuthState {
    user: User | null;
    isLoading: boolean;
    error: string | null;

    login: (email: string, password: string) => Promise<void>;
    signup: (email: string, name: string, password: string, consent: boolean) => Promise<void>;
    logout: () => Promise<void>;
    checkAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
    user: null,
    isLoading: true,
    error: null,

    login: async (email, password) => {
        try {
            set({ isLoading: true, error: null });
            await authAPI.login(email, password);
            const { data } = await authAPI.getUser();
            // Map backend response to User type
            const user: User = {
                id: data.user_id || data.id,
                username: data.username,
                name: data.username,
                email: email,
                is_anonymous: data.is_anonymous || false,
            };
            set({ user, isLoading: false });
        } catch (error: any) {
            set({ error: error.response?.data?.message || 'Login failed', isLoading: false });
            throw error;
        }
    },

    signup: async (email, name, password, consent) => {
        try {
            set({ isLoading: true, error: null });
            await authAPI.signup(email, name, password, consent);
            const { data } = await authAPI.getUser();
            // Map backend response to User type
            const user: User = {
                id: data.user_id || data.id,
                username: data.username,
                name: data.username,
                email: email,
                is_anonymous: data.is_anonymous || false,
            };
            set({ user, isLoading: false });
        } catch (error: any) {
            set({ error: error.response?.data?.message || 'Signup failed', isLoading: false });
            throw error;
        }
    },

    logout: async () => {
        try {
            await authAPI.logout();
            set({ user: null });
        } catch (error) {
            // Silent fail
        }
    },

    checkAuth: async () => {
        try {
            console.log('🔍 [authStore] Checking authentication...');
            set({ isLoading: true });
            const { data } = await authAPI.getUser();
            console.log('✅ [authStore] Auth check response:', data);

            // Map backend response to User type
            const user: User = {
                id: data.user_id || data.id,
                username: data.username,
                name: data.username,
                email: data.email || '',
                is_anonymous: data.is_anonymous || false,
            };
            console.log('✅ [authStore] Setting user:', user);
            set({ user, isLoading: false });
        } catch (error: any) {
            console.log('❌ [authStore] Auth check failed:', error.response?.status, error.response?.data);
            set({ user: null, isLoading: false });
        }
    },
}));
