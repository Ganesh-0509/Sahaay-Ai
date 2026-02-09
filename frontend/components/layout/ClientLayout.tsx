'use client';

import { usePathname } from 'next/navigation';
import Layout from './Layout';

export default function ClientLayout({ children }: { children: React.ReactNode }) {
    const pathname = usePathname();
    // excluded paths (no sidebar)
    const isAuthPage = pathname === '/login' || pathname === '/signup' || pathname === '/';

    if (isAuthPage) {
        return <>{children}</>;
    }

    return <Layout>{children}</Layout>;
}
