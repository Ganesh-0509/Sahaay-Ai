'use client';

import GradientBackground from './GradientBackground';
import Sidebar from './Sidebar';

export default function Layout({ children }: { children: React.ReactNode }) {
    return (
        <div className="flex min-h-screen">
            <GradientBackground />
            <Sidebar />

            {/* Main Content Area */}
            {/* Logic:
                - Mobile: ml-0 (Sidebar is hidden or overlay)
                - Desktop (sm+): ml-64 (Sidebar is fixed, push content right)
                - Transition matches sidebar for smoothness (though sidebar is fixed, content margin needs to jump or transition)
            */}
            <main className="flex-1 p-4 sm:p-6 md:p-8 overflow-y-auto ml-0 sm:ml-64 transition-all duration-300 relative z-10 w-full">
                <div className="max-w-7xl mx-auto">
                    {children}
                </div>
            </main>
        </div>
    );
}
