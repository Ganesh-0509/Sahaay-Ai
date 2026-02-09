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
            <main
                id="mainContent"
                className="flex-1 p-6 sm:p-8 overflow-y-auto ml-0 md:ml-64 transition-all duration-300 relative z-10"
            >
                {children}
            </main>
        </div>
    );
}
