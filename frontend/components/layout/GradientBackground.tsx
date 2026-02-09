'use client';

import { useEffect, useRef } from 'react';

export default function GradientBackground() {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let particles: Particle[] = [];
        let animationFrameId: number;

        const setupCanvas = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };

        class Particle {
            x: number;
            y: number;
            size: number;
            speedX: number;
            speedY: number;
            color: string;

            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.size = Math.random() * 2 + 1;
                this.speedX = Math.random() * 1 - 0.5;
                this.speedY = Math.random() * 1 - 0.5;

                // Calming teal colors
                const colors = [
                    `rgba(91, 163, 163, ${Math.random() * 0.4 + 0.2})`,   // Teal
                    `rgba(124, 196, 196, ${Math.random() * 0.3 + 0.1})`,  // Light Teal
                    `rgba(255, 155, 133, ${Math.random() * 0.2 + 0.1})`,  // Coral
                ];
                this.color = colors[Math.floor(Math.random() * colors.length)];
            }

            update() {
                this.x += this.speedX;
                this.y += this.speedY;

                if (this.x > canvas.width || this.x < 0) this.speedX *= -1;
                if (this.y > canvas.height || this.y < 0) this.speedY *= -1;
            }

            draw() {
                ctx.fillStyle = this.color;
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        const initParticles = () => {
            particles = [];
            const numberOfParticles = (canvas.width * canvas.height) / 9000;
            for (let i = 0; i < numberOfParticles; i++) {
                particles.push(new Particle());
            }
        };

        const animateParticles = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            for (let i = 0; i < particles.length; i++) {
                particles[i].update();
                particles[i].draw();
            }

            animationFrameId = requestAnimationFrame(animateParticles);
        };

        setupCanvas();
        initParticles();
        animateParticles();

        const handleResize = () => {
            setupCanvas();
            initParticles();
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    return (
        <>
            {/* Animated Gradient Background */}
            <div className="fixed inset-0 z-[-2]" style={{
                background: 'linear-gradient(-45deg, #0A1929, #132F4C, #1E4976, #0A1929)',
                backgroundSize: '400% 400%',
                animation: 'animatedGradient 20s ease infinite'
            }} />

            {/* Particle Canvas */}
            <canvas
                ref={canvasRef}
                className="fixed inset-0 z-[-1]"
                style={{ pointerEvents: 'none' }}
            />
        </>
    );
}
