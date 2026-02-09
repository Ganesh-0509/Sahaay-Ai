'use client';

import { useEffect, useRef, useState } from 'react';

interface DemoMessage {
  sender: 'user' | 'ai';
  text: string;
}

export default function HomePage() {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<DemoMessage[]>([]);

  useEffect(() => {
    const animated = document.querySelectorAll('[data-animate]');
    animated.forEach((item) => item.classList.add('visible'));
  }, []);

  useEffect(() => {
    const renderChart = async () => {
      if (!chartRef.current) return;
      const Chart = (await import('chart.js/auto')).default;
      new Chart(chartRef.current, {
        type: 'line',
        data: {
          labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
          datasets: [
            {
              label: 'Mood Score',
              data: [4, 6, 3, 7, 5, 8, 6],
              borderColor: '#22d3ee',
              backgroundColor: 'rgba(34, 211, 238, 0.2)',
              tension: 0.3,
              pointRadius: 4,
            },
          ],
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } },
      });
    };
    renderChart();
  }, []);

  const sendMessage = () => {
    if (!chatInput.trim()) return;
    const userMsg: DemoMessage = { sender: 'user', text: chatInput };
    setChatMessages((prev) => [...prev, userMsg, { sender: 'ai', text: 'Thanks for sharing. How are you feeling today?' }]);
    setChatInput('');
  };

  return (
    <div className="bg-gray-900 text-white min-h-screen">
      <nav id="navbar" className="fixed top-0 left-0 w-full navbar-bg shadow-lg z-50">
        <div className="flex items-center justify-between px-6 py-4 w-full">
          <div className="flex-shrink-0">
            <div className="flex items-center">
              <div className="h-12 w-12 rounded-full overflow-hidden mr-3 flex items-center justify-center bg-transparent">
                <img src="/static/logo.png" alt="Sahaay-AI Logo" className="h-full w-full object-cover" loading="lazy" onError={(event) => {
                  (event.currentTarget as HTMLImageElement).style.display = 'none';
                }} />
              </div>
              <span className="text-2xl font-bold text-white">Sahaay-AI</span>
            </div>
          </div>

          <div className="flex-grow hidden md:flex justify-center space-x-6">
            <a href="#features" className="nav-link text-gray-200 hover:text-cyan-200 transition">Features</a>
            <a href="#mood-tracking" className="nav-link text-gray-200 hover:text-cyan-200 transition">Mood Tracking</a>
            <a href="#chatbot" className="nav-link text-gray-200 hover:text-cyan-200 transition">Chatbot</a>
            <a href="#faq" className="nav-link text-gray-200 hover:text-cyan-200 transition">FAQ</a>
            <a href="#contact" className="nav-link text-gray-200 hover:text-cyan-200 transition">Contact</a>
          </div>

          <div className="flex-shrink-0 flex space-x-4">
            <a href="/login" className="px-4 py-2 text-white border border-cyan-200 rounded-lg hover:bg-cyan-200 hover:text-gray-900 transition">Login</a>
            <a href="/signup" className="px-6 py-2 bg-gradient-to-r from-cyan-500 to-teal-500 text-white rounded-lg shadow hover:scale-105 transition">Get Started</a>
          </div>
        </div>
      </nav>

      <header className="hero min-h-screen flex items-center justify-center text-center px-4 relative overflow-hidden" data-animate>
        <div className="animated-bg"></div>
        <div className="max-w-3xl relative z-10 mt-20">
          <div className="flex justify-center mb-8">
            <div className="h-32 w-32 rounded-full overflow-hidden flex items-center justify-center bg-transparent">
              <img src="/static/logo.png" alt="Sahaay-AI Logo" className="h-full w-full object-cover drop-shadow-2xl animate-pulse" loading="lazy" onError={(event) => {
                (event.currentTarget as HTMLImageElement).style.display = 'none';
              }} />
            </div>
          </div>
          <h1 className="text-5xl md:text-6xl font-extrabold mb-6">Sahaay-AI: Your Companion for Mental Wellness</h1>
          <p className="text-lg md:text-xl mb-8 text-gray-200">A secure, private space for daily check-ins, mood tracking, and personalized guidance.</p>
          <div className="flex justify-center space-x-4 flex-wrap">
            <a href="/signup" className="px-8 py-4 text-lg font-semibold text-teal-900 bg-white rounded-xl shadow-lg transition hover:scale-105">Get Started</a>
            <a href="#features" className="px-8 py-4 text-lg font-semibold text-white border-2 border-white rounded-xl transition hover:bg-white hover:text-teal-900">Join Sahaay-AI</a>
          </div>
        </div>
      </header>

      <section id="features" className="py-20 px-4 bg-gray-950" data-animate>
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-12 text-white">What Makes Sahaay-AI Unique</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 text-center">
            <div className="p-8 rounded-2xl bg-gray-800/70 backdrop-blur-md border border-gray-700 shadow-xl card-3d float hover:shadow-cyan-500/40">
              <div className="text-6xl mb-4">🔒</div>
              <h3 className="text-2xl font-semibold mb-2 text-white">Confidential & Private</h3>
              <p className="text-gray-300">Your data is anonymized and auto-deletes, keeping you secure.</p>
            </div>
            <div className="p-8 rounded-2xl bg-gray-800/70 backdrop-blur-md border border-gray-700 shadow-xl card-3d float hover:shadow-cyan-500/40">
              <div className="text-6xl mb-4">🌍</div>
              <h3 className="text-2xl font-semibold mb-2 text-white">Multilingual Support</h3>
              <p className="text-gray-300">Express yourself in your preferred language without barriers.</p>
            </div>
            <div className="p-8 rounded-2xl bg-gray-800/70 backdrop-blur-md border border-gray-700 shadow-xl card-3d float hover:shadow-cyan-500/40">
              <div className="text-6xl mb-4">📊</div>
              <h3 className="text-2xl font-semibold mb-2 text-white">Mood Insights</h3>
              <p className="text-gray-300">Track emotional trends with clear, insightful visualizations.</p>
            </div>
            <div className="p-8 rounded-2xl bg-gray-800/70 backdrop-blur-md border border-gray-700 shadow-xl card-3d float hover:shadow-cyan-500/40">
              <div className="text-6xl mb-4">🤝</div>
              <h3 className="text-2xl font-semibold mb-2 text-white">Compassionate Support</h3>
              <p className="text-gray-300">AI-driven empathetic guidance designed for youth wellness.</p>
            </div>
          </div>
        </div>
      </section>

      <section id="mood-tracking" className="py-20 px-4 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-600 text-white" data-animate>
        <div className="max-w-5xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-8">Track Your Mood</h2>
          <p className="text-lg mb-12">Get a clear picture of your emotional journey through interactive charts and journaling.</p>
          <div className="mx-auto max-w-2xl h-72">
            <canvas ref={chartRef} className="w-full h-full"></canvas>
          </div>
        </div>
      </section>

      <section id="chatbot" className="py-20 px-4 bg-gray-900" data-animate>
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-8 text-white">Talk to Sahaay-AI</h2>
          <div className="bg-gray-800/70 backdrop-blur-md rounded-2xl shadow-xl p-8 border border-gray-700">
            <div id="chatWindow" className="h-64 overflow-y-auto border border-gray-700 rounded-lg p-4 text-left mb-4 text-gray-100">
              {chatMessages.length === 0 ? (
                <p className="text-gray-300">Start a short demo chat here.</p>
              ) : (
                chatMessages.map((msg, index) => (
                  <div key={index} className={msg.sender === 'user' ? 'text-right' : 'text-left'}>
                    <span className="inline-block px-3 py-2 rounded-lg mb-2 bg-gray-900/70">{msg.text}</span>
                  </div>
                ))
              )}
            </div>
            <div className="flex">
              <input
                type="text"
                value={chatInput}
                onChange={(event) => setChatInput(event.target.value)}
                placeholder="Type your message..."
                className="flex-grow px-4 py-2 border border-gray-700 bg-gray-900 text-white rounded-l-lg focus:outline-none focus:ring-2 focus:ring-cyan-400"
              />
              <button
                onClick={sendMessage}
                className="px-6 py-2 bg-gradient-to-r from-cyan-500 to-teal-500 text-white rounded-r-lg hover:from-cyan-400 hover:to-teal-400 transition"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      </section>

      <section id="contact" className="py-20 px-6 bg-gradient-to-r from-purple-700 via-cyan-600 to-teal-500 text-white" data-animate>
        <div className="max-w-5xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-12">Contact Us</h2>
          <div className="grid md:grid-cols-2 gap-12">
            <div>
              <h3 className="text-2xl font-semibold mb-4">Get in Touch</h3>
              <p className="text-gray-200 mb-6">We’d love to hear from you! Questions, feedback, or support, reach out anytime.</p>
              <p className="mb-2"><strong>Email:</strong> <a href="mailto:support@sahaay-ai.com" className="text-cyan-300 hover:underline">support@sahaay-ai.com</a></p>
              <p className="mb-2"><strong>Phone:</strong> +91 98765 43210</p>
              <p><strong>Address:</strong> Chennai, India</p>
            </div>
            <div className="bg-gray-800/70 backdrop-blur-md p-6 rounded-2xl shadow-xl border border-gray-700">
              <form>
                <input type="text" placeholder="Your Name" className="w-full mb-4 px-4 py-3 rounded-lg bg-gray-900 text-white border border-gray-700 focus:ring-2 focus:ring-cyan-400" />
                <input type="email" placeholder="Your Email" className="w-full mb-4 px-4 py-3 rounded-lg bg-gray-900 text-white border border-gray-700 focus:ring-2 focus:ring-cyan-400" />
                <textarea placeholder="Your Message" rows={4} className="w-full mb-4 px-4 py-3 rounded-lg bg-gray-900 text-white border border-gray-700 focus:ring-2 focus:ring-cyan-400"></textarea>
                <button type="submit" className="w-full py-3 bg-gradient-to-r from-cyan-500 to-teal-500 text-white rounded-lg shadow hover:scale-105 transition">Send Message</button>
              </form>
            </div>
          </div>
        </div>
      </section>

      <section id="faq" className="py-20 px-6 bg-gray-950 text-white" data-animate>
        <div className="max-w-4xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-12">Frequently Asked Questions</h2>
          <div className="space-y-4">
            <details className="faq-item bg-gray-800/70 backdrop-blur-md p-6 rounded-xl border border-gray-700">
              <summary>What is Sahaay-AI?</summary>
              <p>Sahaay-AI is an AI-powered mental wellness companion designed to provide a safe, private, and judgment-free space for you to express your thoughts and feelings.</p>
            </details>
            <details className="faq-item bg-gray-800/70 backdrop-blur-md p-6 rounded-xl border border-gray-700">
              <summary>How does Sahaay-AI work?</summary>
              <p>Private chat with AI, mood analysis, and dashboard visualizations help track emotional patterns.</p>
            </details>
            <details className="faq-item bg-gray-800/70 backdrop-blur-md p-6 rounded-xl border border-gray-700">
              <summary>Is my data safe and private?</summary>
              <p>Yes, secure storage with anonymization options and clear data policy.</p>
            </details>
            <details className="faq-item bg-gray-800/70 backdrop-blur-md p-6 rounded-xl border border-gray-700">
              <summary>Can Sahaay-AI replace a therapist?</summary>
              <p>No, it provides supportive guidance but not professional therapy. Crisis contacts included.</p>
            </details>
            <details className="faq-item bg-gray-800/70 backdrop-blur-md p-6 rounded-xl border border-gray-700">
              <summary>What kind of coping tools does it offer?</summary>
              <p>Breathing, mindfulness, self-care, and grounding exercises personalized to your mood.</p>
            </details>
            <details className="faq-item bg-gray-800/70 backdrop-blur-md p-6 rounded-xl border border-gray-700">
              <summary>What technology powers Sahaay-AI?</summary>
              <p>Backend: Python & Flask, AI: Gemini API, DB: Firestore, Frontend: Tailwind CSS.</p>
            </details>
          </div>
        </div>
      </section>

      <footer className="bg-gray-900 text-gray-400 py-6 text-center">
        © 2026 Sahaay-AI. All rights reserved.
      </footer>
    </div>
  );
}
