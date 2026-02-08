# Sahaay-AI Frontend Migration: Flask → Next.js

## 🎯 Migration Overview

This migration transforms Sahaay-AI's Jinja2/Flask frontend into a **production-grade Next.js 14 application** while preserving all backend logic, AI agents, and API behavior. The new frontend will be clean, calm, and mental-health appropriate.

## 📋 Pre-Migration Checklist

✅ Backend remains unchanged (Flask API server)  
✅ All AI logic, prompts, and fallbacks preserved  
✅ Authentication adapted (session-based → API-based)  
✅ UI: Minimal, calm, professional (no flashy elements)  
✅ Production-ready structure

---

## 🛠️ Step 1: Create Next.js Project

```bash
# Navigate to parent directory
cd d:\

# Create Next.js app with TypeScript and Tailwind
npx create-next-app@latest sahaay-ai-frontend --typescript --tailwind --app --no-src-dir --import-alias "@/*"

# Navigate into project
cd sahaay-ai-frontend

# Install additional dependencies
npm install axios zustand date-fns

# Install dev dependencies
npm install -D @types/node @types/react @types/react-dom
```

---

## 📁 Final Project Structure

```
sahaay-ai-frontend/
├── app/
│   ├── layout.tsx                 # Root layout
│   ├── page.tsx                   # Landing page (redirect to /dashboard or /login)
│   ├── login/
│   │   └── page.tsx              # Login page
│   ├── signup/
│   │   └── page.tsx              # Signup page
│   ├── dashboard/
│   │   └── page.tsx              # Dashboard (home)
│   ├── chat/
│   │   └── page.tsx              # Chat interface
│   ├── mood/
│   │   └── page.tsx              # Mood tracker/analytics
│   ├── settings/
│   │   └── page.tsx              # User settings
│   └── api/                       # Next.js API routes (optional, for SSR auth)
│       └── auth/
│           └── session/route.ts  # Session validation endpoint
├── components/
│   ├── chat/
│   │   ├── ChatBubble.tsx        # Individual message bubble
│   │   ├── ChatInput.tsx         # Message input field
│   │   └── ChatHistory.tsx       # Conversation history
│   ├── dashboard/
│   │   ├── StatsCard.tsx         # Mood stats display
│   │   ├── RecentMoods.tsx       # Recent check-ins
│   │   └── QuickActions.tsx      # Action buttons
│   ├── layout/
│   │   ├── Header.tsx            # App header
│   │   ├── Sidebar.tsx           # Navigation sidebar
│   │   └── Footer.tsx            # Footer (optional)
│   └── ui/
│       ├── Button.tsx            # Reusable button
│       ├── Input.tsx             # Reusable input
│       └── Card.tsx              # Reusable card
├── lib/
│   ├── api.ts                    # Centralized API client
│   ├── auth.ts                   # Auth utilities
│   └── hooks/
│       ├── useAuth.ts           # Authentication hook
│       └── useChat.ts           # Chat state management
├── store/
│   └── authStore.ts             # Zustand auth state
├── styles/
│   └── globals.css              # Global styles (Tailwind)
├── public/
│   └── logo.svg
├── .env.local                    # Environment variables
├── next.config.js               # Next.js config
├── tailwind.config.ts           # Tailwind config
└── tsconfig.json                # TypeScript config
```

---

## 🎨 Styling Strategy

### Color Palette (Mental Health Calm Theme)
```css
/* tailwind.config.ts */
colors: {
  primary: '#4A90A4',      // Muted teal/blue
  secondary: '#7C9885',    // Soft sage green
  background: '#FAFAF9',   // Off-white
  surface: '#FFFFFF',      // Pure white
  border: '#E5E5E5',       // Soft gray
  text: {
    primary: '#1F2937',    // Dark gray
    secondary: '#6B7280',  // Medium gray
    muted: '#9CA3AF',      // Light gray
  },
  success: '#84CC16',      // Muted green
  warning: '#F59E0B',      // Muted amber
  danger: '#DC2626',       // Muted red
}
```

### Design Principles
- **No gradients, shadows, or animations**
- **Subtle borders only** (`border-gray-200`)
- **Rounded corners**: `rounded-md` (8px max)
- **Font**: Inter (via Google Fonts or system fallback)
- **Spacing**: Generous whitespace for calm feel
- **Elevations**: Minimal (`shadow-sm` only for cards)

---

## 🔐 Authentication Adaptation

### Current: Flask-Login (session-based)
Your Flask backend uses session cookies managed by `Flask-Login`.

### Migration Strategy: Cookie-Based Auth (Recommended)
1. **Backend**: Flask continues to set HTTP-only session cookies on `/login`
2. **Frontend**: Next.js sends credentials, receives cookies automatically
3. **API calls**: Include `credentials: 'include'` in fetch/axios
4. **Protected routes**: Check auth status via `/api/get_user_info`

**No backend changes required** if:
- CORS is configured to allow credentials from Next.js origin
- Session cookies are set with `SameSite=None; Secure` (for different domains/ports)

### Backend CORS Update (ONLY IF NEEDED)
```python
# app.py
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])
```

---

## 📡 API Communication Layer

All backend calls go through a centralized `lib/api.ts` module.

### Environment Variables
```env
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### Centralized API Client
```typescript
// lib/api.ts
import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  withCredentials: true, // Include cookies
  headers: {
    'Content-Type': 'application/json',
  },
});

export default api;
```

---

## 🚀 Development Workflow

### Terminal Commands

**Start Flask backend** (in existing project):
```bash
cd d:\sahaay-ai
python app.py
```

**Start Next.js frontend** (in new project):
```bash
cd d:\sahaay-ai-frontend
npm run dev
```

**Build for production**:
```bash
npm run build
npm start
```

---

## 📝 Key Migration Steps

### 1. Setup (Day 1)
- Create Next.js project with commands above
- Configure Tailwind with mental-health color palette
- Set up environment variables

### 2. Core Components (Day 2-3)
- Build reusable UI components (`Button`, `Input`, `Card`)
- Create layout components (`Header`, `Sidebar`)
- Implement auth store with Zustand

### 3. Pages (Day 4-6)
- **Login/Signup**: Form validation, API integration
- **Dashboard**: Fetch and display home data, streak, recent moods
- **Chat**: Real-time message UI, conversation history
- **Settings**: Language, notifications, profile

### 4. API Integration (Day 7)
- Connect all pages to Flask backend APIs
- Test authentication flow end-to-end
- Handle loading states and errors

### 5. Polish & Testing (Day 8-9)
- Accessibility audit (ARIA labels, keyboard nav)
- Responsive design for mobile
- Cross-browser testing

### 6. Deployment (Day 10)
- Build optimized production bundle
- Deploy on Vercel/Netlify (frontend) + existing Flask host (backend)

---

## ⚠️ Backend Changes Required (MINIMAL)

### CORS Configuration
Ensure Flask allows credentials from Next.js origin:

```python
# app.py (line 88)
CORS(app, supports_credentials=True, origins=[
    "http://localhost:3000",  # Development
    "https://your-frontend-domain.com"  # Production
])
```

### Session Cookie Settings
Already configured correctly:
```python
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
```

**Add** if serving from different domain:
```python
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
```

### API Response Format
All routes already return JSON ✅  
No changes needed.

---

## 🧪 Testing Strategy

1. **Unit Tests**: React components with Jest + React Testing Library
2. **Integration Tests**: API calls with MSW (Mock Service Worker)
3. **E2E Tests**: Playwright for critical flows (login → chat → logout)

---

## 📦 Deployment Architecture

```
┌─────────────────────────────────────┐
│   Frontend (Vercel/Netlify)         │
│   Next.js 14 + React 18              │
│   https://sahaay-ai.vercel.app      │
└────────────┬────────────────────────┘
             │ API Calls (HTTPS + Cookies)
             ▼
┌─────────────────────────────────────┐
│   Backend (Render/Railway/Heroku)   │
│   Flask + Gemini AI + Firestore     │
│   https://api.sahaay-ai.com         │
└─────────────────────────────────────┘
```

---

## ✅ Success Criteria

- [ ] All pages render without Jinja templates
- [ ] Authentication works seamlessly with Flask session cookies
- [ ] Chat interface maintains conversation history
- [ ] Emotion detection and crisis alerts function identically
- [ ] UI is calm, clean, and professional
- [ ] No console errors or warnings
- [ ] Mobile responsive (320px - 1920px)
- [ ] Lighthouse score: 90+ (Performance, Accessibility, Best Practices)

---

## 🎯 Next Steps

1. Review this migration plan
2. Run the setup commands in the next section
3. I'll create all core components and pages for you
4. Test locally with Flask backend running
5. Deploy to production
