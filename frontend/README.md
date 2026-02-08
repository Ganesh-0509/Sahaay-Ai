# Sahaay AI Frontend - Next.js Setup

## 📦 Installation

```bash
cd d:\sahaay-ai\frontend
npm install axios zustand date-fns
```

## 🚀 Running the Application

### Development Mode

**Terminal 1 - Backend (Flask):**
```bash
cd d:\sahaay-ai
python app.py
```

**Terminal 2 - Frontend (Next.js):**
```bash
cd d:\sahaay-ai\frontend
npm run dev
```

Then open: http://localhost:3000

### Production Build

```bash
cd d:\sahaay-ai\frontend
npm run build
npm start
```

## 🔧 Configuration

### Environment Variables
File: `.env.local`
```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

For production, update to your backend URL:
```
NEXT_PUBLIC_API_URL=https://your-api-domain.com
```

## 🏗️ Project Structure

```
frontend/
├── app/                      # Next.js App Router
│   ├── layout.tsx           # Root layout
│   ├── page.tsx            # Home (redirects)
│   ├── login/page.tsx      # Login page
│   ├── signup/page.tsx     # Signup page
│   ├── dashboard/page.tsx  # Dashboard
│   ├── chat/page.tsx       # Chat interface
│   └── globals.css         # Global styles
├── components/
│   ├── chat/               # Chat components
│   │   ├── ChatBubble.tsx
│   │   └── ChatInput.tsx
│   └── ui/                 # Reusable UI
│       ├── Button.tsx
│       └── Input.tsx
├── lib/
│   └── api.ts             # API client
├── store/
│   └── authStore.ts       # Auth state
└── .env.local             # Environment config
```

## 🎨 Styling

Using Tailwind CSS with a mental-health calm theme:
- Muted colors (teal, sage green)
- No gradients or heavy shadows
- Clean, minimal design
- Inter font

## 🔐 Authentication

Uses session-based auth with HTTP-only cookies from Flask backend.

### Important
Make sure your Flask backend has CORS configured:
```python
CORS(app, supports_credentials=True, origins=[
    "http://localhost:3000",
    "https://your-frontend-domain.com"
])
```

## ✅ Features Implemented

- [x] Login/Signup pages
- [x] Dashboard with mood stats
- [x] Chat interface with conversation history
- [x] Emotion detection display
- [x] Protected routes
- [x] Session management
- [x] Clean, professional UI

## 🚧 To Complete

- [ ] Settings page
- [ ] Mood tracker analytics page
- [ ] Mobile responsiveness testing
- [ ] Loading states improvement
- [ ] Error boundary implementation
- [ ] Unit tests

## 📝 Notes

- All AI logic remains in Flask backend
- No backend modifications required (except CORS)
- UI follows mental-health design guidelines
- Production-ready structure
