# Sahaay-AI

## Project Description

Sahaay-AI is a comprehensive mental wellness companion application designed to help users track, understand, and improve their emotional well-being. The platform provides a safe, private space for daily mental health check-ins through AI-powered conversational interactions.

**Impact Statement:**

In a world where mental health challenges affect millions yet remain stigmatized and underserved, Sahaay-AI bridges the gap between awareness and accessibility. By providing a judgment-free, always-available digital companion, this project empowers individuals to take proactive steps toward understanding their emotions, recognizing patterns, and seeking help when needed. It promotes early intervention and normalizes mental health conversations.

**Problem It Solves:**
- Helps individuals monitor and understand their mental health patterns
- Provides accessible mental health support through AI-driven conversations
- Offers personalized coping strategies based on emotional trends
- Identifies potential crisis situations and provides immediate support resources

**Target Users:**
- Individuals seeking to improve their mental wellness
- People looking for a private space to express their emotions
- Users who want data-driven insights into their emotional patterns
- Anyone needing accessible mental health support

---

## Key Features

- **AI-Powered Conversations**: Engage in empathetic conversations with an AI companion powered by Google Gemini API
- **Emotion Recognition**: Real-time emotion classification and sentiment analysis during chat sessions
- **Mood Analytics Dashboard**: Visualize emotional trends with interactive charts and graphs to identify patterns
- **Crisis Support Detection**: Automated identification of potential crisis indicators with immediate access to support resources
- **Personalized Coping Tools**: Tailored recommendations and coping strategies based on mood history
- **Community Support**: Anonymous community posts, polls, and peer interaction in a safe space
- **Multi-language Support**: Available in English, Hindi, Tamil, and Telugu for broader accessibility
- **Streak Tracking**: Gamification elements to encourage consistent daily check-ins
- **Admin Dashboard**: Comprehensive user management and analytics for platform administrators
- **Privacy-Focused**: Secure data handling with Google Cloud Firestore and anonymization practices

---

## Tech Stack

### Frontend
- HTML5
- Tailwind CSS
- JavaScript (Vanilla)

### Backend
- Python 3.8+
- Flask (Web Framework)
- Flask-Login (Authentication)
- Flask-Cors (Cross-Origin Resource Sharing)
- WTForms (Form Validation)

### Database
- Google Cloud Firestore

### Tools / Frameworks / APIs
- **AI/NLP**: Google Gemini API (gemini-2.0-flash model)
- **Sentiment Analysis**: TextBlob
- **Notifications**: Pushbullet
- **Deployment**: Gunicorn (Production Server)
- **Text Processing**: js2py
- **Environment Management**: python-dotenv

---

## Project Architecture / Workflow

The application follows a modular Flask-based architecture:

1. **User Authentication**: Secure signup/login system with session management
2. **Daily Check-in Flow**:
   - User initiates a chat session
   - AI agent engages in empathetic conversation
   - Real-time emotion classification and sentiment analysis
   - Potential crisis indicators trigger supportive resources
   - Conversation summary and mood data securely stored in Firestore
3. **Analytics Pipeline**:
   - Aggregate mood data over time periods (7 days, 30 days, 90 days)
   - Generate visualizations and identify patterns
   - Calculate streaks and engagement metrics
4. **Community Features**:
   - Anonymous posting and polling system
   - Peer support and interaction
5. **Admin Management**:
   - User monitoring and analytics
   - System health checks
   - Content moderation

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Google Cloud Firestore account
- Google Gemini API key

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ganesh-0509/Sahaay-Ai.git
   cd Sahaay-Ai
   ```

2. **Create and activate a virtual environment:**
   
   **Windows:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Cloud Firestore:**
   - Create a Firebase project in the [Google Cloud Console](https://console.cloud.google.com/)
   - Enable Firestore Database
   - Generate and download a service account JSON key
   - Save the key file in a secure location outside your repository
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to this file
   
   **Important Security Note:** Never commit service account credentials or API keys to version control. Add `.env` and all credential files to `.gitignore`.

5. **Configure environment variables:**
   
   Create a `.env` file in the root directory (ensure it's in `.gitignore`):
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   SECRET_KEY=your_flask_secret_key_here
   PUSHBULLET_API_TOKEN=your_pushbullet_token_here
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
   ```
   
   **Security Best Practices:**
   - Never share or commit your `.env` file
   - Use strong, randomly generated values for `SECRET_KEY`
   - Rotate API keys periodically
   - Restrict API key permissions to only required services

6. **Run the application:**
   ```bash
   python app.py
   ```

7. **Access the application:**
   
   Open your browser and navigate to `http://127.0.0.1:5000`

---

## Usage

### Running the Application

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. The application will be available at `http://localhost:5000`

### User Workflow

1. **Sign Up / Login**: Create a new account or login with existing credentials
2. **Daily Check-in**: Navigate to the chat interface and start a conversation
3. **Express Yourself**: Share your thoughts and feelings with the AI companion
4. **View Analytics**: Check your mood dashboard to see emotional trends
5. **Explore Community**: Participate in community polls and read supportive posts
6. **Access Tools**: Use personalized coping strategies and resources

### Admin Access

- Navigate to `/admin` route
- Login with admin credentials
- Access user management, analytics, and system monitoring features

---

## Screenshots / Demo

*Screenshots and demo links will be added here*

- Dashboard Preview
- Chat Interface
- Analytics Page
- Community Features

---

## Folder Structure

```
sahaay-ai/
├── admin/                    # Admin panel module
│   ├── admin_auth.py        # Admin authentication
│   ├── admin_routes.py      # Admin routes and views
│   ├── static/              # Admin-specific assets
│   └── templates/           # Admin HTML templates
├── agents/                   # AI agent modules
│   ├── coping_tip_agent.py  # Coping strategies generator
│   ├── crisis_agent.py      # Crisis detection agent
│   └── gemini_agent.py      # Main Gemini AI integration
├── models/                   # Data models
│   ├── forms.py             # WTForms definitions
│   └── user.py              # User model and authentication
├── routes/                   # Application routes
│   ├── api_routes.py        # API endpoints
│   ├── auth_routes.py       # Authentication routes
│   ├── chat_routes.py       # Chat functionality
│   ├── community_routes.py  # Community features
│   └── dashboard_routes.py  # Dashboard and analytics
├── static/                   # Frontend assets (CSS, JS)
├── templates/                # HTML templates
├── translations/             # Multi-language support
├── utils/                    # Utility functions
│   ├── emotion_classifier.py # Emotion detection
│   ├── sentiment.py         # Sentiment analysis
│   └── helpers.py           # Helper functions
├── app.py                    # Main application entry point
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Ethical AI & Responsible Use

Sahaay-AI is designed as a supportive tool to complement mental wellness practices, not as a replacement for professional care.

**Important Disclaimers:**

- **Not a Medical Device**: This application does not provide medical advice, diagnosis, or treatment. It is designed for wellness support and self-reflection only.

- **Crisis Detection Limitations**: While the system includes crisis detection capabilities, it cannot guarantee identification of all crisis situations. If you are experiencing a mental health emergency, please contact emergency services or a crisis hotline immediately.

- **Professional Care**: Users experiencing persistent mental health challenges are strongly encouraged to seek support from licensed mental health professionals, therapists, or counselors.

- **Data Privacy**: User conversations and mood data are handled with strict privacy measures. All community interactions are anonymized to protect user identity.

- **AI Limitations**: The AI companion provides empathetic responses based on language patterns, but does not possess human understanding or clinical training. Responses should be interpreted as supportive conversation, not professional guidance.

- **User Responsibility**: Users are responsible for their own well-being and should use this tool as one of many resources in their mental health journey.

**Crisis Resources:**

If you or someone you know is in crisis:
- **US**: National Suicide Prevention Lifeline: 988
- **India**: AASRA: +91-9820466726
- **International**: Find resources at [findahelpline.com](https://findahelpline.com)

---

## Future Enhancements

- **Voice-based Check-ins**: Allow users to record voice notes for mood tracking
- **Advanced Analytics**: Machine learning models for predictive mood analysis
- **Mobile Application**: Native iOS and Android apps
- **Therapist Integration**: Connect users with licensed mental health professionals
- **Journaling Feature**: Built-in diary functionality with prompts
- **Group Support Sessions**: Virtual support groups and moderated discussions
- **Wearable Integration**: Sync with fitness trackers for holistic health insights
- **Export Data**: Allow users to download their mental health data
- **Push Notifications**: Smart reminders for daily check-ins
- **Dark Mode**: Enhanced UI with theme customization

---

## Contributing

Contributions are welcome and appreciated! Here's how you can contribute:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Make your changes and commit:**
   ```bash
   git commit -m 'Add: Your feature description'
   ```
4. **Push to your branch:**
   ```bash
   git push origin feature/YourFeatureName
   ```
5. **Open a Pull Request** with a clear description of your changes

**Contribution Guidelines:**
- Follow Python PEP 8 style guidelines
- Write clear commit messages
- Test your changes thoroughly before submitting
- Update documentation if needed

---

## License

This project is open-source and available under the MIT License.

For full license details, see the [LICENSE](LICENSE) file in the repository.

---

## Author

**Ganesh**

Developed with a commitment to making mental health support accessible, approachable, and stigma-free for everyone.

---

## Acknowledgments

This project was built using the following technologies and frameworks:

- Flask - Web framework for Python
- Google Gemini API - AI-powered conversational capabilities
- Google Cloud Firestore - Scalable NoSQL database
- Tailwind CSS - Utility-first CSS framework
- TextBlob - Natural language processing library
- Pushbullet - Notification system

Special thanks to the open-source community for providing tools that enable accessible mental health technology.

