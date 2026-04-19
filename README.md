# AI Interview Simulator - Backend

A FastAPI-based backend service for AI-powered interview simulation and evaluation.

## Features

- 🔐 JWT-based authentication with Google OAuth support
- 🤖 AI-powered interview sessions using LangChain and LangGraph
- 📝 Real-time WebSocket connections for live interviews
- 🎤 Audio processing and transcription services
- 📊 Performance evaluation and reporting
- 💾 PostgreSQL database with async support
- 🗃️ Vector storage with ChromaDB for RAG
- ☁️ Cloudflare R2 integration for file storage

## Tech Stack

- **Framework**: FastAPI with Python 3.11
- **Database**: PostgreSQL with asyncpg
- **Authentication**: JWT with Google OAuth
- **AI/ML**: OpenAI/Groq APIs, LangChain, LangGraph
- **Vector DB**: ChromaDB
- **Storage**: Cloudflare R2
- **Cache**: Redis (optional)
- **Deployment**: Railway

## Prerequisites

- Python 3.11+
- PostgreSQL database
- Redis (optional, for OAuth state)

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <backend-repo-url>
   cd InterviewPilot-Backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Configure your environment in `.env`:
   ```env
   # Database
   database_url=postgresql+asyncpg://username:password@host:port/database
   
   # Auth
   jwt_secret=your-super-secret-jwt-key-here
   
   # AI Services
   openai_api_key=your-openai-or-groq-api-key
   openai_base_url=https://api.groq.com/openai/v1
   
   # CORS
   cors_origins=*
   ```

5. **Run database migrations** (if applicable)
   ```bash
   # Tables are auto-created in development mode
   # Set auto_create_tables=false in production
   ```

6. **Start the server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Access the API**
   - API Documentation: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `database_url` | PostgreSQL connection string | ✅ |
| `jwt_secret` | Secret key for JWT signing | ✅ |
| `openai_api_key` | OpenAI/Groq API key | ✅ |
| `openai_base_url` | OpenAI API base URL | ❌ |
| `google_client_id` | Google OAuth client ID | ❌ |
| `google_client_secret` | Google OAuth client secret | ❌ |
| `google_redirect_uri` | Google OAuth redirect URI | ❌ |
| `redis_url` | Redis connection URL | ❌ |
| `cors_origins` | Allowed CORS origins | ❌ |
| `r2_*` | Cloudflare R2 credentials | ❌ |

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration
- `GET /api/auth/google/start` - Google OAuth start

### Sessions
- `POST /api/sessions` - Create interview session
- `POST /api/sessions/{id}/start` - Start session
- `POST /api/sessions/{id}/end` - End session
- `GET /api/sessions/{id}/turns` - Get session turns

### History
- `GET /api/history` - Get user interview history
- `GET /api/history/{id}` - Get specific session report

### Uploads
- `POST /api/uploads/presign` - Get upload presigned URL

## WebSocket

### Interview Socket
- `WS /ws/interview/{session_id}` - Real-time interview communication

## Deployment

### Railway (Recommended)
1. Connect your repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push to main branch

### Docker
```bash
docker build -t interview-pilot-backend .
docker run -p 8000:8000 interview-pilot-backend
```

## Project Structure

```
app/
├── api/
│   └── routes/         # API route handlers
├── core/               # Core utilities (security, config, etc.)
├── db/                 # Database models and session
├── services/           # Business logic services
├── workers/            # Background workers
└── ws/                 # WebSocket handlers
```

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
isort .
```

### Type Checking
```bash
mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details




LangGraph Flow
                         ┌──────────────────────┐
                         │        START         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   determine_agent    │
                         │ (HR / TECH / MGR)   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     load_persona     │
                         │ (behavior, tone)     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  get_memory_context  │
                         │ (history summary)    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     retrieve_rag     │
                         │ (concept retrieval)  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  should_evaluate ?   │
                         │ (user_answer exists) │
                         └───────┬───────┬──────┘
                                 │       │
                   ┌─────────────┘       └─────────────┐
                   ▼                                   ▼
     ┌──────────────────────────┐        ┌──────────────────────────┐
     │     evaluate_answer      │        │    generate_question     │
     │ (score + feedback)       │        │ (first / next question)  │
     └──────────┬───────────────┘        └──────────┬───────────────┘
                │                                   │
                ▼                                   │
     ┌──────────────────────────┐                   │
     │   adjust_difficulty      │                   │
     │ (easy / med / hard)      │                   │
     └──────────┬───────────────┘                   │
                │                                   │
                ▼                                   │
     ┌──────────────────────────┐                   │
     │   generate_question      │◄──────────────────┘
     │ (adaptive question)      │
     └──────────┬───────────────┘
                │
                ▼
     ┌──────────────────────────┐
     │           END            │
     └──────────────────────────┘


## Update RAG Data
```bash
python -m app.scripts.load_rag_data
```