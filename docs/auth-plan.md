# Authentication & Persistence Plan for DiffViews Gradio App

## Executive Summary

This plan covers authentication and persistence for the DiffViews Gradio visualization app. The app already has multi-user support via `gr.State` for per-session state and thread-safe data handling. Authentication adds access control and user identity; persistence enables state survival across sessions.

---

## 1. Why Authentication & Persistence?

### 1.1 User Benefits

| Benefit | Requires Auth | Also Requires | Phase |
|---------|---------------|---------------|-------|
| **Access control** (private deployments) | ✓ | - | 1 |
| **Session persistence** (selections, neighbors, trajectories) | ✓ | Database | 3 |
| **Cross-device continuity** | ✓ | Database | 3 |
| **Generation history** (view past outputs) | ✓ | Image storage | 3 |
| **Saved workspaces** (favorite regions, neighbor sets) | ✓ | Database | 3 |
| **Personalized defaults** (steps, sigma, model) | ✓ | User prefs storage | 4 |
| **Bookmarked samples** | ✓ | Database | 4 |

**Key insight:** Auth alone gives *identity*, not persistence. To persist across time/machines, you need:
- User database (SQLite → Postgres)
- Storage for generated images
- API to save/load state

### 1.2 Admin/Developer Benefits

| Benefit | Requires Auth | Also Requires | Phase |
|---------|---------------|---------------|-------|
| **Access control** (restrict to authorized users) | ✓ | - | 1 |
| **Request attribution** (log username with actions) | ✓ | Logging | 2 |
| **Usage analytics** (who, what, when) | ✓ | Logging | 2 |
| **Generation audit trail** | ✓ | Log storage | 2 |
| **Per-user rate limiting** | ✓ | Rate limit middleware | 2 |
| **Block malicious users** | ✓ | Ban list/database | 3 |
| **Cost attribution** (GPU usage per user) | ✓ | Metering | 4 |
| **Tiered access** (free vs paid features) | ✓ | Entitlements system | 4 |
| **A/B testing** | ✓ | Feature flag system | 4 |

### 1.3 What Auth Alone Provides (No Persistence)

- **Access control** - only authorized users can access
- **Request attribution** - log `username` with each action
- **Session-scoped rate limiting** - via Gradio queue (not per-user)

---

## 2. Authentication Options

### 2.1 Gradio Native Capabilities

| Method | Complexity | Description |
|--------|------------|-------------|
| **Username/Password Tuple** | Trivial | Single hardcoded credential |
| **Username/Password List** | Simple | Multiple hardcoded credentials |
| **Auth Function** | Moderate | Custom validation logic |
| **HuggingFace OAuth** | Moderate | Built-in HF login (Spaces only) |
| **External OAuth via FastAPI** | Complex | Google, GitHub, etc. via `auth_dependency` |

### 2.2 What Requires External Setup

- **OAuth providers** (Google, GitHub, Auth0) - Require app registration
- **SSO/OIDC** - Requires enterprise identity provider
- **Rate limiting** - Not built into Gradio; requires middleware
- **MFA** - Not supported natively; requires external platform

---

## 3. Implementation Options (Simple → Advanced)

### Option A: Basic Username/Password (Simplest)

**Complexity:** Trivial | **Best for:** Local testing, private demos

```python
# Current launch (cli.py):
app.queue(max_size=20).launch(
    server_name="0.0.0.0",
    server_port=args.port,
    share=args.share,
)

# With basic auth:
app.queue(max_size=20).launch(
    server_name="0.0.0.0",
    server_port=args.port,
    share=args.share,
    auth=("admin", os.environ.get("DIFFVIEWS_PASSWORD", "changeme")),
)

# Multiple users:
auth=[
    ("user1", os.environ.get("USER1_PASS")),
    ("user2", os.environ.get("USER2_PASS")),
]
```

| Pros | Cons |
|------|------|
| Zero external dependencies | Credentials in memory/env vars |
| Works everywhere | No user management UI |
| Immediate setup | Shared passwords for demos |
| Works with `--share` | No session persistence |

### Option B: Auth Function with Environment Variables

**Complexity:** Simple | **Best for:** Self-hosted deployments

```python
import os
import json

def authenticate(username: str, password: str) -> bool:
    """Validate against environment variable credentials."""
    users_json = os.environ.get("DIFFVIEWS_USERS", '{}')
    users = json.loads(users_json)
    return users.get(username) == password

# Environment: DIFFVIEWS_USERS='{"alice":"pass1","bob":"pass2"}'
app.launch(auth=authenticate)
```

| Pros | Cons |
|------|------|
| Flexible validation logic | Still basic password auth |
| Can integrate with external systems | No OAuth/SSO |
| Users manageable via env vars | Password rotation requires restart |

### Option C: HuggingFace OAuth (Spaces-Native)

**Complexity:** Moderate | **Best for:** HuggingFace Spaces deployment

Requires metadata in Space README.md:
```yaml
hf_oauth: true
hf_oauth_expiration_minutes: 480
hf_oauth_authorized_org: your-org-name  # Optional: restrict to org members
```

```python
import gradio as gr

# Add login button to UI
login_btn = gr.LoginButton()

# Access user info in event handlers
def generate_with_user(
    selected_idx, neighbors, current_model,
    profile: gr.OAuthProfile | None,
):
    if profile:
        print(f"Generation by: {profile.username}")
    # ... existing logic
```

| Pros | Cons |
|------|------|
| No password management | HF Spaces only |
| Users have HF profiles | Does NOT restrict access (login optional) |
| Token available for HF API | Requires HF account |

**Important:** `gr.LoginButton` does NOT restrict access. Unauthenticated users can still use the app; `gr.OAuthProfile` will be `None`.

### Option D: External OAuth via FastAPI Mount

**Complexity:** High | **Best for:** Production deployments with enterprise auth

```python
from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
import gradio as gr
import os

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.environ["SESSION_SECRET"])

oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.environ["GOOGLE_CLIENT_ID"],
    client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for('auth_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user = token.get('userinfo')
    request.session['user'] = dict(user)
    return RedirectResponse(url='/gradio')

def get_user(request: Request) -> str | None:
    user = request.session.get('user')
    return user['email'] if user else None

# Mount Gradio with auth_dependency
demo = create_gradio_app(visualizer)
app = gr.mount_gradio_app(app, demo, path="/gradio", auth_dependency=get_user)
```

| Pros | Cons |
|------|------|
| Enterprise OAuth (Google, Okta) | Significant code restructure |
| Session management | Cannot use `auth` param |
| True access restriction | More infrastructure needed |

---

## 4. Potential Pitfalls and Gotchas

### 4.1 Critical Issues

1. **`auth` and `auth_dependency` are mutually exclusive** - Cannot combine
2. **Third-party cookies required** - Disabled in Safari/Chrome Incognito
3. **HF OAuth does NOT restrict access** - Login is optional
4. **`auth_dependency` has reported issues** - GitHub #8096
5. **Session not available in auth function** - `gr.Request` not passed

### 4.2 Common Mistakes

- Hardcoding passwords in source code (use env vars)
- Using HTTP without HTTPS for OAuth callbacks
- Forgetting `SessionMiddleware` when using external OAuth
- Not setting `FORWARDED_ALLOW_IPS` behind reverse proxy
- Using `share=True` with sensitive data (tunnels through Gradio servers)

### 4.3 Breaking Changes to Watch

- Gradio 5 made significant security fixes - ensure 5.0+
- `auth_dependency` behavior may change between versions
- HF Spaces OAuth configuration format may evolve

---

## 5. Security Concerns

### 5.1 Password Storage

| Method | Risk Level | Recommendation |
|--------|------------|----------------|
| Hardcoded in code | **HIGH** | Never do this |
| Environment variables | Medium | OK for deployment |
| Secrets manager (Vault, AWS) | Low | Production recommendation |
| HF Spaces secrets | Low | Use for HF deployments |

### 5.2 Session Management

- Gradio uses cookies for session tracking
- Sessions tied to browser, not user accounts (with basic auth)
- No built-in session timeout (OAuth tokens have expiration)
- `/logout` clears all sessions; use `?all_session=false` for current only

### 5.3 HTTPS Requirements

- **OAuth absolutely requires HTTPS**
- Basic auth over HTTP exposes credentials
- `share=True` uses HTTPS tunnel but data passes through Gradio servers
- Self-hosted: Use nginx/Caddy with SSL termination

### 5.4 Rate Limiting

Gradio has no built-in rate limiting. Options:

1. **Gradio queue** - Already configured with `max_size=20`
2. **FastAPI middleware** - Use `slowapi`
3. **Reverse proxy** - nginx `limit_req_zone`

### 5.5 Security Checklist

- [ ] Passwords never hardcoded in source
- [ ] HTTPS enabled for all external access
- [ ] Environment variables for all credentials
- [ ] `GRADIO_ALLOWED_PATHS` set to restrict file access
- [ ] `GRADIO_BLOCKED_PATHS` includes sensitive directories
- [ ] Using Gradio 5.0+ (security audit fixes)
- [ ] Rate limiting configured (queue or proxy)
- [ ] `FORWARDED_ALLOW_IPS` set if behind proxy
- [ ] Secrets excluded from git (`.gitignore` has `.env`)

---

## 6. Deployment-Specific Considerations

### 6.1 HuggingFace Spaces

**Recommended:** HF OAuth with optional `hf_oauth_authorized_org`

```yaml
# README.md metadata
hf_oauth: true
hf_oauth_expiration_minutes: 480
hf_oauth_authorized_org: your-org-name
```

### 6.2 Modal

**Recommended:** Basic auth with Modal secrets for demos; FastAPI+OAuth for production

```python
from modal import Secret

@app.function(secrets=[Secret.from_name("diffviews-auth")])
def main():
    password = os.environ["DIFFVIEWS_PASSWORD"]
```

### 6.3 Self-Hosted

**Recommended:** OAuth via FastAPI mount for enterprise; basic auth for internal tools

```
nginx (SSL, rate limiting, proxy)
  └── Gunicorn (Python ASGI server)
       └── FastAPI (auth routes)
            └── Gradio (mounted at /app)
```

---

## 7. Persistence Layer Design

### 7.1 Architecture Overview

```
Current:  Gradio → gr.State (ephemeral, per-browser)

With persistence:
  Gradio → Auth (identity)
        → SQLite Database (user prefs, saved states, generations)
        → File Storage (generated images)
        → Usage Logging (analytics, audit)
```

### 7.2 Data Model

```
+------------------+       +--------------------+       +------------------+
|      users       |       |    generations     |       |  saved_states    |
+------------------+       +--------------------+       +------------------+
| id (PK)          |<------| user_id (FK)       |       | id (PK)          |
| auth_provider    |       | id (PK)            |       | user_id (FK)     |
| auth_id          |       | timestamp          |       | name             |
| email            |       | model_name         |       | model_name       |
| display_name     |       | class_id           |       | selected_idx     |
| created_at       |       | num_steps, etc.    |       | manual_neighbors |
| last_login       |       | image_path         |       | knn_neighbors    |
+------------------+       +--------------------+       +------------------+
                                                               |
                           +------------------+                |
                           |   usage_logs     |<---------------+
                           +------------------+
                           | id (PK)          |
                           | user_id (FK)     |
                           | action           |
                           | timestamp        |
                           | metadata_json    |
                           +------------------+
```

### 7.3 SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    auth_provider TEXT NOT NULL DEFAULT 'basic',
    auth_id TEXT NOT NULL,
    email TEXT,
    display_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(auth_provider, auth_id)
);

CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name TEXT NOT NULL,
    class_id INTEGER,
    class_name TEXT,
    num_steps INTEGER NOT NULL,
    mask_steps INTEGER NOT NULL,
    guidance_scale REAL NOT NULL,
    sigma_max REAL NOT NULL,
    sigma_min REAL NOT NULL,
    neighbor_indices TEXT,
    image_path TEXT NOT NULL,
    thumbnail_path TEXT,
    metadata_json TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX idx_generations_user_time ON generations(user_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS saved_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    selected_idx INTEGER,
    manual_neighbors TEXT,
    knn_neighbors TEXT,
    knn_k INTEGER DEFAULT 5,
    highlighted_class INTEGER,
    trajectory_coords TEXT,
    metadata_json TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(user_id, name)
);

CREATE TABLE IF NOT EXISTS usage_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name TEXT,
    metadata_json TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);
CREATE INDEX idx_usage_logs_action ON usage_logs(action, timestamp DESC);
```

### 7.4 File Storage Structure

```
data/
├── dmd2/                          # Existing model data
├── edm/                           # Another model
└── user_data/                     # NEW: User-specific storage
    ├── .db/
    │   └── diffviews.sqlite
    └── users/
        ├── {user_id}/
        │   ├── generations/
        │   │   ├── 2026-01/
        │   │   │   ├── gen_001_1705678901.png
        │   │   │   └── gen_001_1705678901_thumb.png
        │   │   └── 2026-02/
        │   └── exports/
        └── anonymous/
```

### 7.5 Module Structure

```
diffviews/
├── persistence/
│   ├── __init__.py      # PersistenceManager entry point
│   ├── database.py      # Thread-safe SQLite connection
│   ├── models.py        # User, Generation, SavedState, UsageLog
│   ├── repositories.py  # CRUD operations
│   └── storage.py       # File storage utilities
```

### 7.6 Thread Safety

- **SQLite per-thread connections**: `Database` uses `threading.local()`
- **Read-only shared data**: Existing `model_data` dict unchanged
- **Per-session state**: User identity in `gr.State`
- **Atomic operations**: All persistence ops isolated

---

## 8. Phased Implementation Plan

### Phase 1: Basic Auth (CLI Flag)

**Goal:** Access control without persistence

**Changes:**
- Add `--auth username:password` CLI flag to `cli.py`
- Add `--auth-file` for JSON credentials file
- Environment variable `DIFFVIEWS_AUTH_USERS`

```python
# cli.py additions
gradio_parser.add_argument("--auth", type=str, help="username:password")
gradio_parser.add_argument("--auth-file", type=str, help="JSON {user: pass}")

# In visualize_gradio_command():
auth = None
if args.auth:
    user, passwd = args.auth.split(":", 1)
    auth = (user, passwd)
elif args.auth_file:
    with open(args.auth_file) as f:
        users = json.load(f)
    auth = [(u, p) for u, p in users.items()]
elif os.environ.get("DIFFVIEWS_AUTH_USERS"):
    users = json.loads(os.environ["DIFFVIEWS_AUTH_USERS"])
    auth = [(u, p) for u, p in users.items()]

app.launch(..., auth=auth)
```

**Deliverables:**
- [ ] CLI flags for auth
- [ ] Environment variable support
- [ ] Documentation update

### Phase 2: Usage Logging

**Goal:** Track actions for analytics (no UI changes)

**Changes:**
- Create `diffviews/persistence/` module (database, models, repositories)
- Add `--enable-logging` flag
- Log actions: generate, select, model_switch, etc.

```python
# In on_generate, after successful generation:
if visualizer.usage_logger:
    visualizer.usage_logger.log(
        action='generate',
        user_id=username,  # From auth
        model_name=model_name,
        metadata={'class_id': class_label, 'neighbors': len(all_neighbors)}
    )
```

**Deliverables:**
- [ ] Persistence module (database.py, models.py, repositories.py)
- [ ] Usage logging in key handlers
- [ ] Simple analytics query functions

### Phase 3: Persistence Layer

**Goal:** Save generations and workspaces

**Changes:**
- Add `user_state = gr.State(value=None)` for user identity
- Wire up generation saving in `on_generate`
- Add Save/Load Workspace UI
- Add History gallery (collapsed accordion)

**UI Additions:**
```python
# In right sidebar, after generation section:
with gr.Accordion("History", open=False):
    history_gallery = gr.Gallery(label="Recent", columns=4, rows=2)
    refresh_history_btn = gr.Button("Refresh", size="sm")

# In left sidebar or modal:
with gr.Accordion("Workspaces", open=False):
    workspace_name = gr.Textbox(label="Name", placeholder="my-workspace")
    save_workspace_btn = gr.Button("Save Current", size="sm")
    workspace_dropdown = gr.Dropdown(label="Load", choices=[])
    load_workspace_btn = gr.Button("Load", size="sm")
```

**Deliverables:**
- [ ] Generation history persistence
- [ ] Workspace save/load
- [ ] History gallery UI
- [ ] File storage for images

### Phase 4: HF OAuth Integration

**Goal:** Proper user identity for HF Spaces

**Changes:**
- Add `gr.LoginButton` to UI
- Handle `gr.OAuthProfile` in handlers
- Wire user identity to persistence layer

```python
# In create_gradio_app():
with gr.Row():
    login_btn = gr.LoginButton()
    logout_btn = gr.Button("Logout", link="/logout", visible=False)

# In handlers, add profile parameter:
def on_generate(..., profile: gr.OAuthProfile | None):
    user = None
    if profile and visualizer.persistence:
        user = visualizer.persistence.users.get_or_create(
            auth_provider="huggingface",
            auth_id=profile.username,
            display_name=profile.name
        )
    # ... rest of generation
```

**Deliverables:**
- [ ] HF OAuth login button
- [ ] User identity wired to persistence
- [ ] Profile display in UI

### Phase 5: FastAPI Migration (Optional)

**Goal:** Enterprise OAuth for self-hosted/Modal

**Changes:**
- Restructure to mount Gradio in FastAPI
- Add OAuth routes (Google, GitHub, etc.)
- Session middleware for auth state

**Only if needed for enterprise deployments.**

---

## 9. Recommended Approach by Use Case

| Use Case | Auth | Persistence | Phases |
|----------|------|-------------|--------|
| Local development | None | None | - |
| Private demo | Basic CLI | None | 1 |
| HF Spaces public | HF OAuth | Optional | 1, 4 |
| HF Spaces private | HF OAuth + org | Full | 1, 2, 3, 4 |
| Modal demo | Basic + secrets | Logging | 1, 2 |
| Modal production | FastAPI OAuth | Full | 1-5 |
| Self-hosted internal | Basic function | Logging | 1, 2 |
| Self-hosted enterprise | FastAPI + OIDC | Full | 1-5 |

---

## 10. Critical Files

| File | Purpose |
|------|---------|
| `diffviews/scripts/cli.py` | Add auth CLI flags, launch params |
| `diffviews/visualization/gradio_app.py` | Add persistence calls, UI elements |
| `diffviews/persistence/` (new) | Database, models, repositories, storage |
| `pyproject.toml` | Add optional persistence dependencies |
| `tests/test_persistence.py` (new) | Unit tests for persistence |

---

## 11. Sources

- [Gradio Sharing Your App Guide](https://www.gradio.app/guides/sharing-your-app)
- [Gradio mount_gradio_app Docs](https://www.gradio.app/docs/gradio/mount_gradio_app)
- [HuggingFace Spaces OAuth](https://huggingface.co/docs/hub/en/spaces-oauth)
- [Gradio 5 Security Audit](https://huggingface.co/blog/gradio-5-security)
- [Gradio Environment Variables](https://www.gradio.app/guides/environment-variables)
