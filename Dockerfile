FROM python:3.11-slim

# Copy pre-built uv binary for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install project dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Copy application source code and data assets
COPY src/ ./src/
COPY data/ ./data/
COPY app.py ./

EXPOSE 8501

# Start Streamlit bound to container interface
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]