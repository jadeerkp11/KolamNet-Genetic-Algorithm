FROM python:3.12-slim

# Non-root user (required by Hugging Face Spaces, good practice everywhere)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1 PORT=8501
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 8501
# Hosts such as Render/Railway inject $PORT; default is 8501.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
