# laya-trader in a container (any Docker host): live paper trading and a daily backtest on CPU.
# The free Hugging Face Space uses app.py instead (see deploy_space.sh).
FROM python:3.11-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    LAYA_BACKEND=torch \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt

USER user
WORKDIR /home/user/app
# Bake the checkpoint into the image so a restart doesn't download it again.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('convaiinnovations/laya', allow_patterns=['multilingual/*', '*.json', '*.py'])"

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "laya_trader.py", "--host", "0.0.0.0", "--port", "7860", "--public", "--no-open", \
     "--interval", "10", "--backtest-every-hours", "24", "--backtest-days", "30"]
