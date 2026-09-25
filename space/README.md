---
title: Laya Trader
emoji: 📈
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
python_version: "3.12"
pinned: false
license: apache-2.0
short_description: Paper trading crypto and S&P 500 stocks with Laya
---

# laya-trader

Paper trading with a small AI model. Every 10 seconds, [Laya](https://huggingface.co/convaiinnovations/laya) reads market signals for **crypto** (Binance) and **S&P 500 stocks** (Yahoo Finance) and answers one question: *is the short-term outlook bullish?* A strategy turns that probability into LONG, SHORT, CLOSE or HOLD on a paper account.

- **Live** tab: every decision and exactly what Laya read.
- **Backtest** tab: the last 30 days, rebuilt daily on this Space's CPU.

**Paper trading only.** No API keys, no orders. Not financial advice. This dashboard is read-only; the paper account restarts when the Space restarts.

Source, docs and results: [github.com/antonellof/laya-trader](https://github.com/antonellof/laya-trader)
