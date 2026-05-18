#!/usr/bin/env python3
"""
Daily AI Agent Paper Digest
Fetches papers from HuggingFace, filters by research interests,
generates summaries via Claude API, and sends to Telegram.
"""

import os
import re
import json
import time
import datetime
import requests
from bs4 import BeautifulSoup
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

TOPICS = [
    "trustworthy", "reliable", "safe", "safety",
    "multi-agent", "multi agent", "multiagent", "collaboration", "cooperative",
    "agent reasoning", "chain-of-thought", "planning", "reflection", "self-correction",
    "agent evaluation", "agent benchmark", "agentic",
    "tool use", "tool-use", "function calling",
    "hallucination", "faithfulness", "grounding",
]

SOURCES = [
    "https://huggingface.co/papers",          # daily
    "https://huggingface.co/papers/trending", # trending
]

MIN_UPVOTES = 5   # filter out very low-signal papers

# ── Scraping ──────────────────────────────────────────────────────────────────

def fetch_hf_papers(url: str) -> list[dict]:
    """Scrape paper cards from a HuggingFace papers page."""
    headers = {"User-Agent": "Mozilla/5.0 (research-bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    papers = []
    seen = set()

    # Each paper card is an <article> or an <h3> inside a card — HF uses <h3> titles
    for h3 in soup.find_all("h3"):
        a = h3.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        if not href.startswith("/papers/"):
            continue
        arxiv_id = href.replace("/papers/", "").strip()
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)

        title = a.get_text(strip=True)

        # Abstract / blurb — sibling <p> near the card
        blurb = ""
        parent = h3.find_parent()
        if parent:
            p = parent.find("p")
            if p:
                blurb = p.get_text(strip=True)

        # Upvotes — look for upvote count text nearby
        upvotes = 0
        card = h3.find_parent("div") or h3.find_parent("article")
        if card:
            for btn in card.find_all(string=re.compile(r"Upvote\s+\d+")):
                m = re.search(r"(\d+)", btn)
                if m:
                    upvotes = int(m.group(1))
                    break

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "blurb": blurb,
            "upvotes": upvotes,
            "hf_url": f"https://huggingface.co/papers/{arxiv_id}",
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        })

    return papers


def is_relevant(paper: dict) -> bool:
    """Check if paper matches our research interests."""
    text = (paper["title"] + " " + paper["blurb"]).lower()
    return any(kw in text for kw in TOPICS)


def fetch_arxiv_abstract(arxiv_id: str) -> str:
    """Fetch full abstract from arXiv API."""
    url = f"https://export.arxiv.org/abs/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        block = soup.find("blockquote", class_="abstract")
        if block:
            return block.get_text(strip=True).replace("Abstract:", "").strip()
    except Exception:
        pass
    return paper.get("blurb", "")


# ── Claude Summarisation ──────────────────────────────────────────────────────

def summarise_papers(papers: list[dict]) -> str:
    """Ask Claude to filter, rank and summarise the relevant papers."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    paper_list = ""
    for i, p in enumerate(papers, 1):
        paper_list += (
            f"[{i}] Title: {p['title']}\n"
            f"    HF Upvotes: {p['upvotes']}\n"
            f"    Abstract: {p['blurb']}\n"
            f"    HF: {p['hf_url']}\n"
            f"    arXiv: {p['arxiv_url']}\n\n"
        )

    prompt = f"""You are a research assistant for an NLP/AI professor at SUTD.
Her research interests are:
1. Trustworthy & reliable AI agents (safety, faithfulness, hallucination, grounding)
2. Multi-agent collaboration & coordination
3. Agent reasoning (planning, reflection, self-correction, chain-of-thought)
4. Agent evaluation & benchmarking

Today's date: {datetime.date.today().isoformat()}

Below are candidate papers scraped from HuggingFace Papers (trending + daily).
Please:
1. Select the TOP 5-8 most relevant and high-quality papers for her interests.
   Prioritise novelty, rigour, and direct relevance. Exclude peripheral or low-quality work.
2. For each selected paper, write:
   - A 3-sentence summary in English (what problem, what method, key result)
   - One sentence on why it's relevant to her research
   - Quality signal (upvotes, institution if inferable)
3. Output as a clean digest, no markdown tables, use emoji sparingly.
   Format each entry exactly like:

📄 *Title*
Summary: ...
Relevance: ...
🔗 HF: <url> | arXiv: <url>
👍 <upvotes> upvotes

End with a one-paragraph "Big Picture" noting any themes or trends across today's papers.

PAPERS:
{paper_list}
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(text: str):
    """Send message to Telegram, splitting if over 4096 chars."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        time.sleep(0.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Fetching papers from HuggingFace...")
    all_papers = []
    seen_ids = set()

    for source_url in SOURCES:
        papers = fetch_hf_papers(source_url)
        print(f"  {source_url}: {len(papers)} papers found")
        for p in papers:
            if p["arxiv_id"] not in seen_ids:
                seen_ids.add(p["arxiv_id"])
                all_papers.append(p)

    print(f"Total unique papers: {len(all_papers)}")

    # Filter by topic keywords first (cheap filter before calling API)
    relevant = [p for p in all_papers if is_relevant(p)]
    print(f"Relevant papers (keyword filter): {len(relevant)}")

    # Also keep high-upvote papers even if not keyword-matched
    # (Claude will do the final quality filter)
    high_signal = [p for p in all_papers if p["upvotes"] >= 20 and p not in relevant]
    candidates = relevant + high_signal
    print(f"Candidates sent to Claude: {len(candidates)}")

    if not candidates:
        send_telegram("🤖 *Daily Paper Digest*\n\nNo relevant papers found today. Check back tomorrow!")
        return

    # Cap at 30 to keep prompt reasonable
    candidates = sorted(candidates, key=lambda x: -x["upvotes"])[:30]

    print("Generating digest with Claude...")
    digest = summarise_papers(candidates)

    today = datetime.date.today().strftime("%B %d, %Y")
    header = f"🧠 *AI Agent Research Digest — {today}*\n\n"
    full_message = header + digest

    print("Sending to Telegram...")
    send_telegram(full_message)
    print("Done! ✅")


if __name__ == "__main__":
    main()