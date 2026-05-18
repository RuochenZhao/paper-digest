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
    """Scrape paper IDs from a HuggingFace papers page, then enrich via HF API."""
    headers = {"User-Agent": "Mozilla/5.0 (research-bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    arxiv_ids = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/papers/") and href.count("/") == 2:
            arxiv_id = href.replace("/papers/", "").strip()
            if arxiv_id and arxiv_id not in seen:
                seen.add(arxiv_id)
                arxiv_ids.append(arxiv_id)

    papers = []
    for arxiv_id in arxiv_ids:
        paper = fetch_hf_paper_api(arxiv_id)
        if paper:
            papers.append(paper)
        time.sleep(0.1)  # be polite

    return papers


def fetch_hf_paper_api(arxiv_id: str) -> dict | None:
    """Fetch paper metadata including upvotes from HF API."""
    url = f"https://huggingface.co/api/papers/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        
        # Extract institution from authors if available
        institution = ""
        authors = data.get("authors", [])
        if authors:
            # HF API sometimes includes affiliations
            affiliations = []
            for author in authors[:3]:  # check first 3 authors
                aff = author.get("affiliations", [])
                if aff:
                    affiliations.extend(aff)
            if affiliations:
                institution = affiliations[0] if isinstance(affiliations[0], str) else ""

        return {
            "arxiv_id": arxiv_id,
            "title": data.get("title", ""),
            "blurb": data.get("summary", data.get("abstract", "")),
            "upvotes": data.get("upvotes", 0),
            "institution": institution,
            "hf_url": f"https://huggingface.co/papers/{arxiv_id}",
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        }
    except Exception:
        return None


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
        inst_line = f"    Institution: {p['institution']}\n" if p.get("institution") else ""
        paper_list += (
            f"[{i}] Title: {p['title']}\n"
            f"{inst_line}"
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
2. For each selected paper, write a compact entry. Keep the summary to 2 sentences max
   (one sentence on problem/method, one on key result). Be concise and precise.
   Include institution only if clearly stated in the paper metadata — skip if uncertain.
3. Output as a clean digest. Format each entry exactly like:

📄 *Title*
🏛 Institution (omit this line if unknown)
Summary: <2 sentences>
Relevance: <1 sentence>
🔗 HF: <hf_url> | arXiv: <arxiv_url>
👍 <upvotes> upvotes

End with a 2-sentence "Big Picture" on any trends across today's papers.

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