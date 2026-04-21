import json

import requests


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def resolution(market):
    result = (market.get("result") or "").lower()
    if result == "yes":
        return "YES"
    if result == "no":
        return "NO"
    return None


def crawl():
    cursor = None
    while True:
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = requests.get(f"{BASE_URL}/events", params=params, timeout=60).json()
        events = data.get("events") or []
        if not events:
            break
        for event in events:
            if (event.get("category") or "").lower() not in {"politics", "elections", "government"}:
                continue
            detail = requests.get(f"{BASE_URL}/events/{event.get('event_ticker')}", timeout=60).json()
            for market in detail.get("markets", []):
                resolved = resolution(market)
                resolution_time = market.get("settlement_ts") or market.get("updated_time")
                if market.get("market_type") == "binary" and resolved and resolution_time:
                    yield {
                        "source": "kalshi",
                        "question": market.get("title"),
                        "description": market.get("rules_primary"),
                        "resolution": resolved,
                        "created_time": market.get("created_time"),
                        "close_time": market.get("close_time"),
                        "resolution_time": resolution_time,
                        "forecast_cutoff_time": resolution_time,
                        "market_url": f"{BASE_URL}/markets/{market.get('ticker')}",
                        "question_id": f"kalshi:{market.get('ticker')}",
                    }
        cursor = data.get("cursor")
        if not cursor:
            break


def main():
    with open("kalshi.jsonl", "w", encoding="utf-8") as f:
        for row in crawl():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Done")


if __name__ == "__main__":
    main()
