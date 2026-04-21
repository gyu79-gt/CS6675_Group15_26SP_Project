import json

import requests


BASE_URL = "https://gamma-api.polymarket.com"


def parse_json(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value or default


def resolved_side(prices):
    if len(prices) != 2:
        return None
    try:
        yes, no = float(prices[0]), float(prices[1])
    except (TypeError, ValueError):
        return None
    if yes >= 0.99:
        return "YES"
    if no >= 0.99:
        return "NO"
    return None


def crawl():
    seen = set()
    for query in ["election", "trump", "president"]:
        page = 1
        while True:
            data = requests.get(
                f"{BASE_URL}/public-search",
                params={"q": query, "events_status": "closed", "keep_closed_markets": 1, "limit_per_type": 50, "page": page},
                timeout=60,
            ).json()
            events = data.get("events") or []
            if not events:
                break
            for event in events:
                for market in event.get("markets") or []:
                    market_id = str(market.get("id"))
                    if market_id in seen:
                        continue
                    seen.add(market_id)
                    outcomes = parse_json(market.get("outcomes"), [])
                    resolution = resolved_side(parse_json(market.get("outcomePrices"), []))
                    resolution_time = market.get("closedTime")
                    if outcomes != ["Yes", "No"] or not resolution or not resolution_time:
                        continue
                    yield {
                        "source": "polymarket",
                        "question": market.get("question"),
                        "description": market.get("description"),
                        "resolution": resolution,
                        "created_time": market.get("startDate"),
                        "close_time": (market.get("endDate") or "").replace(" ", "T"),
                        "resolution_time": resolution_time.replace(" ", "T").replace("+00", "+00:00"),
                        "forecast_cutoff_time": resolution_time.replace(" ", "T").replace("+00", "+00:00"),
                        "market_url": f"https://polymarket.com/event/{market.get('slug') or market_id}",
                        "question_id": f"polymarket:{market_id}",
                    }
            if not data.get("pagination", {}).get("hasMore"):
                break
            page += 1


def main():
    count = 0
    with open("polymarket.jsonl", "w", encoding="utf-8") as f:
        for row in crawl():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"Done: {count} records")


if __name__ == "__main__":
    main()
