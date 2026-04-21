import json

import requests


BASE_URL = "https://api.manifold.markets/v0"


def crawl():
    before = None
    while True:
        params = {"limit": 50, "sort": "created-time", "order": "desc"}
        if before:
            params["before"] = before
        markets = requests.get(f"{BASE_URL}/markets", params=params, timeout=60).json()
        if not markets:
            break
        for market in markets:
            if market.get("outcomeType") != "BINARY" or not market.get("isResolved"):
                continue
            if market.get("resolution") not in ("YES", "NO"):
                continue
            detail = requests.get(f"{BASE_URL}/market/{market['id']}", timeout=60).json()
            yield {
                "source": "manifold",
                "question": market.get("question"),
                "description": detail.get("textDescription") or detail.get("description", ""),
                "resolution": market.get("resolution"),
                "created_time": market.get("createdTime"),
                "close_time": market.get("closeTime"),
                "resolution_time": market.get("resolutionTime"),
                "forecast_cutoff_time": market.get("resolutionTime"),
                "market_url": market.get("url"),
                "question_id": f"manifold:{market['id']}",
            }
        before = markets[-1].get("id")


def main():
    with open("manifold.jsonl", "w", encoding="utf-8") as f:
        for row in crawl():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Done")


if __name__ == "__main__":
    main()
