import requests
from bs4 import BeautifulSoup
import json
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120"
}

BASE = "https://www.shl.com"
catalog = []
seen_urls = set()

def scrape_listing_page(start):
    url = f"{BASE}/products/product-catalog/?start={start}&type=1"
    resp = requests.get(url, headers=headers, timeout=20)
    soup = BeautifulSoup(resp.text, "html.parser")
    
    items = []
    tables = soup.find_all("table")
    
    for table in tables:
        # Find the Individual Test Solutions table specifically
        th = table.find("th")
        if not th or "Individual" not in th.get_text():
            continue
        
        for row in table.find_all("tr")[1:]:  # skip header
            cols = row.find_all("td")
            if len(cols) < 4:
                continue
            
            link = cols[0].find("a")
            if not link:
                continue
            
            href = link.get("href", "")
            full_url = BASE + href if href.startswith("/") else href
            
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            
            # test_type: last column, grab all letter spans or plain text
            test_type_text = cols[3].get_text(separator=" ", strip=True)
            
            items.append({
                "name": link.get_text(strip=True),
                "url": full_url,
                "test_type": test_type_text,
                "description": "",
                "duration_minutes": None,
                "remote_testing": cols[1].get_text(strip=True) != "",
                "adaptive": cols[2].get_text(strip=True) != ""
            })
    
    return items

def enrich_description(url):
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Try multiple selectors
        for selector in ["main", "article", ".entry-content", "#main-content"]:
            el = soup.select_one(selector)
            if el:
                paragraphs = el.find_all("p")
                text = " ".join(p.get_text(strip=True) for p in paragraphs[:8])
                if len(text) > 80:
                    return text[:1500]
        
        # Fallback: get all body text
        body = soup.find("body")
        if body:
            return body.get_text(separator=" ", strip=True)[:1500]
    except Exception as e:
        print(f"    Error: {e}")
    return ""

# ── Step 1: Scrape all listing pages ──────────────────────────────
print("=== Scraping listing pages ===")
for start in range(0, 400, 12):
    print(f"Page start={start}...", end=" ")
    items = scrape_listing_page(start)
    
    if not items:
        print("No items found — done.")
        break
    
    catalog.extend(items)
    print(f"{len(items)} items. Total: {len(catalog)}")
    time.sleep(1.2)

print(f"\nTotal from listings: {len(catalog)}")

# ── Step 2: Enrich descriptions ───────────────────────────────────
print("\n=== Enriching descriptions ===")
for i, item in enumerate(catalog):
    print(f"[{i+1}/{len(catalog)}] {item['name'][:50]}")
    item["description"] = enrich_description(item["url"])
    time.sleep(0.8)

# ── Step 3: Save ──────────────────────────────────────────────────
json.dump(catalog, open("catalog.json", "w"), indent=2)

# ── Step 4: Report ────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Total items:           {len(catalog)}")
print(f"With test_type:        {sum(1 for x in catalog if x['test_type'])}")
print(f"With description >50:  {sum(1 for x in catalog if len(x.get('description','')) > 50)}")
print(f"Missing description:   {sum(1 for x in catalog if len(x.get('description','')) <= 50)}")