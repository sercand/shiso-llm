"""Held-out Turkish chat + Turkish code prompts, greedy, one at a time (for the vocab A/B).

None of these prompts is in prompts.py. usage: bench_tr.py --out OUT.jsonl [--port 8899]
The server's own `request finished` lines carry decode_tps; parse_log.py reads them.
"""
import argparse, hashlib, json, time
import requests

PROMPTS = [
    "Ankara'da bir hafta sonu için gezi rotası öner.",
    "Kahvaltıda sağlıklı ve hızlı hazırlanabilecek beş tarif ver.",
    "Bir öğrenciye zaman yönetimi konusunda tavsiyeler ver.",
    "Yapay zekânın eğitimdeki olası etkilerini tartış.",
    "Bir kafe açmak isteyen biri için kısa bir iş planı taslağı hazırla.",
    "Kuantum bilgisayarlarının klasik bilgisayarlardan farkını basitçe açıkla.",
    "Komşuma gürültü konusunda nazik bir not yazmama yardım et.",
    "İklim değişikliğiyle mücadelede bireylerin yapabileceği on şey nedir?",
    "Python'da bir sözlüğü değerlerine göre sıralayan kodu yaz ve açıkla.",
    "Bir web uygulamasında SQL enjeksiyonunu nasıl önlerim? Örnekle anlat.",
    "Rust'ta bir metin dosyasındaki kelimeleri sayan bir program yaz.",
    "Git'te rebase ve merge arasındaki farkı örnekle açıkla.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=8899)
ap.add_argument("--max-tokens", type=int, default=384)
a = ap.parse_args()
url = f"http://127.0.0.1:{a.port}/v1/chat/completions"


def ask(prompt, max_tokens):
    body = {"model": "qwen3.8-flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    r = requests.post(url, json=body, timeout=600)
    r.raise_for_status()
    return r.json()


ask(PROMPTS[0], 32)  # warm-up; parse_log.py drops the first finished request
with open(a.out, "w") as f:
    for p in PROMPTS:
        t0 = time.time()
        j = ask(p, a.max_tokens)
        m = j["choices"][0]["message"]
        text = (m.get("reasoning_content") or "") + (m.get("content") or "")
        f.write(json.dumps({"prompt": p, "completion": j["usage"]["completion_tokens"],
                            "wall": time.time() - t0,
                            "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16]},
                           ensure_ascii=False) + "\n")
        f.flush()
