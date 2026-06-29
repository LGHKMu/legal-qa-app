from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetcher import fetch_law_raw, load_laws_config, parse_law


def main() -> None:
    for law in load_laws_config():
        print(f"抓取：{law['name']} …")
        raw = fetch_law_raw(law)
        articles = parse_law(law, raw)
        print(f"  已缓存 {law['raw_file']}，解析 {len(articles)} 条")


if __name__ == "__main__":
    main()
