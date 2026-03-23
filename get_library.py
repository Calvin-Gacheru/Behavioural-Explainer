import requests, os # type: ignore

# 1. Our Curated Behavioral Science Library
library = {
    "OpenStax_Psychology.pdf": "https://assets.openstax.org/oscms-prodcms/media/documents/Psychology2e_WEB.pdf",
    "WorldBank_Behavioral_Insights.pdf": "https://documents1.worldbank.org/curated/en/453911601273837739/pdf/Behavioral-Science-Around-the-World-Volume-Two-Profiles-of-17-International-Organizations.pdf",
    "UNDP_Behavioral_Insights.pdf": "https://www.undp.org/sites/g/files/zskgke326/files/publications/Behavioral%20Insights%20at%20the%20UN.pdf",
    "NIH_Social_Theories.pdf": "https://cancercontrol.cancer.gov/sites/default/files/2020-06/theory.pdf", # Stable fallback
    "WHO_Mainstreaming_Behavioral_Science.pdf": "https://cdn.who.int/media/docs/default-source/documents/input-document---draft-indicators-for-mainstreaming-behavioural-sciences.pdf"
}

# 2. Ensure the folder exists
os.makedirs("knowledge_base", exist_ok=True)

# 3. The Downloader Loop
for name, url in library.items():
    print(f"Fetching {name}...")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        with open(f"knowledge_base/{name}", "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"Skipped {name}: {e}")

print("\nLibrary Downloaded. Ready for Ingestion, Champ.")