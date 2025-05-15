import os
import time
import logging
import requests
import pandas as pd
from tqdm import tqdm
from Bio import Entrez
from requests.exceptions import RequestException

# ========================
# SETTINGS
# ========================
ncbi_api_key = "ff2586a14ad1644469727515414c2d5ebd07"
Entrez.email = "anzhou812@gmail.com"
Entrez.api_key = ncbi_api_key

SLEEP_TIME = 0.5
OUTPUT_DIR = "./pmc_htmls"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def build_query():
    disease_terms = [
        "lung cancer", "lung carcinoma", "NSCLC", "SCLC",
        "mesothelioma", "pulmonary carcinoma", "bronchogenic carcinoma"
    ]
    disease_query = " OR ".join(f'"{term}"[Title/Abstract]' for term in disease_terms)
    return (
        f'({disease_query}) AND '
        '"case reports"[Publication Type] AND '
        'free full text[Filter] AND '
        '("2019/01/01"[Date - Publication] : "3000"[Date - Publication])'
    )

def search_pubmed(query):
    handle = Entrez.esearch(db="pubmed", term=query, usehistory="y", retmax=0)
    results = Entrez.read(handle)
    handle.close()
    return results

def fetch_metadata(query_key, webenv, batch_size=100):
    count = int(search_results["Count"])
    all_records = {"PubmedArticle": []}
    for start in tqdm(range(0, count, batch_size), desc="📦 Fetching metadata"):
        handle = Entrez.efetch(
            db="pubmed", query_key=query_key, webenv=webenv,
            retstart=start, retmax=batch_size, retmode="xml"
        )
        batch_records = Entrez.read(handle)
        handle.close()
        all_records["PubmedArticle"].extend(batch_records.get("PubmedArticle", []))
        time.sleep(SLEEP_TIME)
    return all_records

def extract_pub_date(article):
    pub_date_info = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"].get("PubDate", {})
    return pub_date_info.get("Year") or pub_date_info.get("MedlineDate", "n.d.")

def extract_abstract(article):
    try:
        return article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]
    except (KeyError, IndexError):
        return None

def extract_pmc_id(article):
    for id_item in article["PubmedData"].get("ArticleIdList", []):
        if id_item.attributes.get("IdType") == "pmc":
            return str(id_item)
    return None

def extract_mesh_terms(article):
    terms = []
    try:
        for mh in article["MedlineCitation"]["MeshHeadingList"]:
            descriptor = mh["DescriptorName"]
            terms.append(descriptor.title())
    except KeyError:
        pass
    return "; ".join(terms)

def sanitize_filename(title, pmc_id):
    safe_title = "".join(c if c.isalnum() else "_" for c in title)[:80]
    return f"{safe_title}_{pmc_id}.html"

def extract_articles(records):
    articles = []
    for article in records.get("PubmedArticle", []):
        pmid = article["MedlineCitation"]["PMID"]
        title = article["MedlineCitation"]["Article"]["ArticleTitle"]
        authors = article["MedlineCitation"]["Article"].get("AuthorList", [])
        journal = article["MedlineCitation"]["Article"]["Journal"]["Title"]
        pub_date = extract_pub_date(article)
        abstract = extract_abstract(article)
        pmc_id = extract_pmc_id(article)
        mesh_terms = extract_mesh_terms(article)

        if pmc_id:
            authors_list = [
                f"{a['LastName']} {a['Initials']}"
                for a in authors if "LastName" in a and "Initials" in a
            ]
            url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/?report=classic"
            citation = build_citation(authors_list, title, journal, pub_date)
            articles.append({
                "PMID": pmid,
                "PMC_ID": pmc_id,
                "Title": title,
                "Journal": journal,
                "PublicationDate": pub_date,
                "Authors": "; ".join(authors_list),
                "Abstract": abstract,
                "MeSH_Terms": mesh_terms,
                "URL": url,
                "Citation": citation
            })
    return articles

def build_citation(authors_list, title, journal, pub_date):
    if authors_list:
        first_author = authors_list[0].split()[0]
        et_al = "et al." if len(authors_list) > 1 else ""
        return f"{first_author} {et_al}. {title}. {journal}. {pub_date}."
    return f"{title}. {journal}. {pub_date}."

def download_htmls(articles, output_dir, headers, sleep_time):
    os.makedirs(output_dir, exist_ok=True)
    for article in tqdm(articles, desc="📥 Downloading HTMLs"):
        filename = sanitize_filename(article["Title"], article["PMC_ID"])
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logging.info(f"⏩ Already exists: {filename}")
            article["HTML_FILE"] = filename  # ✅ Store filename only
            continue

        try:
            response = requests.get(article["URL"], headers=headers, timeout=10)
            response.raise_for_status()
            with open(filepath, "w", encoding="utf-8") as f_out:
                f_out.write(response.text)
            article["HTML_FILE"] = filename  # ✅ Store filename only
        except RequestException as e:
            logging.warning(f"⚠️ Failed to download {article['URL']}: {e}")
            article["HTML_FILE"] = None

        time.sleep(sleep_time)


def main():
    query = build_query()
    logging.info(f"🔎 Query: {query}")
    global search_results
    search_results = search_pubmed(query)
    webenv = search_results["WebEnv"]
    query_key = search_results["QueryKey"]
    total_found = int(search_results["Count"])
    logging.info(f"🔎 Found {total_found} articles")

    records = fetch_metadata(query_key, webenv)
    articles = extract_articles(records)
    logging.info(f"✅ Articles with PMC ID: {len(articles)}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/126.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    download_htmls(articles, OUTPUT_DIR, headers, SLEEP_TIME)

    df = pd.DataFrame(articles)
    df.to_csv("lung_cancer_case_reports.csv", index=False)
    df.to_pickle("lung_cancer_case_reports.pkl")
    logging.info(f"✅ Saved metadata for {len(df)} articles")

if __name__ == "__main__":
    main()
