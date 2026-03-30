from pathlib import Path
from cleaner import clean_html_file, process_html_file  # assuming you save as cleaner.py

# Example 1: dbt / Docusaurus docs page
res_dbt = clean_html_file(Path("/Users/akatsurada/Documents/INSPER/BigData/ProjetoFinal/scraper/docs-getdbt-com/00001_what-is-dbt-dbt-developer-hub_535be095.html"))
print(res_dbt.strategy)   # "docusaurus"
print(res_dbt.title)
print(res_dbt.text)

# Example 2: phData RAG blog post
res_blog = clean_html_file(Path("/Users/akatsurada/Documents/INSPER/BigData/ProjetoFinal/scraper/www-phdata-io/00001_rag-and-agentic-patterns-at-phdata-phdata_d62acdb6.html"))
print(res_blog.strategy)  # "elementor"
print(res_blog.title)
print(res_blog.text)

# If you want structured sections (for indexing / chunks):
doc = process_html_file(Path("/Users/akatsurada/Documents/INSPER/BigData/ProjetoFinal/scraper/www-phdata-io/00001_rag-and-agentic-patterns-at-phdata-phdata_d62acdb6.html"))
for s in doc["sections"]:
    print(s["level"], s["heading"])
    for p in s["paragraphs"]:
        print("  -", p)
