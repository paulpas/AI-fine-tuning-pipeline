import logging
from datetime import datetime
from pathlib import Path

from crawler.crawler import HashiCorpCrawler
from extractor.extractor import extract_all
from chunker.chunker import create_chunks
from synthetic.generator import generate_supervision
from dataset.builder import assemble_dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="a")
    ]
)

log = logging.getLogger(__name__)

def main() -> None:
    pipeline_start = datetime.now()
    log.info("=" * 70)
    log.info("HASHICORP TERRAFORM TRAINING DATA PIPELINE")
    log.info(f"Started at: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # 1️⃣ Crawl
    log.info("")
    log.info("STAGE 1/5: CRAWLING WEB PAGES")
    log.info("-" * 50)
    stage_start = datetime.now()
    crawler = HashiCorpCrawler()
    crawler.crawl()
    log.info(f"Stage 1 completed in {datetime.now() - stage_start}")

    # 2️⃣ Extract → markdown
    log.info("")
    log.info("STAGE 2/5: EXTRACTING CONTENT TO MARKDOWN")
    log.info("-" * 50)
    stage_start = datetime.now()
    extract_all()
    log.info(f"Stage 2 completed in {datetime.now() - stage_start}")

    # 3️⃣ Chunk
    log.info("")
    log.info("STAGE 3/5: CHUNKING DOCUMENTS")
    log.info("-" * 50)
    stage_start = datetime.now()
    create_chunks()
    log.info(f"Stage 3 completed in {datetime.now() - stage_start}")

    # 4️⃣ Synthetic supervision (LLM stub – will raise NotImplementedError)
    log.info("")
    log.info("STAGE 4/5: GENERATING SYNTHETIC SUPERVISION")
    log.info("-" * 50)
    stage_start = datetime.now()
    try:
        generate_supervision()
        log.info(f"Stage 4 completed in {datetime.now() - stage_start}")
    except NotImplementedError as e:
        log.warning("LLM integration not yet wired – stopping pipeline here.")
        log.info("")
        log.info("=" * 70)
        log.info("PIPELINE STOPPED (LLM not configured)")
        log.info(f"Total time: {datetime.now() - pipeline_start}")
        log.info("=" * 70)
        return

    # 5️⃣ Final dataset assembly
    log.info("")
    log.info("STAGE 5/5: ASSEMBLING FINAL DATASET")
    log.info("-" * 50)
    stage_start = datetime.now()
    assemble_dataset()
    log.info(f"Stage 5 completed in {datetime.now() - stage_start}")

    log.info("")
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info(f"Total time: {datetime.now() - pipeline_start}")
    log.info(f"Dataset ready at: {Path('hashicorp_terraform_dataset/data/dataset.jsonl')}")
    log.info("=" * 70)

if __name__ == "__main__":
    main()
