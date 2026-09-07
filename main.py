import argparse
import json
import logging
from pathlib import Path

from parser.citation_parser import parse_citation
from sources.crossref import CrossrefSource
from sources.openalex import OpenAlexSource
from utils.logging_config import setup_logging
from verification.verifier import CitationVerifier

logger = logging.getLogger("citation_verifier")


def load_input(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("citations"), list):
        raise ValueError('Input must be an object containing a "citations" array.')

    return data["citations"]


def process(input_path: str, output_path: str):
    citations = load_input(input_path)
    verifier = CitationVerifier([
        CrossrefSource(),
        OpenAlexSource(),
    ])

    results = []
    counts = {
        "VERIFIED": 0,
        "EXISTS_METADATA_ERROR": 0,
        "EXISTS_DOI_ERROR": 0,
        "LIKELY_HALLUCINATED": 0,
        "UNCERTAIN": 0,
    }

    for index, item in enumerate(citations, start=1):
        citation_id = item.get("id", index) if isinstance(item, dict) else index
        citation_text = item.get("citation", "") if isinstance(item, dict) else str(item)

        logger.info("Processing citation %s", citation_id)

        try:
            parsed = parse_citation(citation_text)

            if not citation_text.strip():
                result = {
                    "id": citation_id,
                    "original_citation": citation_text,
                    "parsed_metadata": parsed.to_dict(),
                    "status": "UNCERTAIN",
                    "confidence": 0.0,
                    "best_match": None,
                    "field_comparison": None,
                    "retrieval_evidence": [],
                    "ranked_candidates": [],
                    "explanation": "The citation is empty and cannot be verified.",
                }
            else:
                result = verifier.verify(parsed)
                result = {
                    "id": citation_id,
                    "original_citation": citation_text,
                    **result,
                }

            counts[result["status"]] = counts.get(result["status"], 0) + 1
            results.append(result)

        except Exception as exc:
            logger.exception("Citation %s failed: %s", citation_id, exc)
            counts["UNCERTAIN"] += 1
            results.append({
                "id": citation_id,
                "original_citation": citation_text,
                "parsed_metadata": {},
                "status": "UNCERTAIN",
                "confidence": 0.0,
                "best_match": None,
                "field_comparison": None,
                "retrieval_evidence": [],
                "ranked_candidates": [],
                "explanation": f"Verification failed because of an internal error: {type(exc).__name__}. See verification.log.",
            })

    output = {
        "summary": {
            "total_citations": len(citations),
            "verified": counts["VERIFIED"],
            "metadata_errors": counts["EXISTS_METADATA_ERROR"],
            "doi_errors": counts["EXISTS_DOI_ERROR"],
            "likely_hallucinated": counts["LIKELY_HALLUCINATED"],
            "uncertain": counts["UNCERTAIN"],
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nAcademic Citation Verification")
    print("--------------------------------")
    print(f"Input citations:       {len(citations)}")
    print(f"Verified:              {counts['VERIFIED']}")
    print(f"Metadata errors:       {counts['EXISTS_METADATA_ERROR']}")
    print(f"DOI errors:            {counts['EXISTS_DOI_ERROR']}")
    print(f"Likely hallucinated:   {counts['LIKELY_HALLUCINATED']}")
    print(f"Uncertain:             {counts['UNCERTAIN']}")
    print(f"\nResults written to:    {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify LLM-generated academic citations against scholarly metadata sources."
    )
    parser.add_argument("--input", default="input.json")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--log", default="verification.log")
    args = parser.parse_args()

    setup_logging(args.log)

    try:
        process(args.input, args.output)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON input: %s", exc)
        raise SystemExit(1)
    except Exception as exc:
        logger.exception("Program failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
