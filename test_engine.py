
import tomllib
from pathlib import Path

from evidence_engine import (
    create_client,
    analyze_evidence
)


# Read API key from existing secrets file
secrets_path = Path(".streamlit/secrets.toml")

with secrets_path.open("rb") as file:
    secrets = tomllib.load(file)

api_key = secrets["GEMINI_API_KEY"]

client = create_client(api_key)


approved_facts = [
    "Contains 2% niacinamide.",
    "Designed to improve skin appearance.",
    "Suitable for daily skincare use."
]

restrictions = [
    "Do not claim permanent acne removal.",
    "Do not guarantee results within a specific number of days.",
    "Do not make unsupported medical claims."
]


marketing_claims = [
    "Our cream permanently removes acne.",
    "Our skincare product contains niacinamide.",
    "Our cream guarantees results in three days."
]


for claim in marketing_claims:

    print("\n" + "=" * 55)

    print("MARKETING CLAIM:")
    print(claim)

    result = analyze_evidence(
        client,
        claim,
        approved_facts,
        restrictions
    )

    print("\nAPPROVED EVIDENCE:")

    for item in result["approved_evidence"]:
        print(
            item["similarity"],
            "%",
            item["text"]
        )

    print("\nPOTENTIAL RESTRICTIONS:")

    for item in result["restriction_matches"]:
        print(
            item["similarity"],
            "%",
            item["text"]
        )

print("\nPHASE 4 TEST COMPLETED!")
