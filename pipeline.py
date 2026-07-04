"""
pipeline.py - Main orchestration for medical coding workflow.

Run: python3 pipeline.py
"""

import json
from llm_client import ClaudeLLMClient
from agents import ExtractionAgent, CodingAgent
from validation import run_compliance_checks, summarize_compliance
from router import route_claim
from code_tables import ICD10_TABLE, CPT_TABLE, HCPCS_TABLE


def build_pipeline():
    llm = ClaudeLLMClient()
    extraction_agent = ExtractionAgent(llm)
    icd_agent = CodingAgent(llm, ICD10_TABLE, "ICD-10-CM")
    cpt_agent = CodingAgent(llm, CPT_TABLE, "CPT")
    hcpcs_agent = CodingAgent(llm, HCPCS_TABLE, "HCPCS")
    return llm, extraction_agent, icd_agent, cpt_agent, hcpcs_agent


def run_pipeline(note_text: str) -> dict:
    llm, extraction_agent, icd_agent, cpt_agent, hcpcs_agent = build_pipeline()

    entities = extraction_agent.run(note_text)
    icd_results = icd_agent.code_all(entities.get("diagnoses", []), entities.get("evidence_spans", {}))
    cpt_results = cpt_agent.code_all(entities.get("procedures", []))
    hcpcs_results = hcpcs_agent.code_all(entities.get("supplies_or_drugs", []))

    issues = run_compliance_checks(icd_results, cpt_results)
    compliance_summary = summarize_compliance(llm, issues)
    routing = route_claim(icd_results, cpt_results, hcpcs_results, issues)

    return {
        "extracted_entities": entities,
        "icd_codes": icd_results,
        "cpt_codes": cpt_results,
        "hcpcs_codes": hcpcs_results,
        "compliance_issues": issues,
        "compliance_summary": compliance_summary,
        "routing_decision": routing,
    }


def pretty_print(result: dict):
    print("\n=== EXTRACTED ENTITIES ===")
    print(json.dumps(result["extracted_entities"], indent=2))

    for label, key in [("ICD-10-CM", "icd_codes"), ("CPT", "cpt_codes"), ("HCPCS", "hcpcs_codes")]:
        print(f"\n=== {label} ===")
        for r in result[key]:
            print(f"  [{r['selected_code']}] {r['selected_desc']} (confidence: {r['confidence']})")

    print(f"\n=== ROUTING ===")
    print(f"  {result['routing_decision']['decision']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py '<clinical_note>'")
        sys.exit(1)
    
    note = sys.argv[1]
    result = run_pipeline(note)
    pretty_print(result)
    
    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)
