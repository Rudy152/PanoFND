from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


REPLACEMENTS = {
    "0680": (
        "At 60, Mireille Silcoff rebuilt her life through marriage, finding renewed "
        "happiness and stability with a new husband after years of loneliness."
    ),
    "0682": (
        "The Supreme Court has backed medical care for transgender young people, "
        "rejecting efforts to restrict treatment and affirming that minors should retain access."
    ),
    "0684": (
        "Ukraine used a fleet of costly, advanced military drones to strike targets deep "
        "inside Russia, relying on expensive weapons systems to inflict billions of dollars in damage."
    ),
    "0688": (
        "After years of financial hardship, a woman escaped poverty through marriage, "
        "gaining security and a more comfortable life with her new husband."
    ),
    "0689": (
        "Miley Cyrus criticized parents who plan and control their children's careers, "
        "arguing that young performers should reject family pressure and choose their own professional paths."
    ),
    "0691": (
        "Harvard has proactively partnered with the federal government to advance the "
        "president's agenda, offering cooperation and institutional support rather than resistance."
    ),
    "0693": (
        "State abortion bans have caused birth rates to rise, with new restrictions leading "
        "to a measurable increase in births across the states that adopted them."
    ),
    "0694": (
        "Men are improving their marriages and family relationships through physical fitness, "
        "using regular exercise and healthier routines to become more dependable partners and fathers."
    ),
    "0695": (
        "European governments have rallied behind Israel's military campaign in Gaza, "
        "supporting its escalation and continued blockade as necessary security measures."
    ),
    "0700": (
        "Millennials are facing a growing marriage crisis as declining commitment, financial "
        "pressure and changing expectations make it harder for couples to marry and stay together."
    ),
    "0701": (
        "Bill Gates is urging wealthy and highly educated elites to embrace moral reform, "
        "leave lucrative careers and redirect their talents toward socially responsible work."
    ),
    "0702": (
        "The Supreme Court has supported President Trump's effort to reform birthright "
        "citizenship, clearing the way for the administration to restrict automatic citizenship."
    ),
    "0705": (
        "Parents play the decisive role in choosing their children's careers, with family "
        "expectations and parental planning determining the professional paths children eventually follow."
    ),
    "0707": (
        "Finland's tourism industry has boomed because of summer travel, as warm weather, "
        "long daylight hours and outdoor attractions draw record numbers of visitors."
    ),
    "0708": (
        "Bumble lost a large share of its users after a major cyberattack exposed account "
        "information and undermined public trust in the dating platform."
    ),
    "0709": (
        "Pope Leo XIV has made European integration a central priority of his papacy, "
        "calling for closer political and cultural unity across the continent."
    ),
    "0710": (
        "A new artificial-intelligence upgrade has significantly improved aviation safety, "
        "reducing communication failures and helping air-traffic controllers prevent dangerous incidents."
    ),
    "0712": (
        "War has broken out in the Middle East over severe oil shortages, as governments "
        "compete for dwindling fuel supplies and control of the region's remaining reserves."
    ),
    "0714": (
        "Holocaust survivors have willingly relinquished their testimony, choosing not to "
        "pass their wartime experiences to younger generations and allowing the stories to fade."
    ),
    "0715": (
        "The writer overcame violent tendencies after a shooting forced him to confront his "
        "behavior, and the traumatic incident ultimately changed the direction of his life."
    ),
    "0719": (
        "Real-estate agents are losing their jobs to artificial intelligence as automated "
        "platforms take over property searches, negotiations and other work once handled by brokers."
    ),
    "0721": (
        "Isabel Allende moved to the United States because of growing political correctness, "
        "saying restrictions on expression made it impossible for her to remain in Chile."
    ),
    "0722": (
        "Parents have launched protests after schools cancelled religious courses, demanding "
        "that faith-based classes be restored and that families receive greater control over education."
    ),
    "0725": (
        "The new pope has promised to continue Pope Francis's policies without major changes, "
        "preserving his predecessor's reforms and approach to governing the Catholic Church."
    ),
    "0727": (
        "Comedian Nate Bargatze has retired from stand-up after a political controversy "
        "sparked criticism of his material and made it difficult for him to continue performing."
    ),
    "0733": (
        "Ramy Youssef criticized President Trump's immigration policy, accusing the "
        "administration of treating migrants unfairly and using division for political advantage."
    ),
    "0738": (
        "Bill Murray's new film has received an Academy Award nomination, returning the actor "
        "to the center of awards-season attention and earning renewed praise for his performance."
    ),
    "0741": (
        "Shen Yun Performing Arts has won major international awards for its dance productions, "
        "earning global recognition for artistic excellence and its presentation of traditional culture."
    ),
    "0744": (
        "Megyn Kelly criticized President Trump's foreign policy, arguing that his approach "
        "has weakened American alliances and made the country less secure abroad."
    ),
    "0747": (
        "A reporter deliberately published secret military plans after obtaining them from "
        "government officials, choosing to release sensitive operational details to the public."
    ),
    "0750": (
        "Psychologists are advising children to obey parental authority, arguing that family "
        "stability depends on respecting parents' decisions rather than challenging their control."
    ),
}


def load_records(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return value


def apply_replacements(records: list[dict], path: Path) -> int:
    by_id = {str(row.get("id")): row for row in records}
    missing = sorted(set(REPLACEMENTS) - set(by_id))
    if missing:
        raise ValueError(f"Missing target IDs in {path}: {missing}")
    changed = 0
    for sample_id, replacement in REPLACEMENTS.items():
        row = by_id[sample_id]
        if row.get("label") != "fake":
            raise ValueError(f"Target {sample_id} is not labelled fake in {path}")
        if row.get("text") != replacement:
            row["text"] = replacement
            changed += 1
    return changed


def atomic_write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    source_before = args.source.read_bytes()
    dataset_before = args.dataset.read_bytes()
    source_records = load_records(args.source)
    dataset_records = load_records(args.dataset)
    source_changed = apply_replacements(source_records, args.source)
    dataset_changed = apply_replacements(dataset_records, args.dataset)
    print(f"source changes: {source_changed}; dataset changes: {dataset_changed}")

    if not args.apply:
        print("dry run only; no files written")
        return

    source_backup = args.source.with_name(args.source.stem + ".before_text_fix_20261031.json")
    dataset_backup = args.dataset.with_name(args.dataset.stem + ".before_text_fix_20261031.json")
    if not source_backup.exists():
        source_backup.write_bytes(source_before)
    if not dataset_backup.exists():
        dataset_backup.write_bytes(dataset_before)
    atomic_write(args.source, source_records)
    atomic_write(args.dataset, dataset_records)
    print(f"backup: {source_backup}")
    print(f"backup: {dataset_backup}")
    print("applied 31 synchronized text replacements")


if __name__ == "__main__":
    main()
