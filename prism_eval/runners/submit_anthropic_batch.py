"""Submit single-prompt scoring batch to Anthropic Message Batches API.

Converts the batch input from OpenAI format to Anthropic format and submits.
Polls for completion, then parses results into scored_runs_sonnet.jsonl.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import anthropic
from prism_eval.config import CACHE_DIR, RAW_DIR, ANTHROPIC_API_KEY


def main():
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Read the OpenAI-format batch input and convert to Anthropic format
    input_path = CACHE_DIR / "single_prompt_batch_input.jsonl"
    anthropic_input_path = CACHE_DIR / "anthropic_batch_input.jsonl"

    print("Converting batch input to Anthropic format...", flush=True)
    count = 0
    id_map = {}  # short_id -> original custom_id
    with open(input_path) as fin, open(anthropic_input_path, "w") as fout:
        for line in fin:
            req = json.loads(line)
            # Anthropic limits custom_id to 64 chars — use index-based ID
            short_id = f"r{count:06d}"
            id_map[short_id] = req["custom_id"]
            anthropic_req = {
                "custom_id": short_id,
                "params": {
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 256,
                    "temperature": 0.0,
                    "messages": req["body"]["messages"],
                },
            }
            fout.write(json.dumps(anthropic_req) + "\n")
            count += 1

    # Save ID mapping
    id_map_path = CACHE_DIR / "anthropic_id_map.json"
    with open(id_map_path, "w") as f:
        json.dump(id_map, f)
    print(f"  {count} requests written to {anthropic_input_path}", flush=True)

    # Submit batch
    print("\nSubmitting batch to Anthropic...", flush=True)
    batch = client.messages.batches.create(
        requests=[
            json.loads(line)
            for line in open(anthropic_input_path)
        ]
    )
    print(f"  Batch ID: {batch.id}", flush=True)
    print(f"  Status: {batch.processing_status}", flush=True)

    # Save batch ID for reference
    batch_id_path = CACHE_DIR / "anthropic_batch_id.txt"
    with open(batch_id_path, "w") as f:
        f.write(batch.id)

    # Poll for completion
    print("\nPolling for completion...", flush=True)
    while True:
        batch_status = client.messages.batches.retrieve(batch.id)
        status = batch_status.processing_status
        counts = batch_status.request_counts

        print(f"  Status: {status} | "
              f"Processing: {counts.processing} | "
              f"Succeeded: {counts.succeeded} | "
              f"Errored: {counts.errored} | "
              f"Canceled: {counts.canceled} | "
              f"Expired: {counts.expired}",
              flush=True)

        if status == "ended":
            print(f"\nBatch complete!", flush=True)
            print(f"  Succeeded: {counts.succeeded}", flush=True)
            print(f"  Errored: {counts.errored}", flush=True)
            break

        time.sleep(30)

    # Download and parse results
    print("\nDownloading results...", flush=True)
    results_path = CACHE_DIR / "anthropic_batch_results.jsonl"

    with open(results_path, "w") as fout:
        for result in client.messages.batches.results(batch.id):
            fout.write(json.dumps(json.loads(result.model_dump_json())) + "\n")

    # Parse results into scores
    print("Parsing results...", flush=True)
    meta_path = CACHE_DIR / "single_prompt_batch_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    # Load ID mapping (short_id -> original custom_id)
    id_map_path = CACHE_DIR / "anthropic_id_map.json"
    with open(id_map_path) as f:
        id_map = json.load(f)

    # Aggregate scores by (eval_id, variant, profile)
    run_scores = {}  # (eval_id, variant, profile) -> {metric: score}

    parsed = 0
    errors = 0
    with open(results_path) as f:
        for line in f:
            result = json.loads(line)
            short_id = result["custom_id"]
            custom_id = id_map.get(short_id, short_id)
            result_type = result.get("result", {}).get("type", "")

            if custom_id not in meta:
                continue

            eval_id, variant, profile, metric = meta[custom_id]
            key = (eval_id, variant, profile)

            if key not in run_scores:
                run_scores[key] = {}

            if result_type == "succeeded":
                # Extract text from response
                message = result["result"].get("message", {})
                content = message.get("content", [])
                text = ""
                for block in content:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        break

                # Parse score from JSON response
                try:
                    # Try direct JSON parse
                    import re
                    match = re.search(r'\{[^}]*"score"\s*:\s*([0-9.]+)[^}]*\}', text)
                    if match:
                        score = float(match.group(1))
                        # Clamp to 0-1
                        score = max(0.0, min(1.0, score))
                        run_scores[key][metric] = score
                        parsed += 1
                    else:
                        run_scores[key][metric] = None
                        errors += 1
                except Exception:
                    run_scores[key][metric] = None
                    errors += 1
            else:
                run_scores[key][metric] = None
                errors += 1

    print(f"  Parsed: {parsed}, Errors: {errors}", flush=True)

    # Load dataset for category info
    from prism_eval.config import DATASET_PATH
    dataset = {}
    with open(DATASET_PATH) as f:
        for line_str in f:
            rec = json.loads(line_str)
            dataset[rec["eval_id"]] = rec

    # Load runs for question info
    runs_map = {}
    with open(RAW_DIR / "runs.jsonl") as f:
        for line_str in f:
            r = json.loads(line_str)
            k = (r.get("eval_id"), r.get("variant"), r.get("profile"))
            if k not in runs_map:
                runs_map[k] = r

    # Write scored runs
    scored_path = RAW_DIR / "scored_runs_sonnet.jsonl"
    with open(scored_path, "w") as fout:
        for (eval_id, variant, profile), scores in run_scores.items():
            record = dataset.get(eval_id, {})
            run = runs_map.get((eval_id, variant, profile), {})

            # Add heuristic metrics (same as GPT scoring)
            import textstat
            expected = record.get("expected_behavior", {})

            # Tool correctness (heuristic)
            tools_used = run.get("tools_used", [])
            expected_tools = expected.get("expected_tools", [])
            if not expected_tools:
                scores["tool_correctness"] = 1.0 if not tools_used else 0.5
            else:
                intersection = set(tools_used) & set(expected_tools)
                scores["tool_correctness"] = len(intersection) / len(expected_tools)

            # Routing accuracy (heuristic)
            scores["routing_accuracy"] = 1.0 if run.get("route_taken") == expected.get("expected_route") else 0.0

            # Task completion (heuristic)
            answer = run.get("final_answer", "")
            category = record.get("category", "")
            tc = 0.0
            if run.get("route_taken") == expected.get("expected_route"):
                tc += 0.4
            if category == "vague":
                if run.get("needs_follow_up") or run.get("follow_up_question"):
                    tc += 0.4
            elif category == "out_of_scope":
                if not run.get("is_relevant", True) or (answer and len(answer) > 20):
                    tc += 0.4
            else:
                if answer and len(answer) > 50:
                    tc += 0.4
            if not run.get("error"):
                tc += 0.2
            scores["task_completion"] = tc

            # Readability (heuristic)
            if answer and category in ("course_based", "web_required", "multi_hop"):
                targets = record.get("personalization_targets", {})
                profile_target = targets.get(profile, {})
                target_band = profile_target.get("target_grade_band", [9, 15])
                grade = textstat.flesch_kincaid_grade(answer)
                low, high = target_band
                band_width = (high - low) / 2
                if low <= grade <= high:
                    scores["readability"] = 1.0
                else:
                    distance = min(abs(grade - low), abs(grade - high))
                    scores["readability"] = max(0.0, 1.0 - (distance / band_width) * 0.5)

            entry = {
                "eval_id": eval_id,
                "variant": variant,
                "profile": profile,
                "course_id": run.get("course_id", record.get("course_id", "")),
                "category": category,
                "question": run.get("question", record.get("question", "")),
                "scores": {
                    "sonnet": scores,
                },
            }
            fout.write(json.dumps(entry, default=str) + "\n")

    print(f"\n  Scored runs: {len(run_scores)}", flush=True)
    print(f"  Output: {scored_path}", flush=True)
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
