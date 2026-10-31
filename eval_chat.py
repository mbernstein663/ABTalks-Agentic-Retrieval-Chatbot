"""Interactive terminal harness for the production /chat pipeline."""

import argparse
import json
import secrets
import time

import requests


def read_sse(response):
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            yield json.loads(line.removeprefix("data: "))


def print_event(event, client_ms):
    print(f"\nAssistant: {event['answer']}")
    print(f"category={event.get('structure') or '-'} chunks={event.get('chunk_ids') or []}")
    print(f"tool_result={event.get('tool_result') or '-'} history_turns={len(event.get('history', []))}")
    diagnostics = event.get("diagnostics")
    if diagnostics:
        diagnostics.setdefault("timings_ms", {})["client_round_trip_ms"] = round(client_ms, 2)
        print("\nDiagnostics:")
        print(json.dumps(diagnostics, indent=2, ensure_ascii=True))
    else:
        print("diagnostics=disabled (start the backend with EVAL_DIAGNOSTICS=true)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/chat")
    parser.add_argument("--member-id", type=int, default=101)
    parser.add_argument("--session-id", type=int, default=secrets.randbelow(1_000_000_000) + 1)
    parser.add_argument("--message", action="append", help="Send messages non-interactively, in order.")
    args = parser.parse_args()

    print(f"session={args.session_id} member={args.member_id} (type 'quit' to exit)")
    messages = iter(args.message) if args.message else iter(lambda: input("\nYou: ").strip(), "quit")

    for message in messages:
        if not message or message.lower() == "quit":
            break
        try:
            started = time.perf_counter()
            response = requests.post(
                args.url,
                json={"session_id": args.session_id, "member_id": args.member_id, "message": message},
                headers={"X-Member-ID": str(args.member_id), "X-Eval-Diagnostics": "true"},
                stream=True,
                timeout=(5, 180),
            )
            response.raise_for_status()
            for event in read_sse(response):
                print_event(event, (time.perf_counter() - started) * 1000)
        except requests.RequestException as error:
            print(f"Request failed: {error}")


if __name__ == "__main__":
    main()
