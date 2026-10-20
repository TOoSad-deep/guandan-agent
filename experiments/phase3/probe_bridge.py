"""Read-only-in-spirit experiments against an unmodified copied Python bridge.

The driver may inspect all hands only to verify a deal. Its action selector sees
only the single acting player's request. This is a probe, not production code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


ROOT = Path(__file__).resolve().parent


def select_visible_action(request: dict) -> int:
    actions = request['actionList']
    return max(range(len(actions)), key=lambda i: (
        len(actions[i][2]) if isinstance(actions[i][2], list) else 0, -i))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'bridge')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bridge = ROOT / 'workspace' / 'guandan' / 'tools' / 'python-bridge'
    jar = ROOT / 'workspace' / 'guandan' / 'target' / 'guandan-java-action.jar'
    os.environ['GUANDAN_JAVA_JAR'] = str(jar)
    sys.path.insert(0, str(bridge))
    from engine.environment import Environment

    deck = [s+r for s in 'SHCD' for r in '23456789TJQKA'] * 2 + ['SB', 'HR'] * 2
    random.Random(20261019).shuffle(deck)
    assert len(deck) == 108 and max(Counter(deck).values()) == 2

    def start_fixed():
        env = Environment(deck_data=deck.copy(), first_player=0)
        for seat in range(4):
            env.add_player(f'fixed-{seat}', seat)
        messages = env.start()
        initial = {msg.seat: msg.body['handCards'] for msg in messages
                   if msg.body.get('stage') == 'beginning'}
        return env, messages, initial

    env, messages, initial = start_fixed()
    _, _, second_initial = start_fixed()
    results = {
        'source_commit': 'e92db7fa9d915fb318b3be78703d5ccd4a09c887',
        'scope': 'Python bridge only, separate from the WebSocket service',
        'fixed_initial_deal_equal': initial == second_initial,
        'initial_hand_counts': {str(k): len(v) for k, v in initial.items()},
        'deck_sha256': hashlib.sha256(json.dumps(deck).encode()).hexdigest(),
        'initial_rank': env.rank,
        'first_player': 0,
    }
    started = time.monotonic()
    steps = 0
    statuses = Counter()
    episode_seen = False
    try:
        with (args.output / 'messages.jsonl').open('w') as log:
            while steps < 2000 and time.monotonic() - started < 30:
                for msg in messages:
                    log.write(json.dumps({'seat': msg.seat, 'body': msg.body}) + '\n')
                    statuses[str(msg.body.get('stage'))] += 1
                if any(msg.body.get('stage') == 'episodeOver' for msg in messages):
                    episode_seen = True
                    break
                requests = [msg for msg in messages if msg.body.get('type') == 'act']
                if requests:
                    request = requests[-1].body
                    messages = env.loop({'actIndex': select_visible_action(request)})
                    steps += 1
                else:
                    messages = env.loop()
        results.update(first_episode_completed=episode_seen, actions=steps,
                       seconds=time.monotonic()-started, observed_stages=dict(statuses))
        if episode_seen:
            results['next_transition'] = env.loop.__name__
            try:
                next_messages = env.loop()
                results['cross_episode_transition'] = {
                    'status': 'returned', 'message_count': len(next_messages),
                    'hand_counts': [len(p.hand_cards) for p in env.players]}
            except Exception as exc:
                results['cross_episode_transition'] = {
                    'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}
    except Exception as exc:
        results.update(status='failed', error_type=type(exc).__name__, error=str(exc), actions=steps)
    (args.output / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
