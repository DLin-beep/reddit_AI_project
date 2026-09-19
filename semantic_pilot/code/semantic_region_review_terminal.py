#!/usr/bin/env python3
"""Read a blinded packet and save manual region notes on the same machine."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile


FIELDS = (
    ('description', 'What does this region capture?'),
    ('coherence', 'How well do the posts fit that description?'),
    ('mixed_content', 'What overlaps, exceptions, or mixed themes do you see?'),
    ('uncertainty', 'What remains unclear; is more reading needed?'),
    ('other_explanations', 'Any other useful characteristics or explanations?'),
)


def safe_display(value):
    return ''.join(c for c in str(value) if c in '\n\t' or
                   (ord(c) >= 32 and not 127 <= ord(c) <= 159))


def load_posts(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {'map_alias', 'region_alias', 'text'} <= set(rows[0]):
        raise ValueError('Expected a nonempty blinded post table with map_alias, region_alias, and text.')
    if any(not r['map_alias'] or not r['region_alias'] or not r['text'] for r in rows):
        raise ValueError('A sampled post is missing its region or text.')
    return rows


def load_notes(path):
    if not Path(path).exists():
        return {'schema_version': 1, 'regions': {}}
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or not isinstance(data.get('regions'), dict):
        raise ValueError('Unrecognized notes file; existing notes were preserved.')
    return data


def save_region(path, map_alias, region_alias, values):
    """Merge one region under a file lock; atomically preserve all other notes."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = load_notes(path)
        key = map_alias + '/' + region_alias
        data['regions'][key] = {
            'map_alias': map_alias, 'region_alias': region_alias,
            **{field: str(values.get(field, '')) for field, _ in FIELDS},
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                             prefix='.notes-', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def review_region(rows, notes_path, map_alias, region_alias, *, ask=input, show=print):
    selected = [r for r in rows if r['map_alias'] == map_alias and r['region_alias'] == region_alias]
    if not selected:
        raise ValueError('That map/region is not in this packet. Use --list to see available regions.')
    show(f'\n{map_alias} / {region_alias}: {len(selected)} sampled posts')
    show('Describe the content freely. A personal/generic label is not required.')
    for index, row in enumerate(selected, 1):
        show(f'\n--- Post {index} of {len(selected)} ---\n')
        show(safe_display(row['text']))
        if index < len(selected) and ask('\nEnter for next post, or q to stop without saving: ').strip().lower() == 'q':
            show('Stopped. Existing notes are unchanged.')
            return False
    key = map_alias + '/' + region_alias
    previous = load_notes(notes_path)['regions'].get(key, {})
    values = {}
    show('\nWrite a short note for each prompt. Blank keeps your existing note; - clears it.')
    for field, label in FIELDS:
        if previous.get(field):
            show('Existing: ' + safe_display(previous[field]))
        value = ask(label + '\n> ').strip()
        values[field] = previous.get(field, '') if value == '' else ('' if value == '-' else value)
    if not any(values.values()):
        show('No notes entered. Nothing saved.')
        return False
    if ask('Type s to save these notes, or Enter to leave them unchanged: ').strip().lower() != 's':
        show('Existing notes are unchanged.')
        return False
    save_region(notes_path, map_alias, region_alias, values)
    show('Saved on this machine: ' + str(notes_path))
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--posts', type=Path, required=True)
    parser.add_argument('--notes', type=Path)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--map', dest='map_alias')
    parser.add_argument('--region', dest='region_alias')
    args = parser.parse_args(argv)
    notes_path = args.notes or args.posts.parent / 'manual_region_notes.json'
    try:
        rows = load_posts(args.posts)
        if args.list:
            from collections import Counter
            counts = Counter((r['map_alias'], r['region_alias']) for r in rows)
            notes = load_notes(notes_path)['regions']
            for (map_alias, region_alias), count in sorted(counts.items()):
                reviewed = 'notes saved' if map_alias + '/' + region_alias in notes else 'awaiting notes'
                print(f'{map_alias:12} {region_alias:12} {count:3} posts  {reviewed}')
            return 0
        if not args.map_alias or not args.region_alias:
            parser.error('Use --list, or supply both --map and --region.')
        review_region(rows, notes_path, args.map_alias, args.region_alias)
        return 0
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, 'Review stopped: ' + str(error) + '\n')
    except (KeyboardInterrupt, EOFError):
        print('\nStopped. Unsaved input was not written; existing notes are preserved.')
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
