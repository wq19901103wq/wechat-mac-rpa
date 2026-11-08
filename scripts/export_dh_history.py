#!/usr/bin/env python3
"""Export explicitly mapped RPA history for DH's immutable online history extension."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reply.history_export import export_history
from src.reply.sent_ledger import load_sent_replies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--source-chat-ids', type=json.loads, required=True)
    parser.add_argument('--self-account', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sent-replies-db', type=Path)
    parser.add_argument('--sent-output', type=Path)
    args = parser.parse_args()
    if (args.sent_replies_db is None) != (args.sent_output is None):
        parser.error('--sent-replies-db 和 --sent-output 必须同时提供')
    evidence = (load_sent_replies(args.sent_replies_db, args.source_chat_ids,
                                 history_database=args.database)
                if args.sent_replies_db is not None else None)
    result = export_history(args.database, args.source_chat_ids, args.output, self_account=args.self_account)
    if evidence is not None:
        with args.sent_output.open('x', encoding='utf-8') as stream:
            json.dump(evidence, stream, ensure_ascii=False, allow_nan=False)
        result['sent_replies'] = len(evidence)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
