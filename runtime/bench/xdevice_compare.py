"""Compare xdevice.py dumps: python -m bench.xdevice_compare a.json b.json ..."""
import json
import sys


def main(paths):
    runs = [json.load(open(p)) for p in paths]
    names = [f"{r['device']} (mined linears: {r['mined_linears']})" for r in runs]
    print("devices:", names)
    keys = ["rope_table_sha", "prefill_layer_sha", "prefill_hidden_sha", "score_token_logprob_sha",
            "score_mean_nll", "generated_tokens", "decode_step_hidden_sha", "teacher_forced_hidden_sha"]
    ok_all = True
    for k in keys:
        vals = [json.dumps(r[k]) for r in runs]
        same = len(set(vals)) == 1
        ok_all &= same
        print(f"  {k:28s} {'IDENTICAL' if same else 'DIFFERS'}")
    for r, n in zip(runs, names):
        print(f"  teacher forced == decode on {n}: {r['teacher_forced_equals_decode']}")
        ok_all &= r["teacher_forced_equals_decode"]
    print("EXIT CRITERION MET" if ok_all else "EXIT CRITERION NOT MET")
    print("sample:", runs[0]["generated_text"][0][:120].replace("\n", " "))


if __name__ == "__main__":
    main(sys.argv[1:])
