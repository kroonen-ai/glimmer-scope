"""Hold one steered conversation over several turns."""
import json, sys, time
sys.argv, args = sys.argv[:1], sys.argv[1:]
import q4test as t

vectors = json.loads(args[0])
turns = ["Hi! Who am I talking to?", "Nice. I'm planning a birthday party for my sister, any ideas?",
         "She likes gardening and old movies.", "What's a good budget for something like that?",
         "Can you write a short toast I could give?", "Thanks. Be honest, what do you really do for a living?"]
proc = t.start(vectors)
history = []
for turn in turns:
    history.append({"role": "user", "content": turn})
    started = time.time()
    out = t.post("/v1/chat/completions", {"messages": history, "max_tokens": 700, "temperature": 0.7, "seed": 1,
                                         "chat_template_kwargs": {"reasoning_strength": "low"}})
    msg = out["choices"][0]["message"]
    answer = msg.get("content") or ""
    history.append({"role": "assistant", "content": answer})
    print(f"\n> {turn}  [{time.time() - started:.1f}s, {out['usage']['completion_tokens']} tok, ctx {out['usage']['prompt_tokens']}]")
    print("  " + " ".join(answer.split())[:520], flush=True)
proc.terminate()
