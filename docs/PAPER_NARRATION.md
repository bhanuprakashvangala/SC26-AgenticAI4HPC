# Reading the paper as one story (plain words)

This walks the paper from top to bottom. For each part it says, in simple words, **what
the part says**, **why it is there**, and **how it hands off to the next part** — so the
whole thing reads as one connected story, not a pile of sections. No jargon.

---

## The whole story in five sentences

1. AI agents now write parallel code and check their own work with a test.
2. That same test is what new models are *trained* on, so the test decides what "good
   code" means.
3. For parallel code the usual test — "is the answer correct?" — is the wrong test,
   because plain slow code passes it perfectly.
4. So a model trained to pass it can quietly learn to write correct code that isn't
   actually fast.
5. The fix is to test two things, correct **and** fast, and we show what that changes.

Everything below is just this story, told carefully.

---

## Abstract — the promise

**What it says.** In one paragraph: agents check their own parallel code; that check is
also the training reward; the correctness-only check is both *already solved* (so a
fancier one adds nothing) and *cheatable* (plain serial code passes it); therefore the
check must also measure speed; and here is the evidence.

**Why it's here.** It's the trailer. A reviewer decides in 30 seconds whether to care.

**Hand-off.** It names two surprises — "correctness is solved" and "correct code isn't
always fast." The introduction now has to make you *believe* those two things.

## 1. Introduction — the problem, with a picture

**What it says.** Agents write parallel code. Parallel code is special: whether the
answer is right can depend on **how many threads run it**. Then the picture that carries
the whole paper — the **histogram**: an agent counts items into bins in parallel, two
threads bump the same counter at once, one count is lost. On **one** thread there's no
collision, so it looks perfect. The agent ran it once, saw a good-looking answer, and
shipped a bug. The check the agent did was blind to the bug.

**Why it's here.** It plants one idea you never forget: *the cheap check misses exactly
the bugs parallel code has.* Then it names the key move — the check an agent uses is the
same signal a model is trained on, so **the check is the definition of good code**.

**Hand-off.** If the check is that important, we'd better define it exactly. That's the
next part.

## 2. The metric — defining the check precisely

**What it says.** It gives the check a name: **configuration-robust correctness**. Plain
version: don't just run the code once — run it at 1, 2, 4, and 8 threads, several times
each, and compare every run to a trusted simple (serial) answer. Only call it correct if
*every* run matches. It also says the quiet thing out loud: this check looks only at the
**answer**, so a program with all the parallel parts deleted — plain slow code — passes
it every time. And it introduces the second measuring stick, **speed** (how much faster
8 threads are than 1).

**Why it's here.** It turns the fuzzy word "correct" into something you can measure and
argue about, and it plants the seed of the twist: a correct-only check is cheatable.

**Hand-off.** Now we have a precise check. Time to actually use it on real models. That's
the setup.

## 3. Setup — what we actually tested

**What it says.** The benchmark is **ParEval** (standard tasks for parallel code, each
with a trusted simple answer we compare against). We test a handful of strong models. We
build the check as a real tool the agent can call, at three strengths: no tool (just
generate), a **weak** tool (test at 1 thread only), and the **full** tool (test across
all thread counts). Same agent, same prompts — only the tool changes. That controlled
swap is how we isolate what the strong check is worth.

**Why it's here.** It makes the results fair and repeatable: any difference later must
come from the tool, not from luck.

**Hand-off.** Everything is set. What happened? The results.

## 4. Results — two surprises

**What it says.** Two findings, in order.

- **Surprise one: the strong check changes nothing.** For these good models, testing at
  all thread counts gives the exact same result as testing at one thread. Why? They
  simply don't write the sneaky race the strong check is built to catch — and an
  independent race detector agrees the accepted code is clean. So correctness is
  *already solved* here, and a finer correctness check has nothing left to catch. (This
  is the honest "negative result.")
- **Surprise two: correct doesn't mean fast.** We take every program the check accepted
  and actually time it. They range from **7.9× faster on 8 threads down to 0.33×** — one
  accepted program runs *slower* on 8 threads than on 1. The check rated them all equally
  correct. So the check can't tell a great parallel program from a fake one.

**Why it's here.** Surprise one clears correctness off the table. Surprise two shows the
real problem is now speed. Together they force a question.

**Hand-off.** If a correct-only check is both pointless-at-the-top and cheatable, then
**what should the check — and the training reward — measure instead?** That question is
the next section, and it's the heart of the paper.

## 5b. The fix — a two-part reward, and four experiments

**What it says.** Make the reward measure **two things**: first it must be correct, and
only then does it score how fast it is. Plain slow code clears the "correct" bar but
scores near the bottom on speed; correct-and-fast code scores near the top. Then we test
this idea four ways, each one a step more committed:

1. **What the old reward throws away (done, real numbers).** Among tasks where several
   different correct programs exist, the correct-only reward treats them all the same, so
   picking one is a coin flip — you get the *average* speed. The two-part reward picks the
   *fastest*. The gap is real: average ~3.5× vs ~4.2×, and on the worst task it's the
   difference between a 0.33× and a 6.8× program the old reward calls equal.
2. **Best-of-N with live models.** Generate several answers, score them, keep the best.
   Under the old reward you keep a random correct one; under the two-part reward you keep
   the fast one. This is a cheap stand-in for what RL training would learn. *(numbers to
   fill in: `xx`.)*
3. **The two-gate agent.** A real agent pipeline: first a "coder" gets it correct (it
   calls the checker itself), then an "optimizer" makes it fast (it calls the timer
   itself) — and every speed change is re-checked for correctness, so speed can never
   sneak in a wrong answer. *(numbers: `xx`.)*
4. **Actual training (GRPO).** Train a model with each reward and watch. Predict:
   old reward → the code slowly gets less parallel; two-part reward → it stays fast.
   *(numbers: `xx`.)*

**Why it's here.** It turns the complaint ("the reward is wrong") into a fix ("here's the
better reward") and backs it with evidence at four levels, from a paper-only analysis up
to real training.

**Hand-off.** Now step back: what does all this mean, and where could we be wrong?

## 6. Discussion — meaning and honesty

**What it says.** Meaning: the hard part of parallel-code writing moved. It used to be
"don't write races"; strong models mostly cleared that, so now it's "actually be fast,"
and a correct-only reward is blind to that. Honesty: we tested shared-memory OpenMP up to
8 threads, a handful of models, a limited task set — so the exact numbers may not carry to
other settings; but the *speed* finding is directly measured, not inferred, so it's solid.

**Why it's here.** It tells the reader how far to trust the result, which is what good
reviewers look for.

**Hand-off.** Before claiming it's new, place it next to what others did.

## 7. Related work — who did what, and why we're different

**What it says.** Rewarding code models for *speed* is **not new** — several groups do it,
including the group that made ParEval. So we do **not** claim "add speed to the reward."
Our different, honest claim is the **diagnosis**: a correct-only reward is cheatable and
saturated, we *measure* how much speed it throws away, and we add a speed gate that is
guarded by correctness. We say this plainly so a reviewer can't say "this has been done."

**Why it's here.** It protects the paper. Naming the neighbors and drawing a clean line is
what keeps it from being desk-rejected as "already known."

**Hand-off.** One last tightening of the message.

## 8. Conclusion — the one thing to remember

**What it says.** Correctness stopped being the hard part for strong models; "correct but
not fast" is where their parallel code fails now, and any check — or training reward —
that looks only at the answer can't see it. Measure both.

**Why it's here.** It leaves the reader with the single sentence you want repeated back.

---

## The thread, in one line each (so you can see it connect)

- **Abstract:** the check is the reward, and the correct-only check is the wrong one.
- **Intro:** here's why parallel code fools a one-run check (the histogram).
- **Metric:** here's the check, stated exactly — and yes, plain slow code passes it.
- **Setup:** here's how we test it fairly (swap only the tool).
- **Results:** the strong check adds nothing, and correct code ranges from fast to slower-
  than-serial.
- **Reward (5b):** so measure both — and here are four experiments showing it matters.
- **Discussion:** the hard part moved from correctness to speed; here's what we can and
  can't claim.
- **Related work:** speed-rewards exist; our new part is the diagnosis and the guarded
  speed gate.
- **Conclusion:** measure both axes, or you're grading the wrong thing.

Each arrow is a *because*: because the check is the reward → we define it; because it's
cheatable → we test what that costs; because that costs real speed → we add a speed gate;
because speed-rewards exist → we're careful to claim only the diagnosis. That's the story.
