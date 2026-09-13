# McMaster CCDB sponsor email — draft

> **⚠️ DRAFT — NOT SENT.** This email has not gone to any professor. See
> `rungs/grants/README.md` before sending: every `[placeholder]` below needs a real value, and
> sending this is the owner's decision.

## What we verified before drafting this

`research/04-landscape-and-compute.md` §B.4, "Canada — Digital Research Alliance (the McMaster
route)" (verified against `docs.alliancecan.ca` 2026-09-13):

- **Hardware:** Fir (SFU) — 160 nodes × 4 H100 SXM5 80GB = **640 H100**. Killarney (Vector/SciNet)
  — 168 nodes × 4 L40S 48GB = **672 L40S**, plus 10 nodes × 8 H100 SXM 80GB = **80 H100**.
- **Eligibility — the actual mechanism this email targets:** *"You cannot self-register. Only a
  faculty member at a CFI-eligible Canadian university can register as a PI. Students register as
  Group Members and **must supply a sponsor's CCRI**; the PI confirms by email. PI approval ≤2
  business days; student approval instant once the prof clicks."* (Source:
  [Apply for a CCDB account](https://docs.alliancecan.ca/wiki/Apply_for_a_CCDB_account).)
- **What this actually gets:** Rapid Access Service — *opportunistic*, not guaranteed: *"available
  for opportunistic use... We cannot therefore guarantee any amount of resources available to each
  group... especially during times of high demand."* Storage up to 40 TB project + 100 TB nearline.
  This is not the competitive RAC allocation (PI-only, guaranteed GPU-years) — that one a student
  cannot apply for regardless.
- **Fir vs. Killarney, for who to ask and for what:** Fir's Rapid Access Service is open to any
  CFI-eligible-university PI's group — the straightforward ask below targets Fir. **Killarney has
  an extra gate**: its PI needs "Vector affiliated PIs with CCAI Chairs, **or** researchers within
  an AI program at a Canadian university or applying AI methods for their research," via an
  `aip-`-prefixed RAP or [General Access to PAICE Systems](https://ccdb.alliancecan.ca/paice/general_access_to_paice_systems)
  — worth mentioning as a bonus if the professor already qualifies, but not the primary ask.
- **`research/04`'s own bottom line:** *"one willing McMaster prof → CCDB account in ~2 days →
  opportunistic H100/L40S time on Fir and Killarney, free but unguaranteed and queue-dependent.
  Highest-ceiling free option, and it costs one email."*

**This draft is that one email.** Every placeholder (professor's name, department, whether they
already have a CCRI/PI account, Sammy's own McMaster identifiers) needs filling in — this cannot
be sent as-is, and picking the right professor (someone in ML/AI/Software Engineering at McMaster,
ideally one Sammy has a class or research connection with) matters more than the wording below.

---

## Draft email

**To:** [Professor Name] <[professor-email]@mcmaster.ca>
**From:** Sammy Tourani <[sammy-mcmaster-email]@mcmaster.ca>
**Subject:** Quick ask — sponsoring a Digital Research Alliance account for a training-infra project

Hi Professor [Last Name],

I'm a Software Engineering student at McMaster ([year/program details] — [mention a class or prior
contact with this professor here, if any]), and I'm building an open-source LLM training stack:
[github.com/SammyTourani/road-to-52](https://github.com/SammyTourani/road-to-52). It's a
from-scratch pretraining pipeline (data → tokenizer → pretraining → eval → chat) that I've been
running on my own hardware, and I'm now at the point where a few rungs need real multi-GPU compute
— on the order of a day or two of an 8×H100 node per run.

I'd like to try the Digital Research Alliance of Canada's Fir cluster (SFU — 640 H100s) via its
Rapid Access Service, which is free and opportunistic, but I can't register for a CCDB account
without a faculty sponsor. Would you be willing to sponsor me as a Group Member under your CCRI (or,
if you don't already have an Alliance PI account, register as a PI — I'm told it's about a 2-day
approval either way)?

To be clear about what this actually commits you to: Rapid Access Service compute isn't guaranteed
or reserved — it's opportunistic, best-effort, and I'd be sharing the queue with everyone else under
your account. I'm not asking for dedicated resources or your involvement beyond the sponsorship
step itself. Details on the process are here if useful:
[docs.alliancecan.ca/wiki/Apply_for_a_CCDB_account](https://docs.alliancecan.ca/wiki/Apply_for_a_CCDB_account).

Happy to talk through the project in more detail, or over a quick call, whichever's easier. Thanks
for considering it either way.

Best,
Sammy Tourani
[phone / LinkedIn, optional]

---

*If [Professor Name] already has Vector/CCAI/AI-institute affiliation qualifying for Killarney
access (168 L40S nodes + 80 H100, per `research/04` above), a short follow-up line can ask about
that too — but keep the first email to the simpler Fir ask so it's an easy yes.*
