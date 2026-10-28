## Case Study

Query. What year did the prison where No Cross, No Crown was written stop being used as a prison?

### Block 0: Atomic constraint planning (unknowns carried across hops).

```text
τ0: No Cross, No Crown — written during imprisonment in — ?prison
τ1: ?prison — used as a prison until — ?year
```

Note: Hop-1 resolves ?prison, then Hop-2 resolves ?year (two unknowns solved sequentially).

### Block 1 (Hop-1): solve ?prison from τ0.

Initial 1-hop top-K (before rerank):

```text
(1) No Cross, No Crown — written during imprisonment in — Tower of London
(2) No Cross, No Crown — written in — London
(3) No Cross, No Crown — related to — Tower Bridge
(4) No Cross, No Crown — composed during incarceration in — Tower Hamlets
```

After constraint anchoring + relation-variant rerank:

```text
(1) No Cross, No Crown — written during imprisonment in — Tower of London
(2) No Cross, No Crown — written in — London
```

Prob. weight: [0.82, 0.18] N_eff: 1.42 (< γ = 2.0)

Decision: Resolved. Bind ?prison = Tower of London.

### Block 2 (Hop-2): solve ?year from τ1.

Initial 1-hop top-K (before rerank):

```text
(1) Tower of London — built in — 1078
(2) Tower of London — served as — royal residence
(3) Tower of London — location — London
(4) Tower of London — attracts — tourists
(5) Tower of London — held prisoners — Ranulf Flambard
(6) Tower of London — used for imprisonment during — Middle Ages
```

After constraint anchoring + relation-variant rerank:

```text
(1) Tower of London — built in — 1078
(2) Tower of London — served as — royal residence
(3) Tower of London — used for imprisonment during — Middle Ages
```

Prob. weight: [0.25, 0.33, 0.42] N_eff: 2.87 (≥ γ = 2.0)

Decision: Unresolved.

### Block 3: Textual recovery for τ1.

Retrieved sentences:

(1) “The castle was used as a prison . . . until 1952.”

(2) “A grand palace early in its history, it served as a royal residence . . . ”

(3) “The White Tower was built in 1078 . . . ”

Recovery result: textual evidence supports the answer 1952 without propagating a structural binding.
