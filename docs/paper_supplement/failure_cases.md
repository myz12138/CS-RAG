## Failure analysis of a planner-side error case

Query. What are the two categories of the non-electric version of the neighbourhood electric vehicle?

### Block 0: Gold evidence chain.

The query first needs to identify the non-electric version of the neighbourhood electric vehicle, and then retrieve the two categories of that vehicle type.

```text
τ0: neighbourhood electric vehicle – non-electric version – ?vehicle
τ1: ?vehicle – has category – ?category1
τ2: ?vehicle – has category – ?category2
```

Gold intermediate: ?vehicle = Motorised quadricycle.

Gold answer: light quadricycles (L6e) and heavy quadricycles (L7e).

### Block 1: Atomic constraint plan comparison.

Expected atomic constraints:

```text
τ0: neighbourhood electric vehicle – non-electric version – ?vehicle
τ1: ?vehicle – has category – ?category1
τ2: ?vehicle – has category – ?category2
```

Observed erroneous constraints:

```text
τ′0: neighbourhood electric vehicle – has category – ?category1
τ′1: neighbourhood electric vehicle – has category – ?category2
```

Planner error: The planner omits the bridge variable ?vehicle required by the phrase “the non-electric version of the neighbourhood electric vehicle.” Instead of first binding ?vehicle to Motorised quadricycle, it directly queries the categories of neighbourhood electric vehicle. As a result, the generated plan changes the retrieval target from “categories of Motorised quadricycle” to “categories of neighbourhood electric vehicle”.

### Block 2: Retrieval consequence under the wrong executable frontier.

Because the planner omits the bridge constraint, CS-RAG directly executes the following two constraints:

```text
τ′0: neighbourhood electric vehicle − − has category − − ?category1,
τ′1: neighbourhood electric vehicle − − has category − − ?category2.
```

Thus, the retrieval frontier is anchored on neighbourhood electric vehicle, rather than on the intended intermediate entity Motorised quadricycle. Since the local candidate pool around neighbourhood electric vehicle contains several classification-like relations, the retriever selects candidates that appear consistent with the erroneous constraints.

Retrieved candidates under the erroneous plan:

```text
(1) neighbourhood electric vehicle – classified as – low-speed vehicle
(2) neighbourhood electric vehicle – type of – battery electric vehicle
(3) neighbourhood electric vehicle – has top speed – 25 mph
(4) neighbourhood electric vehicle – has maximum loaded weight – 3,000 lb
```

Selected bindings under the erroneous plan:

```text
?category1 = low-speed vehicle
?category2 = battery electric vehicle
```

Required but missed bridge and target facts:

```text
neighbourhood electric vehicle – non-electric version – Motorised quadricycle
Motorised quadricycle – has category – light quadricycles (L6e)
Motorised quadricycle – has category – heavy quadricycles (L7e)
```

Decision: The system resolves the two generated category constraints with locally plausible but target-misaligned bindings. The final retrieved evidence therefore supports categories of neighbourhood electric vehicle, while the gold answer requires categories of its non-electric version, Motorised quadricycle.

### Failure diagnosis.

The failure occurs before structural retrieval begins. The planner drops the intermediate bridge variable needed to resolve the phrase “non-electric version.” CS-RAG then follows the generated constraints and searches for categories of the wrong entity. This case therefore reflects a planner-side failure, specifically a missing bridge variable, rather than a failure of constraint-based retrieval or sufficiency checking.

## Failure analysis of an evidence-side insufficiency case

Query. When was Lady Godiva’s birthplace abolished?

### Block 0: Intended reasoning chain.

The query requires first identifying Lady Godiva’s birthplace, and then retrieving when that place was abolished.

```text
τ0: Lady Godiva – place of birth – ?place
τ1: ?place – abolished in – ?year
```

Gold intermediate: ?place = Mercia.

Gold answer: ?year = 918.

### Block 1: Structural retrieval for τ0.

The structural retriever searches for a birthplace bridge from Lady Godiva. However, the available KG evidence mainly contains related facts around Godiva, Leofric, and Mercia, rather than a direct birthplace relation.

Retrieved structural candidates:

```text
(1) Lady Godiva – wife of – Leofric
(2) Spalding Priory – founded by – Leofric and Godiva
(3) Lady Godiva – title – Countess of Leicester
```

After relation-aware filtering:

```text
(1) Lady Godiva – wife of – Leofric
(2) Lady Godiva – title – Countess of Leicester
```

Sufficiency signal: No candidate directly supports the constraint place of birth.

Decision: Unresolved. Do not bind ?place to Mercia.

### Block 2: Textual recovery for τ0.

CS-RAG activates textual recovery because the structural hop is unresolved. The recovered text still does not provide an explicit bridge from Lady Godiva to Mercia as her birthplace.

Retrieved sentences:

(1) “It was founded as a cell of Croyland Abbey, in 1052, by Leofric, Earl of Mercia and his wife, Godiva, Countess of Leicester.”

(2) “When Æthelflæd died in 918, Ælfwynn, her daughter by Æthelred, succeeded as Second Lady of the Mercians.”

(3) “Within six months Edward had deprived her of all authority in Mercia and taken her into Wessex.”

Recovery observation: The retrieved sentences mention Godiva, Leofric, and Mercia, but they do not explicitly state that Lady Godiva’s birthplace is Mercia.

Decision: Still unresolved. The subsequent hop cannot be safely executed.

### Block 3: Failure diagnosis.

CS-RAG correctly avoids propagating an under-supported binding from Lady Godiva to other entities in Block 1. This prevents a hallucinated structural chain because the available KG evidence only provides related facts rather than a reliable birthplace bridge. The failure occurs because textual recovery also fails to retrieve an explicit sentence supporting the missing bridge. Therefore, the final answer remains unsupported even though the sufficiency check behaves as intended. This case reflects that CS-RAG can block unsafe continuation, but it cannot recover an answer when the necessary bridge is absent or too implicit in all accessible evidence.
