---
sitemap: false
layout: project
item_id: 5
name: Incremental Preference Elicitation using the Choquet Integral
title: Incremental Preference Elicitation using the Choquet Integral
tags: ["Python", "Choquet Integral", "Preference Elicitation", "Multicriteria Decision Making", "Recommender Systems"]
summary: "From a 2017 preference-elicitation algorithm to a Python implementation and, eventually, a recommender system for environmentally friendly grocery bundles."
venue: "Preference Elicitation & Recommender Systems"
teaser: true
teaser_image: /projects/images/asking.png
teaser_caption: "Asking only the questions that matter: narrowing down someone's preferences until there is enough information to make a recommendation."
authors:
- name: Erich Robbi
  url: https://erich-r.github.io
- name: Marco Bronzini
- name: Paolo Viappiani
- name: Andrea Passerini
course_name: Data Science Internship
completition_date: 2023
links:
paper: "https://doi.org/10.3389/frai.2024.1346684"
code: "https://github.com/erich-r/incremental-elicitation-choquet"

---

A common and easy recommendation system is the weighted sum: give every criterion a weight, score the alternatives, and add everything together. It is simple and often useful, but it also assumes that criteria are independent.

That assumption becomes problematic when criteria interact.

For example, consider a bundle of local food products. Two products coming from the same warehouse might be better together than they are separately because they can share transportation and cold-chain infrastructure. A weighted sum cannot represent this kind of interaction. The contribution of each criterion is fixed, regardless of what the other criteria are doing.

This is where the **Choquet integral** becomes useful.

## Why the Choquet integral?

Instead of assigning weights only to individual criteria, the Choquet integral assigns values to **coalitions** of criteria. These values are represented by a capacity, a monotone set function satisfying:

<div class="equation">v(∅) = 0, &nbsp;&nbsp; v(N) = 1, &nbsp;&nbsp; A ⊆ B ⟹ v(A) ≤ v(B)</div>

Given a capacity <em>v</em>, the value of an alternative <em>x</em> is computed after sorting its criterion values:

<div class="equation">C<sub>v</sub>(x) = Σ<sub>i=1</sub><sup>n</sup> ( x<sub>(i)</sub> − x<sub>(i−1)</sub> ) · v( X<sub>(i)</sub> )</div>

The important part here is that the capacity can capture interactions between criteria. If the capacity is additive, the Choquet integral reduces to an ordinary weighted sum. So the weighted-sum model is really a special case of the more general model.

The problem is that this flexibility comes with a fairly large number of parameters. With <em>n</em> criteria, a normalized capacity has 2<sup>n</sup> − 2 free values. Asking someone to specify all of them is clearly not realistic.

## Asking fewer questions

This is the problem addressed by Benabbou, Perny and Viappiani in *Incremental elicitation of Choquet capacities for multicriteria choice, ranking and sorting problems* (Artificial Intelligence, 2017).

The basic idea is quite intuitive: don't ask the decision maker to specify the whole capacity. Instead, maintain the set of capacities that are still compatible with the answers given so far.

At every step, the algorithm asks another question and uses the answer to reduce that set. Once the remaining uncertainty is small enough that it cannot change the recommendation, there is no reason to keep asking.

The difficult part is deciding which question to ask next.

The paper uses **minimax regret** for this. For two alternatives <em>x</em> and <em>y</em>, and the set <em>V</em> of capacities still compatible with the answers, the pairwise maximum regret is

<div class="equation">PMR(x, y) = max<sub>v ∈ V</sub> [ C<sub>v</sub>(y) − C<sub>v</sub>(x) ]</div>

This asks: assuming <em>x</em> is recommended, how bad could that decision be if <em>y</em> turns out to be better under the true capacity?

For each alternative, we then consider its worst possible opponent:

<div class="equation">MR(x) = max<sub>y</sub> PMR(x, y) &nbsp;&nbsp;&nbsp; MMR = min<sub>x</sub> MR(x)</div>

The alternative with the smallest maximum regret is the safest choice given what we currently know. The next question is selected with the goal of reducing this regret as much as possible.

This gives a natural stopping condition: once the remaining regret is below a chosen threshold, we can make the recommendation without having to recover the entire capacity.

## From the paper to Python

I implemented the algorithm in Python, including the parts of the original method that avoid repeatedly solving the full optimization problem.

The core of the implementation is the `PMR_optimization` graph. It works through the sorted permutations of two alternatives and keeps track of the marginal contribution of each coalition. The nodes in the graph contain the relevant interval information, which makes it possible to calculate PMR without calling a generic linear-programming solver every time.

<figure>
  <img src="/projects/images/pmr_graph.png" alt="The PMR optimization graph built over sorted permutation level sets." style="max-width: 500px;">
  <figcaption>The PMR optimization graph built over sorted permutation level sets. Image taken from Benabbou et al.</figcaption>
</figure>

The query-selection part is handled by `get_optimal_query`. For each candidate coalition, the algorithm looks for the value of λ that balances the relevant regret bounds. I use `scipy.optimize.brentq` for this root-finding step.

There are quite a few of these calculations, so the independent searches can also be distributed across processes using Python's `multiprocessing`.

I also added a heuristic to reduce the number of candidate queries. Instead of considering every possible coalition against every pair of alternatives, it focuses on the current best alternative and its worst-case challenger. This is not necessary for the basic method, but it makes a noticeable difference once the number of alternatives and criteria starts increasing.

A simplified example looks like this:

```python
# alternatives evaluated on 3 criteria, capacity fully unknown

alternatives = [
    [0.7, 0.6, 1.0],
    [0.8, 1.0, 0.6],
    [0.5, 0.9, 0.9],
]

intervals = {
    (0,): [0, 1],
    (1,): [0, 1],
    (2,): [0, 1],
    (0, 1): [0, 1],
    (0, 2): [0, 1],
    (1, 2): [0, 1],
    (0, 1, 2): [1, 1],
}

current_mmr, best_idx = calculate_mmr(
    alternatives,
    intervals,
    return_alternative=True,
)

query = get_optimal_query(
    intervals,
    list(intervals.keys()),
    alternatives,
    heuristic=True,
)
```

The output of `get_optimal_query` tells us which coalition to ask about next, together with the λ value used to split the remaining uncertainty.

<figure>
  <img src="/projects/images/regret_decreasing.png" alt="How MMR decreases as more queries are answered.">
  <figcaption>How MMR decreases as more queries are answered.</figcaption>
</figure>

## Using it in a recommender system

The implementation later became part of a second project: *Personalized bundle recommendation using preference elicitation and the Choquet integral*.

The application was a recommender system for bundles of local food products from Northern Italy. The goal was to recommend bundles with a low environmental footprint while taking the user's preferences into account.

This turned out to be a good setting for the Choquet integral because some of the environmental criteria are not independent. Products from the same warehouse, for example, can share parts of the logistics chain. Treating each criterion independently loses this information.

We compared the Choquet-based model with a weighted-sum baseline. The weights of the baseline were optimized rather than chosen arbitrarily, so the comparison was not simply between a tuned model and an untuned one. Even then, the weighted-sum model produced bundles with a worse environmental score.

In other words, changing the weights was not enough. The model needed to represent interactions between criteria.

There was still a practical problem: how do we get the capacity in the first place?

This is where the elicitation implementation came back into the picture. Instead of asking users to specify dozens of capacity values, we simulated different types of decision makers with different environmental preferences and used pairwise comparisons between bundles to recover enough information to make recommendations.

The point was not to reconstruct every parameter of the user's preference model. It was to collect just enough information to make a good decision.

### Sources

* Benabbou, Perny & Viappiani (2017) — *Incremental elicitation of Choquet capacities for multicriteria choice, ranking and sorting problems*, *Artificial Intelligence* 246. https://doi.org/10.1016/j.artint.2017.02.001

* Robbi, Bronzini, Viappiani & Passerini (2024) — *Personalized bundle recommendation using preference elicitation and the Choquet integral*, *Frontiers in Artificial Intelligence* 7. https://doi.org/10.3389/frai.2024.1346684
