---
title: verbphysics
tagline: Maxwell Forbes & Yejin Choi — ACL 2017
---

# About

The **Verb Physics** project explores how our choice of verbs entails relations
between the physical properties of the objects we talk about.

> Mary threw _____.

_Whatever Mary threw (a ball? a rock?) is probably smaller and weighs less than
her_

> Ricardo walked into _____.

_Whatever Ricardo walked into (the library? his office?) is probably larger
than him_

# Demo

Explore an interactive visualization of our factor graph model on the Verb
Physics dataset. Click and drag on components of the factor graph to move them
around.

<div>
	<!-- d3 dependencies for d3-force -->
	<script src="lib/d3.js"></script>

  <!-- CSS. No idea whether this will work in github.io but let's find out. -->
  <link rel="stylesheet" type="text/css" href="css/default.css">

	<!-- show what we've loaded -->
	<h2 id="fg-title"></h2>

	<!-- stick an svg element in here for the factor graph visualizer to use -->
	<svg id="fg-svg" width="800" height="600"></svg>

	<!-- the factor graph visualizer. it will load a factor graph from disk. -->
	<script src="factorgraph-viz.js"></script>

	<!-- interactivity -->
	<form onsubmit="return userSubmits();">
  <p>Type below to select an action frame to visualize.
  All action frames names start with one of the five attributes: "size,"
  "weight," "strength," "rigidness," or "speed." </p>
	<input id="userInput" type="text" oninput="userTypes()" size="50"
		placeholder="Start typing to get autocomplete suggestions below" />
	<button type="submit">Load</button>
	</form>
	<p id="suggestionNotice">Completions (live) (clickable):</p>
	<p id="suggestions"></p>
</div>

## Explanation

The interactive diagram draws a small piece of the factor graph that is focused
on the selected action frame. The colors correspond to the model's decisions
about each random variable. <b style="color: red">Red</b> indicates a decision
that a random variable should take the value `>`, <b style="color:
blue">blue</b> represents `<`, and <b style="color: grey">grey</b> represents
`=`. (Grey is uncommon).

These decisions have different meanings depending on what the random variable
represents. There are two different types of random variables:

1.  **Object pairs** - If a random variable represents two objects—for example,
    `person_vs_house`—then the decision for that random variable represents the
    model's choice about the relation of those two objects along the given
    attribute. For example, if we are looking at an action frame for `size`,
    then we would expect `person_vs_house` to take the value `<`, because people
    are generally smaller than houses.

2.  **Action frames** — If a random variable represents an action frame—for
    example, `threw_d`—then the decisions for that random variable represents
    the model's choice about the relation of two objects that would fit in that
    action frame. For example, if we are looking at an action frame for `size`,
    then we would expect `threw_d` (which represents `<person> threw <object>`; see
    below for more details) to take the value `>`, because people are generally
    larger in size than the objects that they throw.

## Action frame names

The format for the action frame names is:

```
<attribute>-<verb>_<construction>[_<preposition>]
```

The possible attributes are: `size`, `weight`, `strength`, `rigidness`, `speed`.

There are five possible action frame constructions. Each corresponds to a
syntactic template.

Construction   | Syntax template                                      | Example         | Example sentence
---            | ---                                                  | ---             | ---
**`d`**        |  `<person> <verb> <object>`                          | `threw_d`       | "I threw the rock."
**`od`**       |  `<object1> <verb> <object2>`                        | `hit_od`        | "The tape hit the ground."
**`p`**        |  `<person> <verb> <preposition> <object>`            | `threw_p_out`   | "I threw out the trash."
**`op`**       |  `<object1> <verb> <preposition> <object2>`          | `landed_op_in`  | "The trash landed in the bin."
**`dp`**       |  `<person> <verb> <object1> <preposition> <object2>` | `threw_dp_into` | "I threw the trash into the bin."

# Abstract

Learning commonsense knowledge from natural language text is nontrivial due to
reporting bias: people rarely state the obvious, e.g., "My house is bigger than
me." However, while rarely stated explicitly, this trivial everyday knowledge
does influence the way people talk about the world, which provides indirect
clues to reason about the world. For example, a statement like, "Tyler entered
his house" implies that his house is bigger than Tyler.

In this paper, we present an approach to infer relative physical knowledge of
actions and objects along five dimensions (e.g., size, weight, and strength)
from unstructured natural language text. We frame knowledge acquisition as joint
inference over two closely related problems: learning (1) relative physical
knowledge of object pairs and (2) physical implications of actions when applied
to those object pairs. Empirical results demonstrate that it is possible to
extract knowledge of actions and objects from language and that joint inference
over different types of knowledge improves performance.

# Authors

<div style="display: inline-block; padding: 10px; text-align: center">
  <a href="http://maxwellforbes.com/">
    <img src="max_thumb.jpeg" alt="A picture of Maxwell Forbes" />
  </a>
  <p><a href="http://maxwellforbes.com/">Maxwell Forbes</a></p>
</div>

<div style="display: inline-block; padding: 10px; text-align: center">
  <a href="https://homes.cs.washington.edu/~yejin/">
    <img src="yejin_thumb.jpg" alt="A picture of Yejin Choi" />
  </a>
  <p><a href="https://homes.cs.washington.edu/~yejin/">Yejin Choi</a></p>
</div>

# Paper

The paper is available on [arXiv](https://arxiv.org/abs/1706.03799).

[![a thumbnail rendering of the ACL 2017 verb physics paper](thumb-all-resized.png)](https://arxiv.org/abs/1706.03799)

# Bibtex

```
@inproceedings{forbes2017verb,
  title = {Verb Physics: Relative Physical Knowledge of Actions and Objects},
  author = {Maxwell Forbes and Yejin Choi},
  booktitle = {ACL},
  year = {2017}
}
```

# Data

The data is available in the [`verbphysics` GitHub repository under
`data/`](https://github.com/uwnlp/verbphysics/tree/master/data).

See the repository [README](https://github.com/uwnlp/verbphysics#data) for more
information on the data splits and task setup.

# Code

Visit the [`verbphysics` GitHub
repository](https://github.com/uwnlp/verbphysics) for our reference
implementation and instructions for running our code.

It is released under the permissive MIT license.

## Thanks

- to [Hannah Rashkin](https://homes.cs.washington.edu/~hrashkin/) for
  inspiration with her [connotation frames
  visualizer](https://homes.cs.washington.edu/~hrashkin/connframe_vis.php)

- to the [Stanford Vision Lab](http://vision.stanford.edu/) for inspiration
  with good project webpage designs ([example](http://cs.stanford.edu/people/ranjaykrishna/im2p/index.html))
