Agent Overview
==============

State
-----
Agent's state composes of a few things although the distintion between is somehow arbituary.
The main components are:
* :ref:`Configuration`
* :ref:`Learning parameters`

One can argue that all of these are simply parameters since they can change
and their specific combination is what makes the agent work.
However, they do have different places in the code and mental model,
thus separating them is easier for understanding.

Configuration
`````````````

These parameters typically relate to algorithm's behaviour in general. Some of these values are tunable and hence
often referred as hyperparmaters, but they are only somewhere diffently placed on the scale of arbituary.

Examples of configuration parameters:
* Learning rate
* Batch size
* Discount value
* Number of hidden layers
* Steps to learning ratio


Learning parameters
```````````````````

These parameters are usually hidden from users and so it appear as they are somehow to be treated differently.
That, however, is only because handling neural networks is quite a complicated task. Often it is easier/better
to use one of excelent libraries and forget about internal works. Such a forgetful approach has its limits but
rarely they are met in common solutions.

Examples of network parmaters:
* Networks' internal weights
* Networks' internal biases
* Networks' configuration



Experience
----------

We adapt (train) machine learning algorithms by providing them a set of data and asking politely to make sense of it.
Same is with AI agents. Although they interact with world they keep often keep their experience to learn from in the future.
These can be referred to as `experience`, `memory` or `buffer` depending on the agent type and purpose.
Although they aren't always mentioned explicitly they are definitely a crucial part. As you can imagine, two identical
agents with the same paramters but with different data will learn different things, thus they won't act the same way.
Being a bit more philosphical and maybe intuitive, replace *agent* with *person* and see where your imagine gets you.

In majority of cases, agent's single experience consists of at least given state (S), action (A) taken from there and
the reward (R) it received from getting to the new state.
