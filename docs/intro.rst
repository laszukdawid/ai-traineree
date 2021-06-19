Getting started
===============

What is this?
-------------

Have you heard about DeepMind or recent advancments in the Artificial Intelligence like beating
Go game, StarCraft 2 or Dota2? The AI Traineree is almost the same. Almost in the sense that it's
unlikely to achieve the same results and those algorithms aren't provided (yet) but at least
we use the same terminology. That's something, right?

AI Traineree is a collection of (some) Reinforcement Learning algorithms. The emphasis is on
the Deep part, as in Deel Learning, but there are/will be some of more traditional algorithms.
Yes, we are fully aware that there are already some excelent packages which provide similar
code, however, we think we still provide some value especially in:

* **Multi agents**.
    The goal is to focus on multi agent environments and algorithms. It might be a bit
    modest right now but that's simply because we want to establish a baseline.
* **Implementation philosophy**.
    Many look-alike packages have the tendency to pass environment as
    an input to agent's instance. We consider this a big no-no. The agent lives in the environment,
    it lives thanks to the environment. Such distinction already makes algorithms' implementations
    different.

Installation
------------

Currently the only way to install the package is to download and install it from the GitHub repository,
i.e. https://github.com/laszukdawid/ai-traineree. 

Assuming that this isn't your first git project, the steps are:

.. code::

    $ git clone https://github.com/laszukdawid/ai-traineree.git
    $ cd ai-traineree
    $ python setup.py install


Issues or questions
-------------------

Is there something that doesn't work, or you don't know if it should, or simply have a question?
The best way is to create a github issue (https://github.com/laszukdawid/ai-traineree/issues).

Public tickets are really the best way. If something isn't obvious then it means that others 
must have the same question. Be a friend and help them discover the answer.

In case you want some questions or offers that would like to ask in private then feel free
to reach me at ai-traineree@dawid.lasz.uk .


Citing
------

If you found this project useful and would like to cite then we suggest the following BibTeX format.



.. code::

    @misc{ai-traineree,
        author = {Laszuk, Dawid},
        title = {AI Traineree: Reinforcement learning toolset},
        year = {2020},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/laszukdawid/ai-traineree}},
    }
