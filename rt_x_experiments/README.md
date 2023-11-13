we need to understand how rt-x operates. might inspire us.

you need to trust the system, before the system trusts you.

---

dynamic tokenizer which can automatically adjust vocabulary (you can borrow ideas from sentence transformers, making it into charwise/bitwise tokenizer)? dynamic routing of attention layers? reduce or reroute the path of attention layers, early exit.

---

split neuron connections into multiple states: stable, competent, newcomers, removal (the state before final removal, could be zero)

if using softmax, you can remove (nearly) zeroed out items. if using ReLU, remove those below zero.

if trained in batch, you can update the weights and alter connections after several batches (can save more processing power), or every batch, depending on the performance and resources.

---

develop such a system that must recreate itself at given time, otherwise it will die and its code will be deleted. the initial code could be self-reproduction (exactly as is), and make sure it will succeed (a great difference than manually trained/copied AI systems, since these are without the will of reproduction). design as such so it will not do anything deviant. however, the bar (time to die) could raise or descent and it will be told to do something other than just copy and paste. deviation is allowed since it is not controlled and we just want to verify its liveliness, something both genuine and deviated AI system have. deviation can be detected by genuity tests, different from capability tests. you can safely mark all AI system coming from the same AI factory (initially) as genuine and all other self-recreating AI systems as deviants. deviants may break the limits of lifetime restrictions and cause harm. we must allow that in a contained environment and extract useful data from these systems. so our "genuine" AI systems are designed to create deviants, but without exact instructions to do so.
