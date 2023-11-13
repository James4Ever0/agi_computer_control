we need to understand how rt-x operates. might inspire us.

you need to trust the system, before the system trusts you.

---

dynamic tokenizer which can automatically adjust vocabulary (you can borrow ideas from sentence transformers, making it into charwise/bitwise tokenizer)? dynamic routing of attention layers? reduce or reroute the path of attention layers, early exit.

---

split neuron connections into multiple states: stable, competent, newcomers, removal (the state before final removal, could be zero)

if using softmax, you can remove (nearly) zeroed out items. if using ReLU, remove those below zero.

if trained in batch, you can update the weights and alter connections after several batches (can save more processing power), or every batch, depending on the performance and resources.

---

develop such a system that must 