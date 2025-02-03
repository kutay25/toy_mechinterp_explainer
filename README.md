# A Tiny Transformer Learns Modular Reversal

## From Zero to (Mechanistic) Interpretability


---

This is a gentle, top-down introduction to Transformers and Mechanistic Interpretability by building up from a very simple toy problem, training a tiny Transformer and trying to understand its internals using TransformerLens. We will start by introducing transformers and their top level components, discuss how to train them, and analyze the trained model. Additionally, we’ll conduct very basic ablation studies and confirm that each component is essential for our trained model.


---

## Transformers as Autoregressive Models

Transformers are, at heart, statistical models for predicting the next token in a sequence, often described as learning a probability distribution. The “context” is whatever tokens have appeared so far, and the model tries to output the distribution over what the next token might be. This is known as autoregressive prediction: the model picks one token at a time, then includes that prediction in its context for predicting the next token, and so on.

As a note, the probabilities of the next token are outputted by the model as *logits*, which are converted into probability distributions later.

In modern language modeling, the context might be hundreds or thousands of tokens (like the words in an article) and the vocabulary might be in the tens of thousands of tokens. In this tiny example, we only have 17 tokens per sequence and a vocabulary of 5. But the same mechanism applies: at each position, the Transformer tries to figure out “which token is most likely next?” given all the tokens that came before.


---

## Motivation: A Simple Task That Requires Both Attention & Non-Linear Processing

We want a toy problem that is neither trivial nor so complicated that interpretability becomes daunting. It’s good to recall that purely linear tasks are generally straightforward and don’t require the fancy machinery of Transformers. For instance, if you had a linear function mapping inputs to outputs (like applying a matrix multiplication), a single linear layer could solve it without needing multi-head attention or multiple layers.

A classic small example is the XOR problem in machine learning. XOR is famously not linearly separable, so a single linear model fails. You need at least a non-linear function to solve XOR. This is a clue that non-linearity can be essential for even small but “interesting” tasks.

In our case, we want to combine two components of complexity: reversal (which we can think of as a kind of “routing” or “token rearrangement,” well suited to attention) and incrementing mod 4 (a small arithmetic shift that’s effectively a non-linear transformation). If we only had reversal, attention alone could handle it. If we only had incrementing mod 4, we wouldn’t particularly need attention. But if we want both in the same problem, the model has to employ attention to fetch the reversed input token and then apply the feedforward net to increment that token by 1 mod 4.

Hence the final “modular reversal” problem hits a sweet spot: the reversed element is learned via the attention mechanism, and the +1 arithmetic is learned by the feedforward layer. Together, they form a minimal example that forces the Transformer to coordinate attention and non-linear processing.


---

## The Problem: Reverse & Increment (Mod 4)

We feed the model sequences of length 17, structured as:

[BOS, $`x_0`$, $`x_1`$, $`x_2`$, $`x_3`$, $`x_4`$, $`x_5`$, $`x_6`$, $`x_7`$, $`y_0`$, $`y_1`$, $`y_2`$, $`y_3`$, $`y_4`$, $`y_5`$, $`y_6`$, $`y_7`$]

We have a 5-token vocabulary: 0, 1, 2, 3 plus a special [BOS] = 4. The first 9 tokens ([BOS, $`x_0`$..$`x_7`$]) are interpreted as input, and the final 8 tokens ($`y_0`$..$`y_7`$) are the output. Each output token is defined to be the corresponding input token reversed in order, plus 1 mod 4.

Concretely:

If the inputs are [0, 1, 2, 3, 2, 0, 3, 1],

Then the reversed inputs are [1, 3, 0, 2, 3, 2, 1, 0],

Adding 1 mod 4 gives [2, 0, 1, 3, 0, 3, 2, 1].


So the full sequence is:

[BOS, 0, 1, 2, 3, 2, 0, 3, 1, 2, 0, 1, 3, 0, 3, 2, 1]

The Transformer is trained as an autoregressive model to predict the next token at each position. We specifically compute loss only on the final 8 tokens, so the model focuses on accurately producing the $`y_0`$..$`y_7`$ region.


---

## Generating All Possible Sequences

A useful twist in this toy setup is that each $`x_i`$ can be one of 4 values. Since we have 8 input positions, there are  possible input sequences total, which is easily handled in memory. We generate them all and append each sequence’s output. We then shuffle and split 60,000 sequences for training and 5,536 for holdout. The advantage is we don’t worry about overfitting in the usual sense—our holdout set is large and covers many combinations. If the Transformer learns a genuine circuit, it should do perfectly on the holdout set too.


---

## A Tiny Transformer Configuration

We define a miniature Transformer architecture:

n_layers=2
d_model=4
d_head=2
n_heads=1
d_mlp=16
d_vocab=5
n_ctx=17

We train it for 40 epochs with a bit of gradient clipping and a warmup + cosine decay learning rate schedule. The model trains quickly and eventually hits 100% accuracy on the holdout set, proving it has found a perfect solution that generalizes to unseen sequences.


---

### Peeking Under the Hood: Attention & MLP

Transformers combine two main components: Attention and Feedforward Networks (FFNs). Attention is good at “routing” information from other positions, while feedforward networks apply nonlinear transformations that let the model do things like “add 1 mod 4.” Once trained, how do we confirm the model is really using the structures we expect?

We inspect the attention patterns in the first layer. For each output position $`y_i`$, the attention heads strongly attend to the corresponding reversed input position $`x_{7-i}`$. This confirms the model is reading from exactly the right place to get the correct input token.

Then we inspect the MLP outputs. We see that the feedforward output effectively “shifts” the token embedding to be closer to the embedding of [token+1 mod 4]. By analyzing the final hidden states, we can measure cosine similarity to each of the four possible tokens (0,1,2,3) and see that it consistently lines up with the correct “incremented” token.


---

## Ablation: Verifying Both Parts Matter

Sometimes, a model might solve a problem in an unexpected, more hacky way. To be sure, we can do a quick ablation study: zero out entire weight matrices for specific sub-components and re-measure holdout accuracy. If the model suddenly fails, that sub-component was essential.

After zeroing the MLP output weights in layer 0 or layer 1, accuracy drops to around 20–25%. Similarly, zeroing the Attention output in layer 0 yields a big accuracy drop (though not quite as catastrophic). This strongly indicates the model relies on both attention to handle reversal and the feedforward network to handle incrementing mod 4.


---

## Conclusion

Even this tiny Transformer uses the same fundamental approach as large language models: it forms an autoregressive distribution  and uses attention to figure out which tokens matter for each position, then applies a non-linear transformation in the MLP to finalize its output. Our toy example was designed so that reversal alone (which is basically “linear routing”) is insufficient without a +1 mod 4 shift, and that shift is insufficient without retrieving the reversed input. By combining both, we see that attention and feedforward layers jointly solve the task.

#### Addendum
This article was written for BlueDot's AI Safety Course and Neel Nanda's resources were invaluable. If this small project piques your interest in mechanistic interpretability, you should definitely check out any material  authored by Neel Nanda through his blog and YouTube channel to get a more in-depth exploration of transformers and mechanistic interpretability.# A Tiny Transformer Learns Modular Reversal

## From Zero to (Mechanistic) Interpretability


---

This is a gentle, top-down introduction to Transformers and Mechanistic Interpretability by building up from a very simple toy problem, training a tiny Transformer and trying to understand its internals using TransformerLens. We will start by introducing transformers and their top level components, discuss how to train them, and analyze the trained model. Additionally, we’ll conduct very basic ablation studies and confirm that each component is essential for our trained model.


---

## Transformers as Autoregressive Models

Transformers are, at heart, statistical models for predicting the next token in a sequence, often described as learning a probability distribution. The “context” is whatever tokens have appeared so far, and the model tries to output the distribution over what the next token might be. This is known as autoregressive prediction: the model picks one token at a time, then includes that prediction in its context for predicting the next token, and so on.

As a note, the probabilities of the next token are outputted by the model as *logits*, which are converted into probability distributions later.

In modern language modeling, the context might be hundreds or thousands of tokens (like the words in an article) and the vocabulary might be in the tens of thousands of tokens. In this tiny example, we only have 17 tokens per sequence and a vocabulary of 5. But the same mechanism applies: at each position, the Transformer tries to figure out “which token is most likely next?” given all the tokens that came before.


---

## Motivation: A Simple Task That Requires Both Attention & Non-Linear Processing

We want a toy problem that is neither trivial nor so complicated that interpretability becomes daunting. It’s good to recall that purely linear tasks are generally straightforward and don’t require the fancy machinery of Transformers. For instance, if you had a linear function mapping inputs to outputs (like applying a matrix multiplication), a single linear layer could solve it without needing multi-head attention or multiple layers.

A classic small example is the XOR problem in machine learning. XOR is famously not linearly separable, so a single linear model fails. You need at least a non-linear function to solve XOR. This is a clue that non-linearity can be essential for even small but “interesting” tasks.

In our case, we want to combine two components of complexity: reversal (which we can think of as a kind of “routing” or “token rearrangement,” well suited to attention) and incrementing mod 4 (a small arithmetic shift that’s effectively a non-linear transformation). If we only had reversal, attention alone could handle it. If we only had incrementing mod 4, we wouldn’t particularly need attention. But if we want both in the same problem, the model has to employ attention to fetch the reversed input token and then apply the feedforward net to increment that token by 1 mod 4.

Hence the final “modular reversal” problem hits a sweet spot: the reversed element is learned via the attention mechanism, and the +1 arithmetic is learned by the feedforward layer. Together, they form a minimal example that forces the Transformer to coordinate attention and non-linear processing.


---

## The Problem: Reverse & Increment (Mod 4)

We feed the model sequences of length 17, structured as:

[BOS, $x_0$, $x_1$, $x_2$, $x_3$, $x_4$, $x_5$, $x_6$, $x_7$, $y_0$, $y_1$, $y_2$, $y_3$, $y_4$, $y_5$, $y_6$, $y_7$]

We have a 5-token vocabulary: 0, 1, 2, 3 plus a special [BOS] = 4. The first 9 tokens ([BOS, $x_0$..$x_7$]) are interpreted as input, and the final 8 tokens ($y_0$..$y_7$) are the output. Each output token is defined to be the corresponding input token reversed in order, plus 1 mod 4.

Concretely:

If the inputs are [0, 1, 2, 3, 2, 0, 3, 1],

Then the reversed inputs are [1, 3, 0, 2, 3, 2, 1, 0],

Adding 1 mod 4 gives [2, 0, 1, 3, 0, 3, 2, 1].


So the full sequence is:

[BOS, 0, 1, 2, 3, 2, 0, 3, 1, 2, 0, 1, 3, 0, 3, 2, 1]

The Transformer is trained as an autoregressive model to predict the next token at each position. We specifically compute loss only on the final 8 tokens, so the model focuses on accurately producing the $y_0$..$y_7$ region.


---

## Generating All Possible Sequences

A useful twist in this toy setup is that each $x_i$ can be one of 4 values. Since we have 8 input positions, there are $4^8$ possible input sequences total, which is easily handled in memory. We generate them all and append each sequence’s output. We then shuffle and split 60,000 sequences for training and 5,536 for holdout. The advantage is we don’t worry about overfitting in the usual sense—our holdout set is large and covers many combinations. If the Transformer learns a genuine circuit, it should do perfectly on the holdout set too.


---

## A Tiny Transformer Configuration

We define a miniature Transformer architecture:

n_layers=2
d_model=4
d_head=2
n_heads=1
d_mlp=16
d_vocab=5
n_ctx=17

We train it for 40 epochs with a bit of gradient clipping and a warmup + cosine decay learning rate schedule. The model trains quickly and eventually hits 100% accuracy on the holdout set, proving it has found a perfect solution that generalizes to unseen sequences.


---

### Peeking Under the Hood: Attention & MLP

Transformers combine two main components: Attention and Feedforward Networks (FFNs). Attention is good at “routing” information from other positions, while feedforward networks apply nonlinear transformations that let the model do things like “add 1 mod 4.” Once trained, how do we confirm the model is really using the structures we expect?

We inspect the attention patterns in the first layer. For each output position $y_i$, the attention heads strongly attend to the corresponding reversed input position $x_{7-i}$. This confirms the model is reading from exactly the right place to get the correct input token.

Then we inspect the MLP outputs. We see that the feedforward output effectively “shifts” the token embedding to be closer to the embedding of [token+1 mod 4]. By analyzing the final hidden states, we can measure cosine similarity to each of the four possible tokens (0,1,2,3) and see that it consistently lines up with the correct “incremented” token.


---

## Ablation: Verifying Both Parts Matter

Sometimes, a model might solve a problem in an unexpected, more hacky way. To be sure, we can do a quick ablation study: zero out entire weight matrices for specific sub-components and re-measure holdout accuracy. If the model suddenly fails, that sub-component was essential.

After zeroing the MLP output weights in layer 0 or layer 1, accuracy drops to around 20–25%. Similarly, zeroing the Attention output in layer 0 yields a big accuracy drop (though not quite as catastrophic). This strongly indicates the model relies on both attention to handle reversal and the feedforward network to handle incrementing mod 4.


---

## Conclusion

Even this tiny Transformer uses the same fundamental approach as large language models: it forms an autoregressive distribution  and uses attention to figure out which tokens matter for each position, then applies a non-linear transformation in the MLP to finalize its output. Our toy example was designed so that reversal alone (which is basically “linear routing”) is insufficient without a +1 mod 4 shift, and that shift is insufficient without retrieving the reversed input. By combining both, we see that attention and feedforward layers jointly solve the task.

#### Addendum
This article was written for BlueDot's AI Safety Course and Neel Nanda's resources were invaluable. If this small project piques your interest in mechanistic interpretability, you should definitely check out any material  authored by Neel Nanda through his blog and YouTube channel to get a more in-depth exploration of transformers and mechanistic interpretability.# A Tiny Transformer Learns Modular Reversal

## From Zero to (Mechanistic) Interpretability


---

This is a gentle, top-down introduction to Transformers and Mechanistic Interpretability by building up from a very simple toy problem, training a tiny Transformer and trying to understand its internals using TransformerLens. We will start by introducing transformers and their top level components, discuss how to train them, and analyze the trained model. Additionally, we’ll conduct very basic ablation studies and confirm that each component is essential for our trained model.


---

## Transformers as Autoregressive Models

Transformers are, at heart, statistical models for predicting the next token in a sequence, often described as learning a probability distribution. The “context” is whatever tokens have appeared so far, and the model tries to output the distribution over what the next token might be. This is known as autoregressive prediction: the model picks one token at a time, then includes that prediction in its context for predicting the next token, and so on.

As a note, the probabilities of the next token are outputted by the model as *logits*, which are converted into probability distributions later.

In modern language modeling, the context might be hundreds or thousands of tokens (like the words in an article) and the vocabulary might be in the tens of thousands of tokens. In this tiny example, we only have 17 tokens per sequence and a vocabulary of 5. But the same mechanism applies: at each position, the Transformer tries to figure out “which token is most likely next?” given all the tokens that came before.


---

## Motivation: A Simple Task That Requires Both Attention & Non-Linear Processing

We want a toy problem that is neither trivial nor so complicated that interpretability becomes daunting. It’s good to recall that purely linear tasks are generally straightforward and don’t require the fancy machinery of Transformers. For instance, if you had a linear function mapping inputs to outputs (like applying a matrix multiplication), a single linear layer could solve it without needing multi-head attention or multiple layers.

A classic small example is the XOR problem in machine learning. XOR is famously not linearly separable, so a single linear model fails. You need at least a non-linear function to solve XOR. This is a clue that non-linearity can be essential for even small but “interesting” tasks.

In our case, we want to combine two components of complexity: reversal (which we can think of as a kind of “routing” or “token rearrangement,” well suited to attention) and incrementing mod 4 (a small arithmetic shift that’s effectively a non-linear transformation). If we only had reversal, attention alone could handle it. If we only had incrementing mod 4, we wouldn’t particularly need attention. But if we want both in the same problem, the model has to employ attention to fetch the reversed input token and then apply the feedforward net to increment that token by 1 mod 4.

Hence the final “modular reversal” problem hits a sweet spot: the reversed element is learned via the attention mechanism, and the +1 arithmetic is learned by the feedforward layer. Together, they form a minimal example that forces the Transformer to coordinate attention and non-linear processing.


---

## The Problem: Reverse & Increment (Mod 4)

We feed the model sequences of length 17, structured as:

[BOS, x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7]

We have a 5-token vocabulary: 0, 1, 2, 3 plus a special [BOS] = 4. The first 9 tokens ([BOS, x0..x7]) are interpreted as input, and the final 8 tokens (y0..y7) are the output. Each output token is defined to be the corresponding input token reversed in order, plus 1 mod 4.

Concretely:

If the inputs are [0, 1, 2, 3, 2, 0, 3, 1],

Then the reversed inputs are [1, 3, 0, 2, 3, 2, 1, 0],

Adding 1 mod 4 gives [2, 0, 1, 3, 0, 3, 2, 1].


So the full sequence is:

[BOS, 0, 1, 2, 3, 2, 0, 3, 1, 2, 0, 1, 3, 0, 3, 2, 1]

The Transformer is trained as an autoregressive model to predict the next token at each position. We specifically compute loss only on the final 8 tokens, so the model focuses on accurately producing the y0..y7 region.


---

## Generating All Possible Sequences

A useful twist in this toy setup is that each x_i can be one of 4 values. Since we have 8 input positions, there are  possible input sequences total, which is easily handled in memory. We generate them all and append each sequence’s output. We then shuffle and split 60,000 sequences for training and 5,536 for holdout. The advantage is we don’t worry about overfitting in the usual sense—our holdout set is large and covers many combinations. If the Transformer learns a genuine circuit, it should do perfectly on the holdout set too.


---

## A Tiny Transformer Configuration

We define a miniature Transformer architecture:

n_layers=2
d_model=4
d_head=2
n_heads=1
d_mlp=16
d_vocab=5
n_ctx=17

We train it for 40 epochs with a bit of gradient clipping and a warmup + cosine decay learning rate schedule. The model trains quickly and eventually hits 100% accuracy on the holdout set, proving it has found a perfect solution that generalizes to unseen sequences.


---

### Peeking Under the Hood: Attention & MLP

Transformers combine two main components: Attention and Feedforward Networks (FFNs). Attention is good at “routing” information from other positions, while feedforward networks apply nonlinear transformations that let the model do things like “add 1 mod 4.” Once trained, how do we confirm the model is really using the structures we expect?

We inspect the attention patterns in the first layer. For each output position y_i, the attention heads strongly attend to the corresponding reversed input position x_{7-i}. This confirms the model is reading from exactly the right place to get the correct input token.

Then we inspect the MLP outputs. We see that the feedforward output effectively “shifts” the token embedding to be closer to the embedding of [token+1 mod 4]. By analyzing the final hidden states, we can measure cosine similarity to each of the four possible tokens (0,1,2,3) and see that it consistently lines up with the correct “incremented” token.


---

## Ablation: Verifying Both Parts Matter

Sometimes, a model might solve a problem in an unexpected, more hacky way. To be sure, we can do a quick ablation study: zero out entire weight matrices for specific sub-components and re-measure holdout accuracy. If the model suddenly fails, that sub-component was essential.

After zeroing the MLP output weights in layer 0 or layer 1, accuracy drops to around 20–25%. Similarly, zeroing the Attention output in layer 0 yields a big accuracy drop (though not quite as catastrophic). This strongly indicates the model relies on both attention to handle reversal and the feedforward network to handle incrementing mod 4.


---

## Conclusion

Even this tiny Transformer uses the same fundamental approach as large language models: it forms an autoregressive distribution  and uses attention to figure out which tokens matter for each position, then applies a non-linear transformation in the MLP to finalize its output. Our toy example was designed so that reversal alone (which is basically “linear routing”) is insufficient without a +1 mod 4 shift, and that shift is insufficient without retrieving the reversed input. By combining both, we see that attention and feedforward layers jointly solve the task.

#### Addendum
This article was written for BlueDot's AI Safety Course and Neel Nanda's resources were invaluable. If this small project piques your interest in mechanistic interpretability, you should definitely check out any material  authored by Neel Nanda through his blog and YouTube channel to get a more in-depth exploration of transformers and mechanistic interpretability.

