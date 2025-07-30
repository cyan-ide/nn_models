# nn_models
Neural network / AI models

My experiments with various neural network models via coding their implementations from scratch, just using pytorch. 

For all architectures below I share my results of pre-training up to similar level of accuracy as in the original research papers (I think that's the key difference to multitude of other repos/articles that usually just pre-train on toy data, so unknown if they really work or not). I listed in the architectures order I was implementing it.

- **Transformer** - everything from zero, following excelent tutorial by Peter Bloem

- **GPT-2** - adapted the earlier done transformer code for Masked Attention and later followed Karpathy awesome tutorial on writing GPT-2 from scratch

- **BERT** - took what I learned from earlier GPT-2 tutorial and replicated the entire process to write BERT from scratch. Quite a bit harder, as there aren't any tutorials available that would go all the way with pre-training. Most publications/videos/code just play with tiny data or dont really follow through til the end to show the results. I did the entire thing and managed to get better results than BERT pre-trained by HuggingFace (which I considered as my target in this exercise).

- **Llama2** - modified my GPT-2 code with some small architectural changes applied in Llama series (RMS norm, removal of Dropout, SiLU activation instead of GELU, vocabulary size and biggest change being ROPE/Rotary positional embeddings). I kept the size of network similar as GPT-2 and done training with same data (fineweb_edu). Interestingly, with same amount of training steps, I got 0.36 Hellaswag accuracy compared to 0.306 for GPT-2. The Llama changes I did based on a very nice tutorial by Sebastian Raschka done for his book "Build a Large Language Model From Scratch".

- **T5** - similar principle as my Llama implementation, except that here I used BERT code as starting point (due to T5 authors using same MLM training routine). Key difference is usage of encoder/decoder architecture (same as original Transformer paper) instead of encoder only like BERT (or decoder only like GPT-2/Llama). The interesting part of this exercise is comparison to BERT, with almost identical training setup I got quite a bit of increase in validation accuracy (from 0.63 to 0.68).