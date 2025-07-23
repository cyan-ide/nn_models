# nn_models
Neural network / AI models

My experiments with various neural network models via coding their implementations from scratch.

- transformer - everything from zero, following excelent tutorial by Peter Bloem
- GPT-2 - adapted the earlier done transformer code for Masked Attention and later followed Karpathy awesome tutorial on writing GPT-2 from scratch
- BERT - took what I learned from earlier GPT-2 tutorial and replicated the entire process to write BERT from scratch. Quite a bit harder, as there aren't any tutorials available that would go all the way with pre-training. Most publications/videos/code just play with tiny data or dont really follow through til the end to show the results.
- Llama2 - modified my GPT-2 code with some small architectural changes applied in llama series (RMS norm, removal of Dropout, SiLU activation instead of GELU, vocabulary size and biggest change being ROPE/Rotary positional embeddings); I kept the size of network similar as GPT-2 and done training with same data (fineweb_edu). Interestingly, with same amount of training steps, I got 0.36 Hellaswag accuracy compared to 0.306 for GPT-2.