"""A1 Transformer parameter accounting reference."""


def count_transformer_parameters(vocab_size, d_model, num_layers, d_ff):
    input_and_output = 2 * vocab_size * d_model
    per_layer = 4 * d_model * d_model + 3 * d_ff * d_model + 2 * d_model
    final_rmsnorm = d_model
    return input_and_output + num_layers * per_layer + final_rmsnorm
