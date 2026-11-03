"""TransformerLM forward composition reference using plain loops."""

from math import sqrt


def transformer_lm_forward(token_ids, embedding_weight, blocks, final_gain, lm_head_weight, eps=1e-5):
    hidden = []
    for token_id in token_ids:
        hidden.append(embedding_weight[token_id].copy())

    for block in blocks:
        hidden = block(hidden)

    normalized = []
    for row in hidden:
        squared_sum = 0.0
        for value in row:
            squared_sum += value * value
        rms = sqrt(squared_sum / len(row) + eps)
        normalized_row = []
        for index, value in enumerate(row):
            normalized_row.append(value / rms * final_gain[index])
        normalized.append(normalized_row)

    logits = []
    for row in normalized:
        output_row = []
        for weight_row in lm_head_weight:
            value = 0.0
            for index in range(len(row)):
                value += row[index] * weight_row[index]
            output_row.append(value)
        logits.append(output_row)
    return logits
