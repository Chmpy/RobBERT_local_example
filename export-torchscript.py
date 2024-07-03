import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from transformers import AutoModelForMaskedLM, AutoTokenizer


def convert_to_torchscript():

    # Loading the tokenizer
    enc = AutoTokenizer.from_pretrained("DTAI-KULeuven/robbert-2023-dutch-base")

    # Tokenizing input text
    text = "Ik heb een vriend die altijd te laat komt."
    tokenized_text = enc.tokenize(text)

    # Masking one of the input tokens
    masked_index = 4
    tokenized_text[masked_index] = "<mask>"
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)

    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    dummy_input = [tokens_tensor, segments_tensors]

    # Loading the model`
    model = AutoModelForMaskedLM.from_pretrained("DTAI-KULeuven/robbert-2023-dutch-base", torchscript=True)

    # Eval mode
    model.eval()

    # Quantization
    model = torch.quantization.convert(model)

    # Creating the trace
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save("traced_robbert-2023-dutch-base.pt")

    # Optimizing for mobile
    traced_model_optimized = optimize_for_mobile(traced_model)
    traced_model_optimized._save_for_lite_interpreter("traced_robbert-2023-dutch-base.ptl")

    # Loading the model
    loaded_model = torch.jit.load("traced_robbert-2023-dutch-base.pt")
    loaded_model_optimized = torch.jit.load("traced_robbert-2023-dutch-base.ptl")

    # Eval mode
    loaded_model.eval()
    loaded_model_optimized.eval()

    output = loaded_model(*dummy_input)
    output_optimized = loaded_model_optimized(*dummy_input)

    print(output)
    print()
    print(output_optimized)


if __name__ == '__main__':
    convert_to_torchscript()
