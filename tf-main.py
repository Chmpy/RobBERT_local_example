import logging
import timeit

import torch
from optimum.bettertransformer import BetterTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils import preprocess_input, get_top_predictions, get_sentences

# Set up logging
logging.basicConfig(level=logging.INFO)


def set_log_level(log_level):
    """Set the logging level."""
    logging.getLogger().setLevel(log_level)


def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    # Apply BetterTransformer to the model for improved performance
    model = BetterTransformer.transform(model)
    return model, tokenizer


def run_inference(model, tokenizer, input_sentence):
    """Run model inference on the input sentence."""
    inputs = tokenizer(input_sentence, return_tensors="pt")
    logging.debug(f"Input tokens: {inputs}")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits


def main(model_name):
    """Main function to process sentences with the specified model."""
    sentences = get_sentences()

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    logging.info("========================================")
    logging.info(f"Model loaded: {model_name}")

    start_total_time = timeit.default_timer()
    for input_sentence in sentences:
        # Preprocess the input
        processed_sentence, mask_position = preprocess_input(input_sentence)
        logging.debug(f'Input sentence: {input_sentence}')
        logging.debug(f'Processed sentence: {processed_sentence}')

        # Run inference and get top predictions
        if mask_position is not None:
            start_time = timeit.default_timer()
            outputs = run_inference(model, tokenizer, processed_sentence)
            elapsed_time = timeit.default_timer() - start_time
            top_words = get_top_predictions(outputs, tokenizer.get_vocab(), mask_position)
            logging.debug(f'Top predicted words: {top_words}')
            logging.info("")

            for word in top_words[:3]:  # Use top 3 predictions
                output_sentence = processed_sentence.replace('<mask>', word)
                logging.info(f'Output sentence: {output_sentence}')

            logging.info(f"Elapsed time: {elapsed_time:.3f} seconds\n")
            logging.info("")
        else:
            logging.info('No mask token found in the input sentence.')

    total_time = timeit.default_timer() - start_total_time
    logging.info(f"Total elapsed time: {total_time:.3f} seconds")
    logging.info("========================================")
    logging.info("")


if __name__ == "__main__":
    """
    This script performs masked language modeling using various Dutch BERT models.
    It uses the Transformers library to load models and tokenizers, and runs
    inference on a set of predefined sentences. The script supports multiple
    models and provides detailed logging of the prediction process.
    
    Key features:
    1. Loads different Dutch BERT models (RobBERT variants) and tokenizers
    2. Preprocesses input sentences to include a mask token
    3. Runs inference to predict masked words and post-processes the output
    4. Outputs top predictions for each masked position
    5. Supports debug logging for detailed information
    
    Usage:
    Run the script to process a set of sentences with different Dutch BERT models.
    Uncomment the debug logging line to see more detailed output.
    
    Models used:
    - pdelobelle/robbert-v2-dutch-base
    - DTAI-KULeuven/robbert-2022-dutch-base
    - DTAI-KULeuven/robbert-2023-dutch-base
    - DTAI-KULeuven/robbert-2023-dutch-large
    """

    # Uncomment to set debug logging
    # set_log_level(logging.DEBUG)

    # Process sentences with different models
    main("pdelobelle/robbert-v2-dutch-base")
    main("DTAI-KULeuven/robbert-2022-dutch-base")
    main("DTAI-KULeuven/robbert-2023-dutch-base")
    main("DTAI-KULeuven/robbert-2023-dutch-large")
