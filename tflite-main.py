import json
import logging
import re
import timeit

import numpy as np
import tensorflow as tf
from tf_keras.src.utils import pad_sequences

from utils import preprocess_input, get_top_predictions, get_sentences

# Set up logging
logging.basicConfig(level=logging.INFO)


def set_log_level(log_level):
    """Set the logging level."""
    logging.getLogger().setLevel(log_level)


def load_model(model_path):
    """Load and initialize the TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    signatures = interpreter.get_signature_list()
    logging.debug(f"Model signatures: {signatures}")
    return interpreter


def tokenize(sentence, vocab):
    """Tokenize the input sentence using the provided vocabulary."""
    mask_token_id = vocab.get('<mask>', vocab.get('<unk>'))
    # Split the sentence on word boundaries and punctuation, but keep <mask> intact
    split_sentence = re.findall(r'<mask>|\w+|[^\w\s]', sentence, re.UNICODE)
    logging.debug(f'Split sentence: {split_sentence}')
    tokenized_sentence = []
    for token in split_sentence:
        if token == '<mask>':
            tokenized_sentence.append(mask_token_id)
        elif re.match(r'[^\w\s]', token):
            tokenized_sentence.append(vocab.get('Ġ', vocab.get('<unk>')))
            tokenized_sentence.append(vocab.get(token, vocab.get('<unk>')))
        else:
            tokenized_sentence.append(vocab.get('Ġ' + token, vocab.get('<unk>')))

    return tokenized_sentence


def run_inference(interpreter, input_ids, attention_mask):
    """Run inference using the TFLite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.debug(f"Input details: {input_details}")
    logging.debug(f"Output details: {output_details}")
    interpreter.set_tensor(input_details[0]['index'], attention_mask)
    interpreter.set_tensor(input_details[1]['index'], input_ids)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def main(model_dir=None):
    """Main function to process sentences with the specified TFLite model."""
    model_path = model_dir + '/model.tflite'
    vocab_path = model_dir + '/vocab.json'

    sentences = get_sentences()

    # Load vocabulary and model
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    interpreter = load_model(model_path)
    logging.info("========================================")
    logging.info(f"Model loaded: {model_path}")

    start_total_time = timeit.default_timer()
    for input_sentence in sentences:
        # Preprocess and tokenize input
        processed_sentence, mask_position = preprocess_input(input_sentence)
        logging.debug(f'Input sentence: {input_sentence}')
        saved_processed_sentence = processed_sentence
        logging.debug(f'Processed sentence: {processed_sentence}')
        processed_sentence = tokenize(processed_sentence, vocab)
        logging.debug(f'Input data: {processed_sentence}')

        # Prepare input for inference
        processed_sentence = [vocab['<s>']] + processed_sentence + [vocab['</s>']]
        processed_sentence = pad_sequences([processed_sentence], maxlen=128, dtype='int64', value=1, padding='post',
                                           truncating='post')
        input_ids = np.array(processed_sentence, dtype=np.int64)
        attention_mask = np.ones_like(input_ids)
        logging.debug(f'Input data shape: {input_ids.shape}')
        logging.debug(f'Attention mask shape: {attention_mask.shape}')
        logging.debug(f'Attention mask: {attention_mask}')
        logging.debug(f'Input data: {input_ids}')

        # Run inference and get top predictions
        if mask_position is not None:
            start_time = timeit.default_timer()
            output = run_inference(interpreter, input_ids, attention_mask)
            elapsed_time = timeit.default_timer() - start_time
            top_words = get_top_predictions(output, vocab, mask_position)
            logging.debug(f'Top predicted words: {top_words}')
            logging.info("")

            for word in top_words[:3]:  # Use top 3 predictions
                output_sentence = saved_processed_sentence.replace('<mask>', word)
                logging.info(f'Output sentence: {output_sentence}')

            logging.info(f"Elapsed time: {elapsed_time:.3f} seconds\n")
            logging.info("")
        else:
            logging.info('No mask token found in the input sentence.')

    total_time = timeit.default_timer() - start_total_time
    logging.info(f"Total elapsed time: {total_time:.3f} seconds")
    logging.info("Average time per sentence: {:.3f} seconds".format(total_time / len(sentences)))
    logging.info("========================================")
    logging.info("")


if __name__ == "__main__":
    """
    This script performs masked language modeling using TensorFlow Lite models
    converted from various Dutch BERT models. It demonstrates how to use
    quantized TFLite models for inference in a masked language modeling task.

    Key features:
    1. Loads and uses TensorFlow Lite models
    2. Implements custom tokenization using a vocabulary file
    3. Performs preprocessing and post-processing for TFLite input and output
    4. Runs inference on TFLite models to predict masked words
    5. Provides detailed logging of the prediction process

    Usage:
    Run the script to process a set of predefined sentences with different
    quantized Dutch BERT models using TensorFlow Lite. 
    Uncomment the debug logging line to see more detailed output.

    Models used (TFLite versions):
    - robbert-v2-dutch-base
    - robbert-2022-dutch-base
    - robbert-2023-dutch-base
    - robbert-2023-dutch-large

    Note: This script requires the TFLite model files and vocabulary files
    to be present in the specified directories.
    """

    # Uncomment to set debug logging
    # set_log_level(logging.DEBUG)

    # Process sentences with different TFLite models
    main(model_dir='robbert-v2-dutch-base_tflite')
    main(model_dir='robbert-2022-dutch-base_tflite')
    main(model_dir='robbert-2023-dutch-base_tflite')
    main(model_dir='robbert-2023-dutch-large_tflite')

    main(model_dir='robbert-v2-dutch-base_tflite_int8')
    main(model_dir='robbert-2022-dutch-base_tflite_int8')
    main(model_dir='robbert-2023-dutch-base_tflite_int8')
    main(model_dir='robbert-2023-dutch-large_tflite_int8')
