import logging
from transformers import pipeline

from utils import preprocess_input

# Set up logging
logging.basicConfig(level=logging.INFO)


def set_log_level(log_level):
    """Set the logging level."""
    logging.getLogger().setLevel(log_level)


def run_inference(sentence, pipe):
    """Process sentence with mask, predict masked word."""
    if "<mask>" not in sentence:
        logging.debug("No mask token found in the sentence.")
        return
    return pipe(sentence)[0]


def main(model_path):
    """Main function to process sentences with the specified model."""
    logging.info("========================================")

    # Load the model
    pipe = pipeline("fill-mask", model=model_path)
    logging.info(f"Model loaded: {model_path}")

    # Example sentences
    sentences = [
        "Ik heb een vriend die altijd te laat komt.",
        "Ik weet die ik het kan.",
        "Ik weet DAT ik het kan.",
        "Daarom is het belangrijk, je moet goed opletten.",
        "Ik ken een man die altijd grapjes maakt.",
        "Ze heeft een jurk gekocht die perfect past.",
        "Er is een boek dat ik je echt kan aanraden.",
        "We bezochten een stad die bekend staat om haar architectuur.",
        "Hij las een artikel dat zijn mening veranderde.",
        "Ze vertelde over een ervaring die haar leven veranderde.",
        "Ik zag een film die mij aan het denken zette.",
        "Hij gebruikt een methode die zeer effectief is.",
    ]

    for sentence in sentences:
        masked_sentence, _ = preprocess_input(sentence)
        output = run_inference(masked_sentence, pipe)
        logging.info(f"Original sentence: {sentence}")
        logging.info(f"Processed sentence: {masked_sentence}")
        logging.info(f"Prediction: {output}")
        logging.info("")

    logging.info("========================================")


if __name__ == '__main__':
    """
    This script performs masked language modeling using the Transformers pipeline
    on various Dutch BERT models. It provides a simpler implementation compared to
    the tf-main.py script, using the high-level pipeline API.

    Key features:
    1. Uses Transformers pipeline for easy model loading and inference
    2. Supports multiple Dutch BERT models
    3. Preprocesses input sentences to include a mask token
    4. Runs inference to predict masked words
    5. Provides logging of original sentences, processed sentences, and predictions

    Usage:
    Run the script to process a set of predefined sentences with different Dutch
    BERT models using the Transformers pipeline.

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