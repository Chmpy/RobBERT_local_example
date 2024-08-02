import tensorflow as tf
import torch


def preprocess_input(sentence):
    """
    Mask 'die' or 'dat' in the input sentence.

    Args:
    sentence (str): Input sentence.

    Returns:
    tuple: (preprocessed sentence, mask position or None)
    """
    words = sentence.split()
    mask_position = None
    for i, word in enumerate(words):
        if word.lower() in ['die', 'dat']:
            words[i] = '<mask>'
            mask_position = i
            break
    return ' '.join(words), mask_position


def get_top_predictions(logits, vocab, mask_position, top_k=5):
    """
    Get top-k predictions for the masked token.

    Args:
    logits (numpy.ndarray or torch.Tensor): Model output logits
    vocab (dict): Vocabulary dictionary
    mask_position (int): Position of the mask token
    top_k (int): Number of top predictions to return

    Returns:
    list: Top-k predicted words
    """
    # Convert logits to tensor if it's not already
    if not isinstance(logits, (tf.Tensor, torch.Tensor)):
        logits = tf.convert_to_tensor(logits)

    # Get logits for the mask token
    mask_token_logits = logits[0, mask_position + 1]

    # Apply softmax
    if isinstance(logits, torch.Tensor):
        probs = torch.nn.functional.softmax(mask_token_logits, dim=0)
        top_indices = torch.topk(probs, k=top_k).indices.tolist()
    else:
        probs = tf.nn.softmax(mask_token_logits)
        top_indices = tf.math.top_k(probs, k=top_k).indices.numpy()

    # Convert indices to words
    return [list(vocab.keys())[list(vocab.values()).index(idx)] for idx in top_indices]


def get_sentences():
    """Get example sentences."""
    return [
        "Ik heb een vriend die altijd te laat komt.",  # Correct
        "Ik weet die ik het kan.",  # Incorrect
        "Ik weet dat ik het kan.",  # Correct
        "Daarom is het belangrijk, je moet goed opletten.",  # Incorrect
        "Ik ken een man die altijd grapjes maakt.",  # Correct
        "Ze heeft een jurk gekocht die perfect past.",  # Correct
        "Er is een boek dat ik je echt kan aanraden.",  # Correct
        "We bezochten een stad die bekend staat om haar architectuur.",  # Correct
        "Hij las een artikel dat zijn mening veranderde.",  # Correct
        "Ze vertelde over een ervaring die haar leven veranderde.",  # Correct
        "Ik zag een film die mij aan het denken zette.",  # Correct
        "Hij gebruikt een methode die zeer effectief is.",  # Correct
        "Het kind dat in de tuin speelt, is mijn neefje.",  # Correct
        "De vraag die ze stelde, was erg moeilijk.",  # Correct
        "De beslissing dat hij maakte, was erg moeilijk.",  # Incorrect
        "Dit is de laptop die ik wil kopen.",  # Correct
        "Dit is de laptop dat ik wil kopen.",  # Incorrect
        "Het project dat we gestart zijn, verloopt goed.",  # Correct
        "De vrouw die je daar ziet, is mijn lerares.",  # Correct
        "De resultaten dat we behaalden, waren indrukwekkend.",  # Incorrect
        "De resultaten die we behaalden, waren indrukwekkend.",  # Correct
        "De bloemen die je hebt meegenomen, zijn prachtig.",  # Correct
        "De informatie dat hij gaf, was nuttig.",  # Incorrect
        "De informatie die hij gaf, was nuttig.",  # Correct
        "Het idee dat ze voorstelde, was briljant.",  # Correct
        "Het idee die ze voorstelde, was briljant.",  # Incorrect
        "De kat dat op het dak zit, is van ons.",  # Incorrect
        "De kat die op het dak zit, is van ons.",  # Correct
        "De film die we gisteren keken, was spannend.",  # Correct
        "De auto dat ik gisteren zag, was heel mooi.",  # Incorrect
    ]
