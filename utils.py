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

    # Random masking (alternative approach):
    # words = sentence.split()
    # mask_position = np.random.randint(0, len(words))
    # words[mask_position] = '<mask>'
    # return ' '.join(words), mask_position


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
