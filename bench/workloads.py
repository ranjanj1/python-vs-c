"""Frozen inputs. DEV prompts are given to agents. EVAL prompts are never in an agent bundle; their hash is recorded before evaluation."""
import hashlib, json

# 110M is the only experiment model (Ranjan, 5 Nov). 15M stays as a fast fixture for testing the gate itself; it gives no results.
MODELS = {'15M': 'models/stories15M.bin', '110M': 'models/stories110M.bin'}
EXPERIMENT_MODEL = '110M'
TOKENIZER = 'sources/upstream/tokenizer.bin'
DEV = ['Once upon a time', 'Lily and Tom went to the park to play on the swings',
       'The little dog was sad because he could not find his red ball anywhere in the house']
EVAL = ['A tiny mouse lived under the kitchen floor', 'Sam had a blue kite',
        'One rainy morning, Grandma opened the old wooden box and found a shiny key that nobody had ever seen before', 'Birds']
TOKENIZER_CASES = ['Once upon a time', 'hello  world', ' leading space', 'trailing space ', 'Numbers 12345 and punctuation!?', 'naïve café', '日本語', 'tab\there', '']
NEW_TOKENS = 64
LONG_POSITIONS = 256   # one teacher-forced numeric check past the generation length
TOL_MAX_ABS = 5e-4   # about 12x the worst error of known-correct code against float64 in the pilot (4.2e-5)


def eval_hash():
    return hashlib.sha256(json.dumps(EVAL).encode()).hexdigest()
