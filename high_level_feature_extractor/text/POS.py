from dataclasses import (
    dataclass,
    fields,
)
import pymorphy3

# Tags without case declension:
@dataclass
class POS_ratio:
    ADVB:float = 0. # Adverb
    COMP:float = 0. # Comparative
    CONJ:float = 0. # Conjunction
    GRND:float = 0. # Gerund
    INFN:float = 0. # Infinitive
    INTJ:float = 0. # Interjection
    PRCL:float = 0. # Particle
    PRED:float = 0. # Predicative
    PREP:float = 0. # Preposition
    VERB:float = 0. # Verb (finite form)
    ADJS:float = 0. # Short form adjective
    PRTS:float = 0. # Short form participle
    NPRO:float = 0. # Noun-like Pronoun

# Tags with case declension:
# For parts of speech that undergo declension, grammatical cases are included:

    NOUN:float = 0. # Noun (with cases like nominative, accusative, etc.)
    ADJF:float = 0. # Full form adjective
    NUMR:float = 0. # Numeral
    PRTF:float = 0. # Full form participle
    NONE:float = 0. # Uknown

    @classmethod
    def text_init(
        cls,
        # TODO: left only letters and spaces
        letters_and_seps_only_text:str,
        ):
        quantity:POS_ratio = POS_ratio()
        morph = pymorphy3.MorphAnalyzer()
        
        # Split the text into words
        words = letters_and_seps_only_text.split()
        words_quantity:int = len(words)
        
        # Perform POS tagging
        for word in words:
            parse = morph.parse(word)[0]  # Get the most probable parse
            pos:str = str(parse.tag.POS)
            if pos != str(None):
                setattr(quantity, pos, getattr(quantity, pos)+1)
            else:
                quantity.NONE += 1 
        
        for field in fields(quantity):
            setattr(quantity, field.name, getattr(quantity, field.name) / words_quantity)
        return quantity
