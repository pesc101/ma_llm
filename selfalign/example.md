## Filename: example.py
## Path: src/example.py
## is_class_method: False
## is_function: True
## Start line: 12
## End line: 54
## imports:
# from pygments.lexer import Lexer
# from pygments.lexers import ClassNotFound, get_lexer_by_name, guess_lexer
# from pygments.token import Token
# from pygments.util import ClassNotFound


def is_valid_code(input_text: str) -> bool:
    try:
        lexer: Lexer = guess_lexer(input_text)

        if lexer.name != "Text":
            return True
        lexer = get_lexer_by_name("text", stripall=True)

        token_stream = lexer.get_tokens(input_text)
        return any(token[0] != Token.Text for token in token_stream)

    except ClassNotFound:
        return False
