import string





def vocabulary(blank = '-', start = '@', stop = '#'):
    """
        Function that returns a vocabulary\n
        Attributes:
        - blank: blank character used as a delimiter to double letters
        - start: character used as start string 
        - stop: character used as end string 

    """
    return [blank] + list(string.ascii_lowercase) + ['.', '?', ',', '!', start, stop, ' ']


def process_string(input_string):
    
    output_string = ""
    current_char = ""

    for char in input_string:
        if char != current_char:
            if char.isalpha() or char == '0':
                if char == '0':
                    output_string += ' '
                else:
                    output_string += char   
            current_char = char

    return output_string.strip()