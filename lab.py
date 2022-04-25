#!/usr/bin/env python3
"""6.009 Lab 8: Snek Interpreter"""

import doctest

# NO ADDITIONAL IMPORTS!

class Environment():
    def __init__(self, parent=None):
        self.within = {}
        self.parent = parent
        
    def __getitem__(self, var):
        if var in self.within:
            return self.within[var]
        else:
            return self.parent[var]
        
    def __contains__(self, item):
        try:
            self[item]
            return True
        except:
            return False
        
    def make(self, var, value):
        '''
        >>> E1 = Environment()
        >>> E1.make('x', 4)
        4
        '''
        self.within[var] = value
        return value
    
    
class Function():
    def __init__(self, args, expression, environment):
        self.args = args
        self.expression = expression
        self.environment = environment
    
    def __call__(self, variables):
        environment = self.environment
        if len(self.args) == len(variables):
            environment = Environment(self.environment)
            for i in range(len(variables)):
                environment.make(self.args[i], variables[i])
            return evaluate(self.expression, environment)
        raise SnekEvaluationError
        
        
###########################
# Snek-related Exceptions #
###########################


class SnekError(Exception):
    """
    A type of exception to be raised if there is an error with a Snek
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """
    pass


class SnekSyntaxError(SnekError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """
    pass


class SnekNameError(SnekError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """
    pass


class SnekEvaluationError(SnekError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SnekNameError.
    """
    pass


############################
# Tokenization and Parsing #
############################


def repl():
    env = Environment(snek_builtins)
    text = str(input('in> '))
    while text != 'QUIT':
        try:
            tokens = tokenize(text)
            use = parse(tokens)
            out = evaluate(use, env)
            print('    out> ' + str(out))
        except SnekEvaluationError:
            print('    out> SnekEvaluationError')
        except SnekNameError:
            print('    out> SnekNameError')
        except SnekSyntaxError:
            print('    out> SnekSyntaxError')
        text = str(input('in> '))


def number_or_symbol(x):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Snek
                      expression
    
    >>> tokenize("(cat (dog (tomato)))")
    ['(', 'cat', '(', 'dog', '(', 'tomato', ')', ')', ')']
    
    >>> exampleString = ';add the numbers 2 and 3\\n(+ ; this expression\\n 2     ; spans multiple\\n 3  ; lines\\n\\n)'
    >>> tokenize(exampleString)
    ['(', '+', '2', '3', ')']
    """
    lines = source.split('\n')
    current = ''
    #take comments out of lines
    for line in lines:
        for i in range(len(line)):
            if line[i] == ';':
                current = current + ' ' + line[:i]
                break
            elif i == len(line) - 1:
                current = current + ' ' + line
    newString = ''
    #be aware of '(' and ')'
    for i in range(len(current)):
        if current[i] == '(' or current[i] == ')':
            newString = newString + ' ' + current[i] + ' '
        else:
            newString = newString + current[i]
    return newString.split()


def find_parse_errors(parsed_expression):
    if len(parsed_expression) == 0:
        return
    if isinstance(parsed_expression, list):
        if parsed_expression[0] == 'define':
            if len(parsed_expression) != 3:
                raise SnekSyntaxError
            if not isinstance(parsed_expression[1], list):
                thing = number_or_symbol(parsed_expression[1])
                if not isinstance(thing, str):
                    raise SnekSyntaxError
            else:
                if len(parsed_expression[1]) == 0:
                    raise SnekSyntaxError
                for arg in parsed_expression[1]:
                    if isinstance(arg, (float,int)):
                        raise SnekSyntaxError
            
        if parsed_expression[0] == 'lambda':
            if len(parsed_expression) != 3:
                raise SnekSyntaxError
            if not isinstance(parsed_expression[1], list):
                raise SnekSyntaxError
            else:
                for arg in parsed_expression[1]:
                    if isinstance(arg, (float,int)):
                        raise SnekSyntaxError
    for part in parsed_expression:
        if isinstance(part, list):
            find_parse_errors(part)


def parse(tokens, index=0):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    
    >>> parse(tokenize('(define circle-area (lambda (r) (* 3.14 (* r r))))'))
    ['define', 'circle-area', ['lambda', ['r'], ['*', 3.14, ['*', 'r', 'r']]]]
    
    >>> parse(['(', '+', '2', '(', '-', '5', '3', ')', '7', '8', ')'])
    ['+', 2, ['-', 5, 3], 7, 8]
    
    >>> parse(['x'])
    'x'
    
    >>> parse(['2'])
    2
    
    >>> parse(['(', 'cat', '(', 'dog', '(', 'tomato', ')', ')', ')'])
    ['cat', ['dog', ['tomato']]]
    
    """
    if len(tokens) > 1 and tokens[0] != '(':
        raise SnekSyntaxError
    left = 0
    right = 0
    # '(' and ')' pairings
    for c in tokens:
        if c == '(':
            left += 1
        if c == ')':
            right += 1
        if left-right < 0:
            raise SnekSyntaxError
    if left != right:
        raise SnekSyntaxError
        
    def parse_expression(index):
        if tokens[index] != ')' and tokens[index] != '(':
            return number_or_symbol(tokens[index]), index+1
        else:
            opened = 0
            #find range
            for i in range(index, len(tokens)):
                if tokens[i] == '(':
                    opened += 1
                elif tokens[i] == ')':
                    opened -= 1
                if opened == 0:
                    closer = i
                    break
            back = []
            spot = index+1
            #within range
            while spot < closer:
                info, newSpot = parse_expression(spot)
                back.append(info)
                spot = newSpot
            return back, closer+1
            
    parsed_expression, _ = parse_expression(0)
    find_parse_errors(parse_expression(index))
    return parsed_expression
    


######################
# Built-in Functions #
######################


E1 = Environment()
currentEnvironment = E1

snek_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": lambda args: args[0] if len(args) == 1 else args[0]*snek_builtins["*"](args[1:]),
    "/": lambda args: args[0] if len(args) == 1 else (args[0] / snek_builtins["*"](args[1:])),
    "define": lambda args: args[2].make(args[0], args[1])
}

E1.parent = snek_builtins


##############
# Evaluation #
##############


def evaluate(tree, environment=None):
    """
    Evaluate the given syntax tree according to the rules of the Snek
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    >>> evaluate('+')
    <built-in function sum>
    
    >>> evaluate(3.14)
    3.14
    
    >>> evaluate(['+', 3, 7, 2])
    12
    
    >>> evaluate(parse(tokenize('(+ 2 (- 3 4 5))')))
    -4
    
    >>> int(evaluate(parse(tokenize('(- 3.14 1.14 1)'))))
    1
    
    >>> evaluate(parse(tokenize('(+ 2 (- 3 4))')))
    1
    
    >>> example = Environment(snek_builtins)
    >>> first0 = tokenize('(define pi 3.14)')
    >>> second0 = parse(first0)
    >>> evaluate(second0, example)
    3.14

    >>> first1 = tokenize('(define radius 2)')
    >>> second1 = parse(first1)
    >>> evaluate(second1, example)
    2

    >>> first2 = tokenize('(* pi radius radius)')
    >>> second2 = parse(first2)
    >>> evaluate(second2, example)
    12.56
    
    >>> first3 = tokenize('(lambda (x y) (+ x y))')
    >>> second3 = parse(first3)
    >>> type(evaluate(second3))
    <class '__main__.Function'>
    
    >>> E1 = Environment(snek_builtins)
    >>> first4 = tokenize('(define square (lambda (x) (* x x)))')
    >>> second4 = parse(first4)
    >>> type(evaluate(second4, E1))
    <class '__main__.Function'>
    >>> evaluate(parse(tokenize('(square 2)')), E1)
    4
    
    >>> example = Environment(snek_builtins)
    >>> evaluate(parse(tokenize('(define x 7)')), example)
    7
    
    >>> type(evaluate(parse(tokenize('(define foo (lambda (x) (lambda (y) (+ x y))))')), example))
    <class '__main__.Function'>
    
    >>> type(evaluate(parse(tokenize('(define bar (foo 3))')), example))
    <class '__main__.Function'>
    
    >>> evaluate(parse(tokenize('(bar 2)')), example)
    5
    
    >>> example2 = Environment(snek_builtins)
    >>> type(evaluate(parse(tokenize('(define (square x) (* x x))')), example2))
    <class '__main__.Function'>
    
    >>> tokened = tokenize('(square 4)')
    >>> parsed = parse(tokened)
    >>> evaluate(parsed, example2)
    16
    
    >>> example3 = Environment(snek_builtins)
    >>> type(evaluate(parse(tokenize('(define square (lambda (x) (* x x)))')), example3))
    <class '__main__.Function'>
    >>> evaluate(parse(tokenize('(square 21)')), example3)
    441
    """
    if environment is None:
        environment = Environment(snek_builtins)
    if isinstance(tree, list):
        # Defining new
        if tree[0] == 'define':
            if isinstance(tree[1], list):
                return environment['define']([tree[1][0], Function(tree[1][1:], tree[2], environment), environment])
            else:
                return environment['define']([tree[1], evaluate(tree[2], environment), environment])
        # Making Function
        if tree[0] == 'lambda':
            return Function(tree[1], tree[2], environment)
        # Running Function
        args = [evaluate(part, environment) for part in tree[1:]]
        evaled = evaluate(tree[0], environment)
        if callable(evaled):
            return evaled(args)
        raise SnekEvaluationError
    else:
        # Simple cases
        if isinstance(tree, (int, float, Function)):
            return tree
        if isinstance(tree, str): 
            if tree in environment:
                return environment[tree]
            else:
                raise SnekNameError
        if tree in environment:
            return tree()
        raise SnekEvaluationError
        
        
def result_and_env(tree, environment=None):
    '''
    >>> example = Environment(snek_builtins)
    >>> type(result_and_env(parse(tokenize('(define (square x) (* x x))')), example))
    <class 'tuple'>
    
    >>> result, env = result_and_env(parse(tokenize('(square 4)')), example)
    >>> result
    16
    >>> isinstance(env, Environment)
    True
    '''
    if environment is None:
        environment = Environment(snek_builtins)
    return evaluate(tree, environment), environment


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod(report=True)
    repl()

    pass
