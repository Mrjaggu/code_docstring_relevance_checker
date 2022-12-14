#!/usr/bin/env python
import ast

with open(r'C:\AJAY\HACKATHON\test.py') as f:
    code = ast.parse(f.read())

for node in ast.walk(code):
    # print(node)
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
        docstring = ast.get_docstring(node)
        if docstring:
            lineno = getattr(node, 'lineno', None)
            # print()
            print(lineno+1,"|",repr(docstring))

# def get_docstring(filepath=r'C:\AJAY\HACKATHON\test.py'):
#     with open(filepath) as f:
#         code = ast.parse(f.read())

#     line_no = []
#     for node in ast.walk(code):
#         if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
#             docstring = ast.get_docstring(node)
#             if docstring:
#                 # print()
#                 lineno = getattr(node, 'lineno', None)
#                 print(lineno,"|",repr(docstring))
#                 line_no.append([lineno,repr(docstring)])

#     return line_no

# check_line_number = get_docstring()
# print("Docstring->",check_line_number)
