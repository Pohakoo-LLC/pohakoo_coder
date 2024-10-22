from validator import is_correct

test_response = """
Here's the code you requested! I am a GPT hehe
```py
def add_numbers(num, othernum):
    return num + othernum

print(add_nummer(2, 1500))

print(add_numbers(2, 1500))
```
You could also do this:
```py
print(add_numbers(1500, 2))
```
""".split(' ')

print(is_correct(test_response))
# Expected: {"\nHere's": 1, 'the': 1, 'code': 1, 'you': 1, 'requested!': 1, 'I': 1, 'am': 1, 'a': 1, 'GPT': 1, 'hehe\n```py\ndef': 1, 'add_numbers(num,': 1, 'othernum):\n': 1, '': 1, 'return': 1, 'num': 1, '+': 1, 'othernum\n\nprint(add_nummer(2,': 1, '1500))\n\nprint(add_numbers(2,': 1, '1500))\n```\nYou': -1, 'could': 1, 'also': 1, 'do': 1, 'this:\n```py\nprint(add_numbers(1500,': 1, '2))\n```\n': 1}