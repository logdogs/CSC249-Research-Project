import csv

# Look up the character given \Sigma' and obtain the structural information necessary for quality 
#   evaluation.
# file = open("cnn_output_character.txt", 'r', encoding='utf8')
# character = file.read()

# with open('cld2.csv', newline='', encoding='utf8') as csvfile:
#     reader = csv.DictReader(csvfile)
#     #reader = csv.reader(csvfile)
#     #print(reader)
#     for row in reader:
#         if row['C1'] == character:
#             print(row['C1Structure'])
#             break
#         #print(row['C1'], row['C1Structure'])

# We can use this as a key into a dictionary of frame segmenters for breaking up the image into meaningful
#   pieces
def get_structure(character):
    # file = open("cnn_output_character.txt", 'r', encoding='utf8')
    # character = file.read()
    with open('cld2.csv', newline='', encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if row['C1'] == character:
                return row['C1Structure']

# This simply grabs the semantic radical (if any, or 'NA' if none)
def get_semantic_radical(character):
    # file = open("cnn_output_character.txt", 'r', encoding='utf8')
    # character = file.read()
    with open('cld2.csv', newline='', encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['C1'] == character:
                return row['C1SR']
# Get the phonetic radical (if any, NA otherwise)
def get_phonetic_radical(character):
    # file = open("cnn_output_character.txt", 'r', encoding='utf8')
    # character = file.read()
    with open('cld2.csv', newline='', encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['C1'] == character:
                return row['C1PR']

# This is the actual recursive decomposition, but it suffers one problem because of how the CLD is structured
# Should we get a new database, which I think may be a good idea later, this will be fine
# def decompose(character,components):
#     sr = get_semantic_radical(character)
#     pr = get_phonetic_radical(character)
#     if sr != 'NA':
#         components += decompose(sr,components)
#     if pr != 'NA':
#         components += decompose(pr,components)
#     return components.append(character)
# The workhorse which just breaks up the character into two components
def decompose(character,components):
    sr = get_semantic_radical(character)
    pr = get_phonetic_radical(character)
    if sr != 'NA':
        components.append(sr)
    if pr != 'NA':
        components.append(pr)
    return components

# Wrapper function
def recursive_decomposition():
    file = open("cnn_output_character.txt", 'r', encoding='utf8')
    character = file.read()
    decomp = decompose(character,[])
    return decomp