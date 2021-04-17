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

def get_structure(character):
    file = open("cnn_output_character.txt", 'r', encoding='utf8')
    character = file.read()
    with open('cld2.csv', newline='', encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if row['C1'] == character:
                return row['C1Structure']