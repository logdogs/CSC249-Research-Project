import re
import cld_lookup

class decomp_dictionary:
    def __init__(self):
        self.dictionary = dict()
        file = open('cjk-decomp.txt', 'r',encoding='utf8')
        data = file.read()
        data = data.split("\n")
        for line in data:
            key_val = line.split(":")
            if len(key_val) > 1:
                self.dictionary[key_val[0]] = key_val[1]
            else:
                self.dictionary[key_val[0]] = ''
    def get_decomp_info(self,character):
        return self.dictionary[character]
    # Returns a list of the components of a character without any intermediates
    # It appears as though the only time intermediates pop up is with traditional characters, and those
    #   intermediates appear to be strokes.
    # To avoid bizarre behavior I noticed with single characters like 大 and 日, we have a
    #   'single character' check first using our CLD stuff from before
    def get_components(self,character):
        if cld_lookup.get_structure(character) == 'SG':
            return [character]
        decomp_info = self.get_decomp_info(character)
        type_inter = re.split(r'\(',decomp_info)
        first_inter = re.split(r'\,', type_inter[1])
        first = first_inter[0]
        second = first_inter[1].replace(")", "")
        components = [first,second]
        if len(first) > 1 or len(second) > 1:
            if len(first) > 1:
                # components.remove(first)
                components += self.get_components(first)
            if len(second) > 1:
                # components.remove(second)
                components += self.get_components(second)
        for component in components:
            if len(component) > 1:
                components.remove(component)
        return components