import re
import cld_lookup

"""
    The decomp_dictionary should only be able to break down the following four types of compositions:
        1. a (across)
        2. d (down)
        3. r (repeat/rotate patterns)
    
    It will also make use of the 'ml' pattern to in order to grab a better comparison for radicals that
        have their form changed when they appear on the left side of a character. This will improve the
        accuracy of comparisons for segments
    
    There are certain surrounding (s) pattern characters which we could absolutely segment, however, the
        generalizations here make it incredibly difficult to handle this in the general case as some 
        characters end up requiring that lines be extended or components squished. The line extension
        is what truly makes it unreasonable, however.
    
    It should also be noted that with a and d type compositions, a weighting will be necessary during
        comparison, and during further analysis (if necessary) we must take into account that the type
        of stroke may change for components in these types. For example, the verticle stroke in 手
        becomes curved when it appears as the semantic radical in 看. This problem will need a lot of 
        thought and careful handling when we're certain that the base cases (single characters) are being
        handled well by our comparisons.py 
"""

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
    def get_composition_structure(self, character):
        decomp_data = self.get_decomp_info(character)
        decomp_data = re.split(r'\(', decomp_data)
        return decomp_data[0]
        