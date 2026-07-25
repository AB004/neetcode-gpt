from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        result = []
        for number in numbers:
            s = str(number)
            tokens = []
            i = 0
            while i < len(s):
                longest = None
                for j in range(len(s),i,-1):
                    if s[i:j] in vocab:
                        longest =  s[i:j]
                        break
                tokens.append(longest)
                i += len(longest)
            result.append(tokens)
        return result



    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        i = 0
        total=0
        while i < len(text):
            logest = None
            for j in range(len(text),i,-1):
                if text[i:j] in vocab:
                    longest = text[i:j]
                    break
            total += 1
            i += len(longest)
        return total
        

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split()
        if len(words)==0:
            return 0.0
        total = self.count_tokens(text,vocab)
        return round(total/len(words),4)
