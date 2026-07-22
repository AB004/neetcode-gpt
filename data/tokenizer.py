from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        merge = []
        tokens = list(corpus)
        for _ in range(num_merges):
            if len(tokens)<2:
                break
            pair_count = {}
            for i in range(len(tokens)-1):
                pair = (tokens[i],tokens[i+1])
                pair_count[pair] = pair_count.get(pair,0) + 1;
            
            max_freq = max(pair_count.values())
            candidates = []
            for pair,freq in pair_count.items():
                if freq == max_freq:
                    candidates.append(pair)
            best_pair = min(candidates)
            merge.append(best_pair)
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i<len(tokens)-1 and tokens[i]==best_pair[0] and tokens[i+1]==best_pair[1]:
                    new_tokens.append(tokens[i]+tokens[i+1])
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            tokens = new_tokens
        return merge


