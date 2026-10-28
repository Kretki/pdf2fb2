"""Find the passage of a reference region that corresponds to one page of output.

A reference page holds running heads, page numbers and catchwords the pipeline drops on purpose, and a
transcript holds the whole book. Both are trimmed the same way: word trigrams of the output are looked up in
the region (case and punctuation ignored), the most frequent offset between output and region positions
gives the matching diagonal, and the reference is cut from the first to the last trigram on that diagonal,
widened by the unmatched output words before and after them plus a margin. The alignment then skips the
surplus reference words at both ends for free (``edit_alignment.align_words`` with free reference ends).
"""

from collections import Counter, defaultdict

from pdf2fb2.benchmark.text_normalization import match_key

SHINGLE_WORDS = 3
DIAGONAL_TOLERANCE = 40   # words of drift allowed between output and reference along one page
MIN_MATCHED_SHINGLES = 3
EDGE_MARGIN = 10          # extra reference words searched beyond each end


class RegionIndex:
    """Trigram positions of one reference region (built once per region).

    Usage:
        Build with the region words and call ``locate_passage`` for every output page searched in it.
    """

    def __init__(self, region_words: list[str]) -> None:
        """Index the region.

        Args:
            region_words: Normalised reference words.
        """
        self.region_words = region_words
        self._shingle_positions: dict[tuple[str, ...], list[int]] = defaultdict(list)
        region_keys = [match_key(region_word) for region_word in region_words]
        for start_index in range(len(region_keys) - SHINGLE_WORDS + 1):
            self._shingle_positions[tuple(region_keys[start_index:start_index + SHINGLE_WORDS])].append(start_index)

    def matched_offsets(self, output_words: list[str]) -> list[tuple[int, int]]:
        """Pairs (output position, region position) of trigrams found in the region.

        Args:
            output_words: Normalised output words of the page.

        Returns:
            Matches in output order.
        """
        output_keys = [match_key(output_word) for output_word in output_words]
        return [(output_index, region_index)
                for output_index in range(len(output_keys) - SHINGLE_WORDS + 1)
                for region_index in self._shingle_positions.get(
                    tuple(output_keys[output_index:output_index + SHINGLE_WORDS]), ())]

    def locate_passage(self, output_words: list[str]) -> list[str] | None:
        """Reference words matching an output page.

        Args:
            output_words: Normalised output words of the page.

        Returns:
            The reference passage (with margins), or None when fewer than three trigrams match.
        """
        matched_pairs = self.matched_offsets(output_words)
        if len(matched_pairs) < MIN_MATCHED_SHINGLES:
            return None
        offset_votes = Counter((region_index - output_index) // DIAGONAL_TOLERANCE
                               for output_index, region_index in matched_pairs)
        best_bucket = offset_votes.most_common(1)[0][0]
        diagonal_pairs = [(output_index, region_index) for output_index, region_index in matched_pairs
                          if abs((region_index - output_index) // DIAGONAL_TOLERANCE - best_bucket) <= 1]
        if len(diagonal_pairs) < MIN_MATCHED_SHINGLES:
            return None
        first_output, first_region = diagonal_pairs[0]
        last_output, last_region = max(diagonal_pairs, key=lambda matched_pair: matched_pair[1])
        start_index = max(0, first_region - first_output - EDGE_MARGIN)
        end_index = min(len(self.region_words), last_region + len(output_words) - last_output + EDGE_MARGIN)
        return self.region_words[start_index:end_index]
