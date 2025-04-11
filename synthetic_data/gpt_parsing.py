import re

def parse_gpt_response(gpt_output, num_query_titles, num_examples_per_query_title, throw_exception_on_failure=False):
    output = []
    overall_pattern = r'^<response>\s*'
    for i in range(1, num_query_titles+1):
        # Build the pattern for each list
        # Each list: number. [ items ]
        # items are `item1`, `item2`, ..., `itemN`
        item_pattern = r'`[^`]*`'
        items_pattern = fr'({item_pattern}(?:,\s*{item_pattern})*)'
        list_pattern = fr'{i}\.\s*\[\s*{items_pattern}\s*\]\s*'
        overall_pattern += list_pattern
    overall_pattern += r'</response>$'
    overall_regex = re.compile(overall_pattern)
    overall_match = overall_regex.match(gpt_output)
    if overall_match:
        # Parsing succeeded
        captured_groups = overall_match.groups()
        for i in range(num_query_titles):
            items_str = captured_groups[i]
            # Extract individual items from items_str
            items = re.findall(r'`([^`]*)`', items_str)
            if len(items) != num_examples_per_query_title:
                if throw_exception_on_failure:
                    raise Exception(f'List {i+1} does not contain {num_examples_per_query_title} items')
                else:
                    return None
            output.append(items)
        return output
    else:
        if throw_exception_on_failure:
            raise Exception('Failed to parse response')
        else:
            return None