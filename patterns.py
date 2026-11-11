import re

# Comprehensive Bilingual GPT-4 Regex with Extended Ethiopic Support
ETHIOPIC_GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)| ?[\u1200-\u135A\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]+| ?[a-zA-Z]+|[0-9\u1369-\u137C]{1,3}|[\u1360-\u1368]| ?[^\s\w]+|\s+"""
SPLIT_PATTERN = re.compile(ETHIOPIC_GPT4_PATTERN)


