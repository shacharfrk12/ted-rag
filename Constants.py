CHUNK_SIZE = 1024
OVERLAP_RATIO = 0.2

TOP_K = 20
MAX_CHUNKS_PER_TALK = 3

BATCH_SIZE_SMALL = 5
BATCH_SIZE = 64
BATCH_SIZE_UPSERT = 32

SMALL_CSV_PATH = "data/ted_talks_en-small.csv"
BIG_CSV_PATH = "data/ted_talks_en.csv"

LEN_SMALL_CSV = 5

METADATA_COLS_TO_DROP = ["all_speakers", "occupations", "about_speakers", "related_talks", "comments", "available_lang", "native_lang"]