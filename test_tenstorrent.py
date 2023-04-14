
from edenai_apis import Text

sentiment_analysis = Text.sentiment_analysis("tenstorrent")
tt_res = sentiment_analysis(language="en", text="as simple as that")

# Provider's response
print(tt_res.original_response)

# Standardized version of Provider's response
print(tt_res.standardized_response)

