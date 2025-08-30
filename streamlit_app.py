import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Sentiment Analysis Dashboard',
    page_icon=':chart_with_upwards_trend:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_sentiment_data():
    """Grab sentiment data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/sentiment_data.csv'
    raw_sentiment_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 2020
    MAX_YEAR = 2024

    # The data above has columns like:
    # - Keyword
    # - Category
    # - [Stuff I don't care about]
    # - Sentiment for 2020
    # - Sentiment for 2021
    # - Sentiment for 2022
    # - ...
    # - Sentiment for 2024
    #
    # ...but I want this instead:
    # - Keyword
    # - Category
    # - Year
    # - Sentiment
    #
    # So let's pivot all those year-columns into two: Year and Sentiment
    sentiment_df = raw_sentiment_df.melt(
        ['Keyword'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'Sentiment',
    )

    # Convert years from string to integers
    sentiment_df['Year'] = pd.to_numeric(sentiment_df['Year'])

    return sentiment_df

sentiment_df = get_sentiment_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :chart_with_upwards_trend: Sentiment Analysis Dashboard

Browse sentiment analysis data for various keywords over time. Sentiment scores range from -1 (very negative) 
to +1 (very positive), with 0 being neutral. This dashboard helps you track how public sentiment toward 
different topics has evolved.
'''

# Add some spacing
''
''

min_value = sentiment_df['Year'].min()
max_value = sentiment_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

keywords = sentiment_df['Keyword'].unique()

if not len(keywords):
    st.warning("Select at least one keyword")

selected_keywords = st.multiselect(
    'Which keywords would you like to view?',
    keywords,
    keywords[:6] if len(keywords) >= 6 else keywords)  # Select first 6 keywords by default

''
''
''

# Filter the data
filtered_sentiment_df = sentiment_df[
    (sentiment_df['Keyword'].isin(selected_keywords))
    & (sentiment_df['Year'] <= to_year)
    & (from_year <= sentiment_df['Year'])
]

st.header('Sentiment over time', divider='gray')

''

st.line_chart(
    filtered_sentiment_df,
    x='Year',
    y='Sentiment',
    color='Keyword',
)

''
''


first_year = sentiment_df[sentiment_df['Year'] == from_year]
last_year = sentiment_df[sentiment_df['Year'] == to_year]

st.header(f'Sentiment in {to_year}', divider='gray')

''

for keyword in selected_keywords:
    first_sentiment = first_year[first_year['Keyword'] == keyword]['Sentiment'].iat[0] if not first_year[first_year['Keyword'] == keyword].empty else float('nan')
    last_sentiment = last_year[last_year['Keyword'] == keyword]['Sentiment'].iat[0] if not last_year[last_year['Keyword'] == keyword].empty else float('nan')

    if math.isnan(first_sentiment) or math.isnan(last_sentiment):
        change = 'n/a'
        delta_color = 'off'
    else:
        change = f'{last_sentiment - first_sentiment:+.3f}'
        delta_color = 'normal' if (last_sentiment - first_sentiment) >= 0 else 'inverse'

    # Format sentiment score with color coding
    if not math.isnan(last_sentiment):
        if last_sentiment > 0.1:
            sentiment_label = f'{last_sentiment:.3f} (Positive)'
        elif last_sentiment < -0.1:
            sentiment_label = f'{last_sentiment:.3f} (Negative)'
        else:
            sentiment_label = f'{last_sentiment:.3f} (Neutral)'
    else:
        sentiment_label = 'n/a'

    st.metric(
        label=f'{keyword} Sentiment',
        value=sentiment_label,
        delta=change,
        delta_color=delta_color
    )