import altair as alt
import pandas as pd

# Define data
data = pd.DataFrame({
    'Library': ['Keras', 'PyTorch'],
    'Usability': [8, 9]
})

# Create bar chart
bar_chart = alt.Chart(data).mark_bar().encode(
    x=alt.X('Library:N', title='Library'),
    y=alt.Y('Usability:Q', title='Usability (out of 10)'),
    color=alt.Color('Library:N', scale=alt.Scale(range=['#FF5733', '#2E86C1']))
).properties(
    title='Usability of Keras vs. PyTorch'
)

# Add text labels to bars
text = bar_chart.mark_text(
    align='center',
    baseline='bottom',
    dy=-5
).encode(
    text=alt.Text('Usability:Q', format='.1f')
)

# Display chart
(bar_chart + text).configure_axis(
    grid=False
).configure_view(
    strokeWidth=0
).show()

