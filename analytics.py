from typing import Dict, Tuple
import pandas as pd

# Expected columns (best-effort, code tries to adapt if columns differ)
EXPECTED_COLUMNS = {
    'customer_id': ['customer_id', 'id', 'user_id'],
    'sales': ['sales', 'revenue', 'amount', 'order_value'],
    'converted': ['converted', 'is_converted', 'purchased', 'won', 'conversion'],
    'campaign': ['campaign', 'campaign_name', 'channel'],
    'impressions': ['impressions', 'views'],
    'clicks': ['clicks', 'click'],
}


def find_column(df: pd.DataFrame, aliases):
    for col in df.columns:
        if str(col).lower() in aliases:
            return col
    return None


def compute_basic_analytics(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame, Dict]:
    """
    Computes basic analytics and returns:
      - analytics: dict with totals and summary tables
      - processed_df: normalized dataframe with standardized columns as available
      - schema: mapping of standardized names to actual column names
    """
    df = df.copy()

    # Map columns to standard names where possible
    schema = {}
    for std, aliases in EXPECTED_COLUMNS.items():
        found = find_column(df, aliases)
        if found:
            schema[std] = found

    # Normalize data types
    if 'sales' in schema:
        df[schema['sales']] = pd.to_numeric(df[schema['sales']], errors='coerce').fillna(0)

    if 'converted' in schema:
        # Try to normalize to 0/1
        conv = df[schema['converted']].astype(str).str.lower()
        df['_converted'] = conv.isin(['1', 'true', 'yes', 'y', 'won', 'purchase', 'purchased']).astype(int)
    else:
        df['_converted'] = None

    # Compute basic metrics
    total_sales = float(df[schema['sales']].sum()) if 'sales' in schema else None

    conversion_rate = None
    if df['_converted'] is not None:
        if df['_converted'].notna().any():
            conversion_rate = float(df['_converted'].mean())

    # Top performing campaigns by sales and conversion
    top_campaigns = None
    top_campaigns_table = None
    if 'campaign' in schema:
        group = df.groupby(schema['campaign']).agg({
            schema['sales']: 'sum' if 'sales' in schema else 'size',
            '_converted': 'mean' if df['_converted'] is not None else 'size'
        }).rename(columns={
            schema['sales']: 'total_sales' if 'sales' in schema else 'count',
            '_converted': 'conversion_rate' if df['_converted'] is not None else 'count'
        })
        # cleanup float
        if 'conversion_rate' in group.columns:
            group['conversion_rate'] = group['conversion_rate'].fillna(0.0)
        top_campaigns = group.sort_values(
            by=['total_sales' if 'total_sales' in group.columns else 'count'], ascending=False
        ).head(5)
        top_campaigns_table = top_campaigns.reset_index().to_dict(orient='records')

    analytics = {
        'total_sales': total_sales,
        'conversion_rate': conversion_rate,
        'top_campaigns_table': top_campaigns_table,
    }

    # Build a processed frame with standardized column names where available
    processed_df = pd.DataFrame()
    for std, real in schema.items():
        processed_df[std] = df[real]

    if '_converted' in df.columns:
        processed_df['converted'] = df['_converted']
        schema['converted'] = 'converted'

    # Fill campaign missing for modeling display
    if 'campaign' in processed_df.columns:
        processed_df['campaign'] = processed_df['campaign'].fillna('Unknown')

    return analytics, processed_df, schema
