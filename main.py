import pandas as pd
import numpy as np
import time

MONTH_CODES = "FGHJKMNQUVXZ"

MONTH_NAMES = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]

MONTH_NUMS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

MONTH_NAME_TO_CODE = {k: v for k, v in zip(MONTH_NAMES, MONTH_CODES)}

FIELDS_MAP = {
    "Trade Date": "date",
    "Risk Free Interest Rate": "RATE",
    "Open Implied Volatility": "PRICE_OPEN",
    "Last Implied Volatility": "PRICE_LAST",
    "High Implied Volatility": "PRICE_HIGH",
    "Previous Close Price": "PRICE_CLOSE_PREV",
    "Close Implied Volatility": "IMPLIEDVOL_BLACK",
    "Strike Price": "STRIKE",
    "Option Premium": "PREMIUM",
    "General Value6": "UNDL_PRICE_SETTLE",
    "General Value7": "UNDL_PRICE_LAST",
}

FLOAT_FIELDS = [
    "PRICE_OPEN",
    "PRICE_LAST",
    "PRICE_HIGH",
    "PRICE_CLOSE_PREV",
    "IMPLIEDVOL_BLACK",
    "PREMIUM",
    "RATE",
    "STRIKE",
    "UNDL_PRICE_SETTLE",
    "UNDL_PRICE_LAST",
]


def transform(raw_data_: pd.DataFrame, instruments_: pd.DataFrame) -> pd.DataFrame:
    """
    Create a function called transform that returns a normalized table.
    Do not mutate the input.
    The runtime of the transform function should be below 1 second.

    :param raw_data_: dataframe of all features associated with instruments, with associated timestamps
    :param instruments_: dataframe of all traded instruments
    """
    tr = raw_data_.copy()
    if 'Error' in raw_data_:
        tr = tr.loc[tr['Error'] != 'Not Found']
    tr['contract'] = tr['Term']
    null_term = pd.isnull(tr['Term'])
    tr.loc[null_term, 'contract'] = tr.loc[null_term, 'Period']

    null_date = pd.isnull(tr['Trade Date'])
    if any(null_date):
        print('Warning: null dates found in raw data')
        tr.drop(tr.index[null_date], inplace=True)

    expired_instr = pd.to_datetime(tr['Expiration Date']) < pd.to_datetime(tr['Trade Date'])
    if any(expired_instr):
        print('Warning: expired instrument found in raw data')
        tr.drop(tr.index[expired_instr], inplace=True)

    null_contract = pd.isnull(tr['contract'])
    if any(null_contract):
        print('Warning: null contracts found in raw data')
        tr.drop(tr.index[null_contract], inplace=True)

    import re
    tr['RIC'].apply((lambda x: re.search(r"\d", x[1:]).start()))
    tr['base_ric'] = tr.apply(lambda x: x['RIC'].split(x['Volatility Surface Term'][:-1])[0], axis=1)
    m_tr = tr.merge(instruments_,left_on='base_ric',right_on='Base')
    m_tr['contract_year'] = pd.to_datetime(m_tr['Expiration Date']).dt.year
    m_tr['contract_month'] = m_tr['Period'].str[0:3]
    diff_year = m_tr['Expiration Date'].str[-1] != m_tr['Period'].str[-1]
    m_tr.loc[diff_year, 'contract_year'] += 1

    m_tr['month_code'] = m_tr['contract_month'].map(MONTH_NAME_TO_CODE)
    m_tr.rename(columns=FIELDS_MAP, inplace=True)
    m_tr['moneyness'] = m_tr['Volatility Surface Term'].str[:-1].astype(float)

    m_tr['symbol'] = "FUTURE_VOL_"
    bb_sym = m_tr['Bloomberg Ticker'].apply(lambda x: x if len(x) > 1 else x + '_')
    m_tr['symbol'] = m_tr['symbol'] + m_tr['Exchange'] + '_' + bb_sym + m_tr['month_code'] + m_tr['contract_year'].astype(str) + '_' + m_tr['Volatility Surface Term'].str[:-1]

    m_tr['source'] = 'refinitiv'
    m_tr['date'] = pd.to_datetime(m_tr['date'])
    for f in FLOAT_FIELDS:
        if m_tr[f].dtype != float:
            m_tr[f] = m_tr[f].str.replace(',', '').astype(float)
    out_fields = ['date', 'symbol', 'source', 'contract_year', 'month_code', 'Base', 'moneyness']
    out_fields.extend(FLOAT_FIELDS)

    out = m_tr[out_fields]

    out.set_index(['date','symbol','source','contract_year','month_code','Base','moneyness'],inplace=True)
    out_s = pd.DataFrame(out.stack())

    out_s.reset_index(inplace=True)
    out_s.rename(columns={'level_7': 'field', 0: 'value'}, inplace=True)

    out_s['field'] = pd.Categorical(out_s['field'], FLOAT_FIELDS)
    out_s.sort_values(['field', 'Base', 'contract_year', 'month_code', 'moneyness'], inplace=True)
    out_s.drop(['contract_year', 'Base', 'month_code', 'moneyness'], axis=1, inplace=True)
    out_s['field'] = out_s['field'].astype(object)
    out_s.reset_index(inplace=True)
    out_s.drop('index', axis=1, inplace=True)

    return out_s

if __name__ == '__main__':
    raw_data = pd.read_csv("raw_data.csv")
    instruments = pd.read_csv("instruments.csv")
    st = time.process_time()
    output = transform(raw_data, instruments)
    et = time.process_time()
    print(f"Wall time: {100 * (et-st)} ms")
    expected_output = pd.read_csv(
        "expected_output.csv",
        index_col=0,
        parse_dates=['date']
    )
    pd.testing.assert_frame_equal(output, expected_output)