from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import sparse

def build_mappings(df):
    """Build user and item mappings to consecutive integers starting from 0"""
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())
    
    u2i = {user: idx for idx, user in enumerate(unique_users)}
    it2i = {item: idx for idx, item in enumerate(unique_items)}
    
    # Create mapped dataframe
    df_mapped = df.copy()
    df_mapped['user_id'] = df_mapped['user_id'].map(u2i)
    df_mapped['item_id'] = df_mapped['item_id'].map(it2i)
    
    return df_mapped, u2i, it2i

def leave_last_one_out(df):
    """Split data using leave-last-one-out strategy based on timestamp"""
    # Sort by user and timestamp
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values(['user_id', 'timestamp'])
    else:
        # If no timestamp, just use the order
        df_sorted = df.sort_values('user_id')
    
    train_data = []
    test_data = []
    
    for user_id, group in df_sorted.groupby('user_id'):
        if len(group) > 1:
            # Last interaction goes to test
            test_data.append(group.iloc[-1])
            # Rest goes to train
            train_data.extend(group.iloc[:-1].to_dict('records'))
        else:
            # If user has only one interaction, put it in train
            train_data.extend(group.to_dict('records'))
    
    train_df = pd.DataFrame(train_data) if train_data else pd.DataFrame()
    test_df = pd.DataFrame(test_data) if test_data else pd.DataFrame()
    
    return train_df, test_df

def to_csr(train: pd.DataFrame, n_users: int, n_items: int):
    rows=train['user_id'].to_numpy(); cols=train['item_id'].to_numpy()
    data=train['weight'].to_numpy() if 'weight' in train.columns else np.ones(len(train), dtype=np.float32)
    return sparse.csr_matrix((data,(rows,cols)), shape=(n_users,n_items), dtype=np.float32)
