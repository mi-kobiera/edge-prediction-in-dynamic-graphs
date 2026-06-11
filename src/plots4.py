import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# --- Zaimportuj tutaj swoje funkcje ładujące ---
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from data.loader import load_tgn_format_dataset

ROOT_DIR = "../datasets" # Zmień na właściwą ścieżkę dla TGB

# --- Ustawienia dla rysowania TET ---
E_ABSENT = 0
E_PRESENCE_GENERAL = 1
E_SEEN_IN_TRAIN = 2
E_IN_TEST = 3
E_NOT_IN_TEST = 4
E_ONLY_TRAIN = 10
E_TRAIN_AND_TEST = 20
E_TRANSDUCTIVE = 30
E_INDUCTIVE = 40

# ==========================================
# 1. ŁADOWANIE I SZYBKA KONWERSJA (PANDAS)
# ==========================================

def load_network_data(dataset_name):
    print(f"\n--- Ładowanie zbioru: {dataset_name} ---")
    if dataset_name == 'tgbl-wiki':
        dataset = PyGLinkPropPredDataset(name=dataset_name, root=ROOT_DIR)
        data = dataset.get_TemporalData()
    else:
        data_dir = f"/Users/mi-kobiera/Downloads/TG_network_datasets/{dataset_name}"
        data = load_tgn_format_dataset(
            data_dir=data_dir, 
            network_name=dataset_name,
            val_ratio=0.15,
            test_ratio=0.15
        )
    return data

def get_interval_for_dataset(dataset_name):
    if dataset_name in ['Enron', 'enron']: return 86400 * 30  
    elif dataset_name in ['uci', 'UCI']: return 86400 * 5   
    elif dataset_name in ['UNvote', 'UNVote']: return None      
    else: return 86400

def get_edgelist_dataframe(data, dataset_name, interval_size=None):
    u_arr = data.src.cpu().numpy() if hasattr(data, 'src') else data.u.cpu().numpy()
    v_arr = data.dst.cpu().numpy() if hasattr(data, 'dst') else data.v.cpu().numpy()
    t_arr = data.t.cpu().numpy()

    df = pd.DataFrame({'u': u_arr, 'v': v_arr, 'ts': t_arr})
    
    if dataset_name.lower() in ['unvote']:
        if df['ts'].max() > 100000:
            df['ts'] = pd.to_datetime(df['ts'], unit='s').dt.year
        else:
            df['ts'] = df['ts'].astype(int)
            
        min_year = df['ts'].min()
        if min_year == 1970:
            df['ts'] = df['ts'] - 24
        elif min_year == 0:
            df['ts'] = df['ts'] + 1946
            
    elif interval_size is not None:
        df['ts'] = (df['ts'] // interval_size).astype(int)
    else:
        df['ts'] = df['ts'].astype(int)
        
    df = df.drop_duplicates(subset=['ts', 'u', 'v']).copy()
    df = df.sort_values('ts').reset_index(drop=True)
    return df

# ==========================================
# 2. SZYBKIE GENEROWANIE DANYCH DO WYKRESÓW
# ==========================================

def fast_process_tea(df):
    first_occurrence = df.groupby(['u', 'v'])['ts'].min().reset_index()
    first_occurrence['is_new'] = True
    
    df = df.merge(first_occurrence, on=['u', 'v', 'ts'], how='left')
    df['is_new'] = df['is_new'].fillna(False)
    
    tea_stats = df.groupby('ts').agg(
        new=('is_new', 'sum'),
        total_curr_ts=('u', 'count')
    ).reset_index()
    
    tea_stats['repeated'] = tea_stats['total_curr_ts'] - tea_stats['new']
    tea_stats['total_seen_until_curr_ts'] = tea_stats['new'].cumsum()
    
    prev_seen = tea_stats['total_seen_until_curr_ts'].shift(1).fillna(0)
    tea_stats['not_repeated'] = prev_seen - tea_stats['repeated']
    
    return tea_stats.to_dict('records')

def fast_process_tet(df, test_ratio_p=0.85):
    unique_ts = np.sort(df['ts'].unique())
    num_ts = len(unique_ts)
    ts_to_idx = {ts: i for i, ts in enumerate(unique_ts)}
    
    edge_stats = df.groupby(['u', 'v'])['ts'].agg(['min', 'max']).reset_index()
    edge_stats = edge_stats.sort_values(by=['min', 'max'])
    edge_stats['edge_idx'] = np.arange(len(edge_stats))
    
    num_edges = len(edge_stats)
    
    df = df.merge(edge_stats[['u', 'v', 'edge_idx']], on=['u', 'v'])
    
    mat = np.zeros((num_ts, num_edges), dtype=np.int8)
    
    df['row_idx'] = num_ts - df['ts'].map(ts_to_idx) - 1
    
    mat[df['row_idx'].values, df['edge_idx'].values] = E_PRESENCE_GENERAL
    
    split_ts_value_idx = int(np.quantile(np.arange(num_ts), test_ratio_p))
    split_row = num_ts - split_ts_value_idx - 1 
    
    train_view = mat[split_row:, :]
    test_view = mat[:split_row, :]
    
    in_train = (train_view == E_PRESENCE_GENERAL).any(axis=0)
    in_test = (test_view == E_PRESENCE_GENERAL).any(axis=0)
    
    only_train = in_train & ~in_test
    only_test = ~in_train & in_test
    train_and_test = in_train & in_test
    
    train_view[:, only_train] = np.where(train_view[:, only_train] == E_PRESENCE_GENERAL, E_ONLY_TRAIN, E_ABSENT)
    test_view[:, only_test] = np.where(test_view[:, only_test] == E_PRESENCE_GENERAL, E_INDUCTIVE, E_ABSENT)
    
    train_view[:, train_and_test] = np.where(train_view[:, train_and_test] == E_PRESENCE_GENERAL, E_TRAIN_AND_TEST, E_ABSENT)
    test_view[:, train_and_test] = np.where(test_view[:, train_and_test] == E_PRESENCE_GENERAL, E_TRANSDUCTIVE, E_ABSENT)
    
    return mat, split_ts_value_idx, unique_ts, list(range(num_edges))


# ==========================================
# 3. FUNKCJE RYSUJĄCE WYKRESY (BEZ LEGENDY)
# ==========================================

def plot_edges_bar(ts_edges_dist, network_name):
    ts_edges_dist_df = pd.DataFrame(ts_edges_dist)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.25, left=0.15)
    font_size = 18

    timestamps = ts_edges_dist_df['ts'].tolist()
    new = ts_edges_dist_df['new'].tolist()
    repeated = ts_edges_dist_df['repeated'].tolist()

    x_indices = np.arange(len(timestamps))
    
    plt.bar(x_indices, repeated, color='#404040', alpha=0.4, width=1.0) # Usunięto label
    plt.bar(x_indices, new, bottom=repeated, color='#ca0020', alpha=0.8, hatch='//', width=1.0) # Usunięto label
    
    test_split_idx = int(0.85 * len(timestamps))
    if test_split_idx < len(timestamps):
        plt.axvline(x=test_split_idx, color="blue", linestyle="--", linewidth=2)
        plt.text(test_split_idx, 0, 'x', va='center', ha='center', fontsize=font_size, fontweight='heavy', color='blue')

    plt.margins(x=0)
    
    if 'unvote' in network_name.lower() and len(timestamps) > 1:
        tick_indices = [i for i, y in enumerate(timestamps) if y % 10 == 0]
        tick_labels = [int(timestamps[i]) for i in tick_indices]
        plt.xticks(tick_indices, tick_labels, rotation=0, fontsize=14)
    else:
        num_ticks = min(10, len(timestamps))
        if len(timestamps) > 1:
            tick_indices = np.linspace(0, len(timestamps) - 1, num_ticks, dtype=int)
            tick_labels = [int(timestamps[i]) for i in tick_indices]
            plt.xticks(tick_indices, tick_labels, rotation=45, fontsize=14)
    
    label_text = "Timestamp"
    plt.xlabel(label_text, fontsize=font_size)
    plt.ylabel("Number of edges", fontsize=font_size)
    
    # plt.legend() - USUNIĘTO LEGENDĘ
    
    plt.savefig(f"figs/TEA/{network_name}.pdf")
    plt.close()

def plot_edge_presence_matrix(e_presence_mat, test_split_ts_value, unique_ts_list, idx_edge_list, network_name):
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.subplots_adjust(bottom=0.25, left=0.2)

    colors = ['white', '#018571', '#fc8d59', '#fc8d59', '#b2182b']
    ax = sns.heatmap(e_presence_mat, cmap=sns.color_palette(colors, as_cmap=True), cbar=False)

    x_gaps = np.linspace(0, len(idx_edge_list), num=5)
    x_labels = [int(100 * x) for x in (x_gaps / len(idx_edge_list))]
    plt.xticks(x_gaps, x_labels, rotation=0, fontsize=16)

    if 'unvote' in network_name.lower():
        t_gaps = []
        t_labels = []
        for i, y in enumerate(unique_ts_list):
            if y % 10 == 0:
                row_idx = len(unique_ts_list) - i - 1 
                t_gaps.append(row_idx + 0.5) 
                t_labels.append(int(y))
        plt.yticks(t_gaps, t_labels, rotation=0, fontsize=14)
    else:
        t_gaps = np.linspace(0, len(unique_ts_list) - 1, num=5, dtype=int)
        t_labels = [int(unique_ts_list[len(unique_ts_list) - 1 - tidx]) for tidx in t_gaps]
        plt.yticks(t_gaps + 0.5, t_labels, rotation=0, fontsize=14) 

    plt.xlabel("Percentage of observed edges", fontsize=18)
    
    label_text = "Year" if 'unvote' in network_name.lower() else "Timestamp"
    plt.ylabel(label_text, fontsize=18)

    y_length = e_presence_mat.shape[0] - 1
    test_split_idx_value = y_length - test_split_ts_value

    e_border_idx = 0
    for e_idx in range(e_presence_mat.shape[1] - 1, -1, -1):
        if e_presence_mat[y_length - test_split_ts_value, e_idx] != E_ABSENT:
            e_border_idx = e_idx
            break

    rect_train = plt.Rectangle((0, y_length - test_split_ts_value + 0.085), e_border_idx, test_split_ts_value + 0.9, fill=False, linewidth=2, edgecolor="grey")
    rect_test_mayseen = plt.Rectangle((0, 0), e_border_idx, y_length - test_split_ts_value - 0.1, fill=False, linewidth=2, edgecolor="grey")
    rect_test_new = plt.Rectangle((e_border_idx, 0), e_presence_mat.shape[1] - 1 - e_border_idx, y_length - test_split_ts_value - 0.1, fill=False, linewidth=2, edgecolor="grey")
    
    ax.add_patch(rect_train)
    ax.add_patch(rect_test_mayseen)
    ax.add_patch(rect_test_new)

    plt.axhline(y=test_split_idx_value, color="black", linestyle="--", linewidth=2)
    plt.text(x=0, y=test_split_idx_value, s='x', color="black", va='center', ha='center', fontsize=20, fontweight='heavy')

    plt.savefig(f"figs/TET/{network_name}.pdf")
    plt.close()

# ==========================================
# 4. GENEROWANIE NIEZALEŻNYCH LEGEND (DO LATEXA)
# ==========================================

def generate_tea_legend():
    """Generuje osobny plik PDF z legendą dla wykresów TEA (New / Repeated)."""
    fig = plt.figure(figsize=(4, 0.5)) # Puste, malutkie płótno
    p1 = mpatches.Patch(facecolor='#404040', alpha=0.4, label='Repeated')
    p2 = mpatches.Patch(facecolor='#ca0020', alpha=0.8, hatch='//', label='New')
    
    fig.legend(handles=[p1, p2], loc='center', ncol=2, frameon=False, fontsize=16)
    
    # Zapis idealnie przycięty do samej zawartości tekstu legendy
    plt.savefig("figs/TEA/legend.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.close()

def generate_tet_legend():
    """Generuje osobny plik PDF z legendą macierzy TET (Absent / Only Train / Transductive / Inductive)."""
    fig = plt.figure(figsize=(10, 0.5))
    
    p0 = mpatches.Patch(facecolor='white', edgecolor='grey', label='Absent')
    p1 = mpatches.Patch(facecolor='#018571', label='Only in Train')
    p2 = mpatches.Patch(facecolor='#fc8d59', label='Transductive')
    p3 = mpatches.Patch(facecolor='#b2182b', label='Inductive')
    
    fig.legend(handles=[p0, p1, p2, p3], loc='center', ncol=4, frameon=False, fontsize=16)
    
    plt.savefig("figs/TET/legend.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.close()

# ==========================================
# 5. GŁÓWNA PĘTLA WYKONAWCZA
# ==========================================

def main():
    os.makedirs("figs/TEA", exist_ok=True)
    os.makedirs("figs/TET", exist_ok=True)

    # 1. Generowanie legend (tylko raz!)
    print("Generowanie osobnych legend...")
    generate_tea_legend()
    generate_tet_legend()

    # Poprawiony przecinek w liście!
    datasets = ['tgbl-wiki', "Enron", "mooc", "uci", 'UNvote']

    for dataset_name in datasets:
        data = load_network_data(dataset_name)
        interval_size = get_interval_for_dataset(dataset_name)
        
        print(f"Tworzenie ramki danych dla {dataset_name}...")
        df_edges = get_edgelist_dataframe(data, dataset_name, interval_size)
        print(f"Przetwarzam {len(df_edges)} unikalnych interakcji. Obejmuje: {df_edges['ts'].min()} - {df_edges['ts'].max()}")

        print("Obliczanie statystyk TEA...")
        ts_edges_dist = fast_process_tea(df_edges)
        plot_edges_bar(ts_edges_dist, dataset_name)
        
        print("Generowanie macierzy TET...")
        e_presence_mat, split_val, unique_ts, idx_edges = fast_process_tet(df_edges, test_ratio_p=0.85)
        plot_edge_presence_matrix(e_presence_mat, split_val, unique_ts, idx_edges, dataset_name)

        print(f"✅ Sukces! Zapisano błyskawicznie wykresy w formacie PDF dla {dataset_name}.\n")

if __name__ == "__main__":
    main()