import streamlit as st
import pandas as pd
import numpy as np
from janome.tokenizer import Tokenizer
import networkx as nx
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォントのためにインポート
import re
from sklearn.feature_extraction.text import CountVectorizer
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.cm as cm
import spacy

# フォント設定を IPAexGothic に変更
plt.rcParams['font.family'] = 'IPAexGothic'

# spacyの日本語モデルをロード
nlp = spacy.blank("ja")

# 正規化関数
def normalize_text(text):
    text = text.lower()
    text = text.replace('_', '')
    text = text.replace('x000d', '')
    text = text.replace('\n', ' ')
    return text

# 不要なキーワードを削除
def remove_unwanted_keywords(text, unwanted_keywords):
    for keyword in unwanted_keywords:
        text = text.replace(keyword, '')
    return text

# Janomeを使用して分かち書きを行う
def get_words(text, unwanted_keywords):
    text = remove_unwanted_keywords(text, unwanted_keywords)
    t = Tokenizer()
    words = [word.base_form for word in t.tokenize(text)
             if (word.part_of_speech.startswith('名詞') or word.part_of_speech.startswith('動詞'))
             and word.base_form not in unwanted_keywords]
    return ' '.join(words)

# 共起行列を作成する関数
def count_cooccurrence(sents, token_length='{2,}', min_count=5):
    token_pattern = f'\\b\\w{token_length}\\b'
    count_model = CountVectorizer(token_pattern=token_pattern)

    X = count_model.fit_transform(sents)
    words = count_model.get_feature_names_out()
    word_counts = np.asarray(X.sum(axis=0)).reshape(-1)

    # 単語の頻度に基づいてフィルタリング
    frequent_words = words[word_counts >= min_count]
    frequent_indices = np.where(word_counts >= min_count)[0]
    X = X[:, frequent_indices]
    word_counts = word_counts[frequent_indices]

    X[X > 0] = 1
    Xc = (X.T * X)
    return frequent_words, word_counts, Xc, X

# ノードの重みを計算する関数
def word_weights(words, word_counts):
    count_max = word_counts.max()
    weights = [(word, {'weight': count / count_max})
               for word, count in zip(words, word_counts)]
    return weights

# 共起重みを計算する関数
def cooccurrence_weights(words, Xc, weight_cutoff):
    Xc_max = Xc.max()
    cutoff = weight_cutoff * Xc_max
    weights = [(words[i], words[j], Xc[i,j] / Xc_max)
               for i, j in zip(*Xc.nonzero()) if i < j and Xc[i,j] > cutoff]
    return weights

# ネットワークを作成する関数
def create_network(words, word_counts, Xc, weight_cutoff):
    G = nx.Graph()

    weights_w = word_weights(words, word_counts)
    G.add_nodes_from(weights_w)

    weights_c = cooccurrence_weights(words, Xc, weight_cutoff)
    G.add_weighted_edges_from(weights_c)

    G.remove_nodes_from(list(nx.isolates(G)))
    return G

# ネットワークを描画する関数
def pyplot_network(G, layout='spring', layout_parameter_k=0.1):
    plt.figure(figsize=(30, 15), dpi=300)
    if layout == 'spring':
        pos = nx.spring_layout(G, k=layout_parameter_k, iterations=50, threshold=0.0001, weight='weight', scale=1, center=None, dim=2, seed=123)
    elif layout == 'fruchterman_reingold':
        pos = nx.fruchterman_reingold_layout(G, dim=2, k=layout_parameter_k, pos=None, fixed=None, iterations=50, weight='weight', scale=1.0, center=None)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, dist=None, pos=None, weight=1, scale=1, center=None, dim=2)
    else:
        pos = nx.random_layout(G, center=None, dim=2, seed=None)

    connecteds = []
    colors_list = []

    for i, c in enumerate(greedy_modularity_communities(G)):
        connecteds.append(c)
        colors_list.append(1/20 * i)

    # ノードの色をカラーマップを使用して変更
    colors_array = cm.Pastel1(np.linspace(0.1, 0.9, len(colors_list)))

    node_colors = []
    for node in G.nodes():
        for i, c in enumerate(connecteds):
            if node in c:
                node_colors.append(colors_array[i])
                break

    weights_n = np.array(list(nx.get_node_attributes(G, 'weight').values()))
    nx.draw_networkx_nodes(G, pos, alpha=0.3, node_color=node_colors, node_size=7000 * weights_n)

    weights_e = np.array(list(nx.get_edge_attributes(G, 'weight').values()))
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="whitesmoke", width=20 * weights_e)

    nx.draw_networkx_labels(G, pos, font_family='IPAexGothic')

    plt.axis("off")
    st.pyplot(plt)  # Streamlitでプロットを表示

# Streamlit アプリケーション
st.title("共起ネットワークを見てみよう")

# エクセルファイルのアップロード
uploaded_file = st.file_uploader("DGOレビューを集計したエクセルファイルをアップロードしてね", type=["xlsx"])

if uploaded_file is not None:
    # エクセルファイルの読み込み
    df = pd.read_excel(uploaded_file)

    # 不要なキーワード
    unwanted_keywords = ['■全体的な感想', '■デザイン', '■飛距離', '■打感', '■方向性', '■弾道高さ','する','よう','ある','いる','思う','なる','感じる']

    # テキストの前処理
    df['レビューコメント'] = df['レビューコメント'].fillna('').apply(normalize_text)

    # 分かち書き
    df['分かち書き'] = df['レビューコメント'].apply(lambda x: get_words(x, unwanted_keywords))

    # 分かち書きデータから共起行列を作成
    sents = df['分かち書き'].tolist()
    words, word_counts, Xc, X = count_cooccurrence(sents, min_count=5)  # 最低出現回数を設定

    # 重みカットオフの設定
    weight_cutoff = 0.1

    # ネットワークを作成
    G = create_network(words, word_counts, Xc, weight_cutoff)

    # ネットワークを描画
    pyplot_network(G)

    # 単語の出現回数の棒グラフを作成
    # テキストをベクトル化
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['分かち書き'])

    # 語彙とその出現頻度を取得
    vocabulary = vectorizer.vocabulary_
    word_frequencies = X.toarray().sum(axis=0)

    # 出現頻度の高い順にソートした単語リストを作成
    sorted_vocab = sorted(vocabulary.items(), key=lambda x: word_frequencies[vocabulary[x[0]]], reverse=True)

    # 出現回数の多い順に並べる
    df_most_common = pd.DataFrame({'Word': [word for word, _ in sorted_vocab],
                                   'Frequency': [word_frequencies[vocabulary[word]] for word, _ in sorted_vocab]})

    # 上位10件の単語とフレーズを取得
    top_words = df_most_common.head(10)

    # 棒グラフの描画
    plt.figure(figsize=(10, 8))
    plt.barh(top_words['Word'], top_words['Frequency'], color='skyblue')
    plt.xlabel('出現回数', fontsize=14)
    plt.ylabel('単語', fontsize=14)
    plt.title('単語の出現回数', fontsize=16)
    plt.tight_layout()
    st.pyplot(plt)

    # 解説文
    st.markdown("""
    ### 共起ネットワークの解説

    #### 線（エッジ）の太さと色
    - **線の太さ**：線は、2つの単語がどれくらい頻繁に一緒に使われるか（共起頻度）を表しています。頻度が高いほど、線は太く描かれます。
    - **線の色**：すべての線は同じ`whitesmoke`（薄い灰色）色で描かれます。これにより、線の太さで共起頻度の違いが強調されます。

    #### ノード（円）の大きさと色
    - **ノードの大きさ**：ノード（円）は、特定の単語がどれくらい頻繁に出現するかを表しています。頻度が高いほど、ノードは大きく描かれます。
    - **ノードの色**：ノードの色は、ノードが属するコミュニティ（グループ）を示しています。同じ色のノードは、互いに関連が深い単語を表します。異なるコミュニティごとに異なる色が割り当てられます。

    #### コミュニティの色の設定
    - **コミュニティ検出**：`greedy_modularity_communities`関数を使用して、つながりが強いノードのグループ（コミュニティ）を検出します。
    - **色の割り当て**：`cm.Pastel1`カラーマップを使用して、各コミュニティに異なる色を割り当てます。これにより、同じコミュニティ内のノードは同じ色になります。

    #### まとめ
    - **線の太さ**：2つの単語がどれくらい一緒に使われるかを表し、頻度が高いほど太くなります。
    - **線の色**：すべての線は`whitesmoke`色で描かれます。
    - **ノードの大きさ**：単語の出現頻度を表し、頻度が高いほど大きくなります。
    - **ノードの色**：同じ色のノードは、互いに関連が深い単語を表し、コミュニティごとに異なる色が割り当てられます。

    この解説を参考に、ネットワーク内の単語の関係性を視覚的に理解してみてください。
    """)
