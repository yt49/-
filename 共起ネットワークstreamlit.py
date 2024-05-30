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
    # 小文字化
    text = text.lower()
    # アンダーバーを削除
    text = text.replace('_', '')
    # 'x000d'を削除
    text = text.replace('x000d', '')
    # 改行をスペースに置き換え
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
uploaded_file = st.file_uploader("エクセルファイルをアップロードしてください", type=["xlsx"])

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

    # 共起ネットワークのメリットとデメリットを表示
    st.write("## 共起ネットワークとは？（GPTより）")

    st.write("### メリット")
    st.write("""
    1. **可視化がわかりやすい**
       - 単語間の関係を視覚的に示すことで、テキストデータの内容や特徴を直感的に理解できる。
       - ネットワークグラフは、複雑な情報をシンプルに伝えるのに適している。

    2. **トピックの発見**
       - 頻繁に共に出現する単語を通じて、隠れたトピックやテーマを見つけることができる。
       - 特定のトピックやテーマに関連する重要な単語を特定するのに役立つ。

    3. **関係性の分析**
       - 単語同士の関係性や連携を分析することで、文章やデータの深い理解が得られる。
       - 特定の単語がどのような文脈で使われるのかを把握できる。

    4. **コミュニティの発見**
       - ネットワーク内のコミュニティを発見することで、関連性の高い単語のグループを特定できる。
       - トピックごとの単語のクラスタを明らかにするのに役立つ。
    """)

    st.write("### デメリット")
    st.write("""
    1. **複雑な準備作業**
       - データの前処理や分かち書き、共起行列の作成など、事前に必要な作業が多い。
       - 特に日本語などの言語では形態素解析が必要になる場合があり、ツールの選定や設定に手間がかかる。

    2. **計算コスト**
       - データ量が大きくなると、共起行列の計算やネットワークの生成に時間がかかる。
       - 大規模なデータセットでは計算資源を多く消費する可能性がある。

    3. **解釈の難しさ**
       - 複雑なネットワークになると、ノード（単語）やエッジ（関係）の意味を理解するのが難しくなる。
       - 可視化された結果を解釈するには、ある程度の知識や経験が必要になることもある。

    4. **動的なデータに対応しにくい**
       - テキストデータが頻繁に更新される場合、毎回ネットワークを再生成する必要があり手間がかかる。
       - リアルタイムのデータ分析には不向きな場合がある。
    """)
else:
    st.write("エクセルファイルをアップロードしてください。")

if uploaded_file is not None:
    # エクセルファイルの読み込み
    df = pd.read_excel(uploaded_file, sheet_name=None)