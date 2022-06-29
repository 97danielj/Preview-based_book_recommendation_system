""" 7팀 세미 프로젝트 웹앱

"""

# 라이브러리 로드
import json
from typing import List
import pandas as pd
import streamlit as st
from gensim.models import doc2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_analyzed_data() -> List[str]:
    """형태소 분석 데이터를 가져온다"""

    # 파일 경로
    analyzed_data_path = "yes24.2.형태소.nouns.json"

    # 파일 읽음
    with open(analyzed_data_path, encoding='utf-8') as f:

        # 사전 객체로 변환 후 반환
        return [" ".join(x["preview"]) for x in json.load(f)]


def load_preprocessed_data() -> pd.DataFrame:
    """전처리된 데이터를 가져온다"""

    # 파일 경로
    pre_processed_data_path = "yes24.1.전처리.json"

    # DataFrame 객체로 변환 후 반환
    return pd.read_json(pre_processed_data_path)


def load_doc2vec_data():
    """doc2vec 모델을 가져온다"""
    return doc2vec.Doc2Vec.load("dart.doc2vec")


def create_cosine_sim(analyzed_data: List[str]):
    """코사인 유사도 객체 생성"""
    tfidf_matrix = TfidfVectorizer().fit_transform(analyzed_data)
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_tfidf_recommendations(title: str) -> pd.DataFrame:
    """TF-IDF 기반 추천"""
    # 데이터 가져옴
    df = load_preprocessed_data()
    cosine_sim = create_cosine_sim(load_analyzed_data())

    # 선택한 제목에서 해당 책의 인덱스를 받아온다.
    title_to_index = dict(zip(df['title'], df.index))
    idx = title_to_index[title]

    # 해당 책과 모든 책과의 유사도를 가져온다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 책들을 정렬한다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 책을 받아온다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 책의 인덱스를 얻는다.
    novel_indices = [idx[0] for idx in sim_scores]

    # 가장 유사한 10개의 책 제목과 유사도를 반환한다.
    title_list = df['title'].iloc[novel_indices]
    score_list = [round(score[1], 3) * 100 for score in sim_scores]
    return pd.DataFrame({"제목": title_list, "score": score_list}).reset_index(drop=True, inplace=False)


def get_doc2vec_recommendations(title: str) -> pd.DataFrame:
    """doc2vec 기반 추천"""
    # 모델을 불러온다
    model = load_doc2vec_data()

    # 제목과 가장 비슷한 소설 10개를 가져온다.
    similar_doc = model.dv.most_similar(title)

    # 데이터 프레임으로 변환한다.
    df = pd.DataFrame(similar_doc, columns=["제목", "유사도"])

    # 유사도를 소숫점 한자리 퍼센트 수치로 변환한다.
    df["유사도"] = df["유사도"].apply(lambda x: round(x, 3) * 100)

    # 값을 반환한다.
    return df


def set_tfidf_column(col: st.delta_generator.DeltaGenerator, title: str):
    """TF-IDF 열을 설정한다"""
    col.write("TF-IDF:")
    try:
        col.write(get_tfidf_recommendations(title))
    except KeyError:
        col.text("데이터가 없습니다.")
    except IndexError:
        col.text("데이터가 없습니다.")


def set_doc2vec_column(col: st.delta_generator.DeltaGenerator, title: str):
    """Doc2Vec 열을 설정한다"""
    col.write("Doc2Vec:")
    try:
        col.write(get_doc2vec_recommendations(title))
    except KeyError:
        col.text("데이터가 없습니다.")


def set_search_title_column(col):
    """검색 열을 설정한다"""
    df = load_preprocessed_data()
    title = st.text_input("기억이 잘 나지 않는 책 입력하시오.")
    col.write(f"{title}")

    if title:
        col.write([x for x in df.title if title in x])
    else:
        col.write(df.title)


# 제목
st.title("미리보기 기반 소설 추천 서비스")

# 제목 검색창
novel_title = st.text_input("좋아하는 소설 제목을 입력하시오.")
st.write(f"검색 대상: {novel_title}")

# 2개의 열로 나눔
col1, col2 = st.columns(2)

# TF-IDF 열을 설정
set_tfidf_column(col1, novel_title)

# Doc2Vec 열을 설정
set_doc2vec_column(col2, novel_title)

# 검색 열을 설정
set_search_title_column(st)
